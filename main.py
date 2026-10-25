"""
PETG Hexapod Vacuum & Labeled Telemetry Stair-Climbing Simulator
Controls:
  - SPACE : Pause / Resume
  - G     : Cycle Standard Gaits (Wave -> Tripod -> Ripple)
  - Z     : Spawn Stairs & Execute Sequential Stair Climb
  - R     : Restart Simulation (Resets to Wave Gait & Start Position)
  - Slider: Include/Exclude Vacuum System Weight (+180g)
  - Slider: Live Payload / Cargo Mass (Up to 2.0 kg)
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Physical & Material Parameters ---
BASE_MASS_KG = 2.2  
GRAVITY = -9.81

L_COXA  = 0.035
L_FEMUR = 0.095  
L_TIBIA = 0.105  

LINK_RADIUS = 0.010
JOINT_RADIUS = 0.014
FOOT_RADIUS = 0.018  

# Motor Specifications (MG996R Servo)
MG996R_STALL_KGCM = 11.0    
MOTOR_SPEC_TEXT = "Servo: MG996R | Stall Torque: 11.0 kg·cm @ 6.0V | Speed: 0.17 s/60°"

BODY_LENGTH = 0.240
BODY_WIDTH = 0.180
BODY_MIN_HEIGHT = 0.040
BODY_TARGET_HEIGHT = 0.110  

SPAWN_X_START = 0.0 

LEG_HIPS_REL = {
    "FR": [ 0.090, -0.070, 0.0, -45.0],  
    "MR": [ 0.000, -0.090, 0.0, -90.0],  
    "RR": [-0.090, -0.070, 0.0, -135.0], 
    "FL": [ 0.090,  0.070, 0.0,  45.0],  
    "ML": [ 0.000,  0.090, 0.0,  90.0],  
    "RL": [-0.090,  0.070, 0.0,  135.0], 
}

GAIT_NAMES = ["WAVE", "TRIPOD", "RIPPLE"]
GAIT_CONFIGS = {
    # swing_ratio must stay below the leg-to-leg phase spacing (1/6 ~= 0.167 for
    # Wave/Ripple, which stagger all 6 legs individually) or consecutive legs'
    # swing windows overlap and more than one leg is airborne at a time —
    # defeating the point of these gaits. Tripod's 3-leg groups share a single
    # offset by design, so its swing_ratio isn't bound by that constraint.
    "WAVE":   {"offsets": {"RR": 0.000, "MR": 0.166, "FR": 0.333, "RL": 0.500, "ML": 0.666, "FL": 0.833}, "swing_ratio": 0.15},
    "TRIPOD": {"offsets": {"FR": 0.000, "ML": 0.000, "RR": 0.000, "FL": 0.500, "MR": 0.500, "RL": 0.500}, "swing_ratio": 0.30},
    "RIPPLE": {"offsets": {"FR": 0.000, "MR": 0.333, "RR": 0.666, "FL": 0.500, "ML": 0.833, "RL": 0.166}, "swing_ratio": 0.15},
}

LEG_COLORS = {
    "FR": (1.00, 0.30, 0.30, 1.0),
    "MR": (1.00, 0.60, 0.10, 1.0),
    "RR": (1.00, 0.90, 0.10, 1.0),
    "FL": (0.20, 0.80, 1.00, 1.0),
    "ML": (0.60, 0.40, 1.00, 1.0),
    "RL": (0.20, 0.95, 0.40, 1.0),
}

NOMINAL_REACH = 0.105  
MAX_SAFE_REACH = 0.170 
LIFT_HEIGHT = 0.060    
STAIR_LIFT_HEIGHT = 0.080 

FORWARD_SPEED = 0.0006 
TICKS_PER_CYCLE = 200
STRIDE_LENGTH = FORWARD_SPEED * TICKS_PER_CYCLE

PRE_LIFTOFF_DELAY = 30  
LIFTOFF_STEPS = 90     
HISTORY_LEN = 120

def get_surface_height_at(x, y, ignore_id):
    ray_from = [x, y, 0.4]
    ray_to = [x, y, -0.1]
    results = p.rayTest(ray_from, ray_to)
    for res in results:
        obj_id, _, _, hit_pos, _ = res
        if obj_id != -1 and obj_id != ignore_id:
            return hit_pos[2]
    return 0.0

class HexapodKinematicEngine:
    def __init__(self):
        pass

    def solve_ik_cartesian(self, body_pos, body_rot_mat, hip_local, base_yaw_deg, target_foot_world, total_mass, grounded_legs=5):
        hip_world = body_pos + body_rot_mat @ hip_local
        target_body_frame = body_rot_mat.T @ (target_foot_world - hip_world)
        
        xy_dist = np.linalg.norm(target_body_frame[:2])
        if xy_dist > MAX_SAFE_REACH:
            target_body_frame[:2] = (target_body_frame[:2] / xy_dist) * MAX_SAFE_REACH
            
        yaw_rad = np.radians(base_yaw_deg)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw_inv = np.array([
            [cy,  sy, 0],
            [-sy, cy, 0],
            [0,   0,  1]
        ])
        
        local_x, local_y, local_z = R_yaw_inv @ target_body_frame

        coxa_rad = np.arctan2(local_y, local_x)
        r_xy = np.sqrt(local_x**2 + local_y**2) - L_COXA
        r_xy = np.clip(r_xy, 0.02, L_FEMUR + L_TIBIA - 0.01)
        
        d = np.sqrt(r_xy**2 + local_z**2)
        d = np.clip(d, 1e-4, L_FEMUR + L_TIBIA - 1e-4)

        cos_alpha2 = np.clip((L_FEMUR**2 + d**2 - L_TIBIA**2) / (2 * L_FEMUR * d), -1.0, 1.0)
        femur_rad = np.arctan2(local_z, r_xy) + np.arccos(cos_alpha2)
        
        cos_beta = np.clip((L_FEMUR**2 + L_TIBIA**2 - d**2) / (2 * L_FEMUR * L_TIBIA), -1.0, 1.0)
        tibia_rad = np.arccos(cos_beta) - np.pi

        force_z_n = ((total_mass * abs(GRAVITY)) / max(1, grounded_legs)) * 0.70
        coxa_torque_kgcm = (force_z_n * (abs(local_y) + 0.015 * abs(np.sin(coxa_rad)))) * 10.197 * 0.60
        femur_torque_kgcm = (force_z_n * r_xy * (1.0 + 0.03 * np.cos(coxa_rad))) * 10.197 * 0.65
        tibia_torque_kgcm = (force_z_n * (L_TIBIA * np.sin(abs(tibia_rad)) + 0.015 * r_xy)) * 10.197 * 0.60

        return (coxa_rad, femur_rad, tibia_rad), coxa_torque_kgcm, femur_torque_kgcm, tibia_torque_kgcm

def solve_fk_points(body_pos, body_rot_mat, hip_local, base_yaw_deg, angles):
    coxa_rad, femur_rad, tibia_rad = angles
    cc, sc = np.cos(coxa_rad), np.sin(coxa_rad)
    cf, sf = np.cos(femur_rad), np.sin(femur_rad)
    cft, sft = np.cos(femur_rad + tibia_rad), np.sin(femur_rad + tibia_rad)

    coxa_pt_leg = np.array([L_COXA * cc, L_COXA * sc, 0.0])
    knee_pt_leg = coxa_pt_leg + np.array([L_FEMUR * cc * cf, L_FEMUR * sc * cf, L_FEMUR * sf])
    foot_pt_leg = knee_pt_leg + np.array([L_TIBIA * cc * cft, L_TIBIA * sc * cft, L_TIBIA * sft])

    yaw_rad = np.radians(base_yaw_deg)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([[cy, -sy, 0], [sy,  cy, 0], [0,   0,  1]])

    hip_world  = body_pos + body_rot_mat @ hip_local
    coxa_world = body_pos + body_rot_mat @ (hip_local + R_yaw @ coxa_pt_leg)
    knee_world = body_pos + body_rot_mat @ (hip_local + R_yaw @ knee_pt_leg)
    foot_world = body_pos + body_rot_mat @ (hip_local + R_yaw @ foot_pt_leg)

    return hip_world, coxa_world, knee_world, foot_world

def _make_capsule(radius, length, rgba):
    col = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)

def _make_sphere(radius, rgba):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)

def _segment_pose(p1, p2):
    mid = (p1 + p2) / 2.0
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-9: return mid, [0, 0, 0, 1], 0.0
    z_axis = np.array([0.0, 0.0, 1.0])
    v = vec / length
    axis = np.cross(z_axis, v)
    axis_norm = np.linalg.norm(axis)
    dot = np.clip(np.dot(z_axis, v), -1.0, 1.0)
    if axis_norm < 1e-9:
        quat = [0, 0, 0, 1] if dot > 0 else p.getQuaternionFromAxisAngle([1, 0, 0], np.pi)
    else:
        quat = p.getQuaternionFromAxisAngle((axis / axis_norm).tolist(), np.arccos(dot))
    return mid, quat, length

class LegRig:
    def __init__(self, color):
        joint_rgba = (0.3, 0.3, 0.3, 1.0)
        spherical_foot_rgba = (0.8, 0.8, 0.85, 1.0)  
        self.link_ids = [
            _make_capsule(LINK_RADIUS, L_COXA, color),
            _make_capsule(LINK_RADIUS, L_FEMUR, color),
            _make_capsule(LINK_RADIUS * 0.85, L_TIBIA, color),
        ]
        self.joint_ids = [
            _make_sphere(JOINT_RADIUS, joint_rgba),
            _make_sphere(JOINT_RADIUS * 0.85, joint_rgba),
            _make_sphere(FOOT_RADIUS, spherical_foot_rgba), 
        ]

    def update(self, hip_w, coxa_w, knee_w, foot_w):
        for body_id, (p1, p2) in zip(self.link_ids, ((hip_w, coxa_w), (coxa_w, knee_w), (knee_w, foot_w))):
            mid, quat, length = _segment_pose(p1, p2)
            if length >= 1e-9:
                p.resetBasePositionAndOrientation(body_id, mid.tolist(), quat)
        for body_id, pt in zip(self.joint_ids, (hip_w, coxa_w, foot_w)):
            p.resetBasePositionAndOrientation(body_id, pt.tolist(), [0, 0, 0, 1])

def init_simulation():
    p.connect(p.GUI, options="--width=960 --height=1000")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.resetDebugVisualizerCamera(0.60, 45, -35, [0.3, 0, 0.15])
    
    p.loadURDF("plane.urdf")
    p.changeVisualShape(0, -1, rgbaColor=[0.1, 0.1, 0.12, 1])
    
    c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BODY_LENGTH / 2, BODY_WIDTH / 2, 0.012])
    c_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[BODY_LENGTH / 2, BODY_WIDTH / 2, 0.012], rgbaColor=[0.2, 0.2, 0.25, 1])
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, baseVisualShapeIndex=c_vis, basePosition=[SPAWN_X_START, 0, BODY_MIN_HEIGHT])
    
    btn_pause = p.addUserDebugParameter("Pause / Resume [SPACE]", 1, 0, 0)
    btn_gait = p.addUserDebugParameter("Switch Gait [G]", 1, 0, 0)
    btn_restart = p.addUserDebugParameter("Restart Sim / Wave [R]", 1, 0, 0)
    btn_vacuum = p.addUserDebugParameter("Include Vacuum System (0=No, 1=Yes)", 0.0, 1.0, 1.0)
    btn_payload = p.addUserDebugParameter("Payload / Cargo Mass (kg)", 0.0, 2.0, 0.0) 
    
    return body_id, btn_pause, btn_gait, btn_restart, btn_vacuum, btn_payload

def spawn_absolute_stairs():
    stair_bodies = []
    stair1_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.30, 0.6, 0.03])
    stair1_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.30, 0.6, 0.03], rgbaColor=[0.35, 0.35, 0.40, 1])
    stair_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=stair1_col, baseVisualShapeIndex=stair1_vis, basePosition=[0.60, 0.0, 0.03]))
    
    stair2_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.30, 0.6, 0.06])
    stair2_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.30, 0.6, 0.06], rgbaColor=[0.45, 0.45, 0.50, 1])
    stair_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=stair2_col, baseVisualShapeIndex=stair2_vis, basePosition=[1.20, 0.0, 0.06]))
    return stair_bodies

def init_telemetry_plot():
    plt.style.use('dark_background')
    plt.ion()
    fig, (ax_coxa, ax_femur, ax_tibia, ax_clearance) = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.canvas.manager.set_window_title("18-Servo Telemetry Dashboard with Motor Specs")
    fig.patch.set_facecolor('#181818')

    try:
        mngr = plt.get_current_fig_manager()
        mngr.window.geometry("960x1000+960+0")
    except Exception:
        pass
    
    # Add Motor Spec Header on the figure
    fig.suptitle(MOTOR_SPEC_TEXT, color='orange', fontsize=10, fontweight='bold', y=0.98)

    # Subplot 1: Coxa Torque
    ax_coxa.set_facecolor('#101010')
    ax_coxa.set_title("J1 Coxa Joint Torque Load", color='white', fontsize=10, pad=4)
    ax_coxa.set_ylabel("Torque (kg·cm)", color='lightgray', fontsize=9)
    ax_coxa.set_ylim(0, MG996R_STALL_KGCM + 2.0)
    ax_coxa.axhline(y=MG996R_STALL_KGCM, color='#ff3333', linestyle='--', linewidth=1.2, label='MG996R Stall Limit (11 kg·cm)')
    ax_coxa.legend(loc='upper left', facecolor='#181818', edgecolor='none', fontsize=7)
    ax_coxa.grid(True, color='#282828')

    # Subplot 2: Femur Torque
    ax_femur.set_facecolor('#101010')
    ax_femur.set_title("J2 Femur Joint Torque Load", color='white', fontsize=10, pad=4)
    ax_femur.set_ylabel("Torque (kg·cm)", color='lightgray', fontsize=9)
    ax_femur.set_ylim(0, MG996R_STALL_KGCM + 2.0)
    ax_femur.axhline(y=MG996R_STALL_KGCM, color='#ff3333', linestyle='--', linewidth=1.2, label='MG996R Stall Limit')
    ax_femur.grid(True, color='#282828')

    # Subplot 3: Tibia Torque
    ax_tibia.set_facecolor('#101010')
    ax_tibia.set_title("J3 Tibia Joint Torque Load", color='white', fontsize=10, pad=4)
    ax_tibia.set_ylabel("Torque (kg·cm)", color='lightgray', fontsize=9)
    ax_tibia.set_ylim(0, MG996R_STALL_KGCM + 2.0)
    ax_tibia.axhline(y=MG996R_STALL_KGCM, color='#ff3333', linestyle='--', linewidth=1.2, label='MG996R Stall Limit')
    ax_tibia.grid(True, color='#282828')

    # Subplot 4: Foot Clearance / Height Profile
    ax_clearance.set_facecolor('#101010')
    ax_clearance.set_title("Foot Z-Height Trajectories (Terrain Tracking)", color='white', fontsize=10, pad=4)
    ax_clearance.set_ylabel("Height (m)", color='lightgray', fontsize=9)
    ax_clearance.set_xlabel("Simulation Steps (Ticks)", color='lightgray', fontsize=9)
    ax_clearance.set_ylim(-0.005, STAIR_LIFT_HEIGHT + 0.12)
    ax_clearance.grid(True, color='#282828')

    time_hist = deque(maxlen=HISTORY_LEN)
    c_hist = {name: deque(maxlen=HISTORY_LEN) for name in LEG_HIPS_REL}
    f_hist = {name: deque(maxlen=HISTORY_LEN) for name in LEG_HIPS_REL}
    t_hist = {name: deque(maxlen=HISTORY_LEN) for name in LEG_HIPS_REL}
    h_hist = {name: deque(maxlen=HISTORY_LEN) for name in LEG_HIPS_REL}

    c_lines, f_lines, t_lines, h_lines = {}, {}, {}, {}

    for name in LEG_HIPS_REL:
        color = LEG_COLORS[name][:3]
        c_lines[name], = ax_coxa.plot([], [], label=name, color=color, linewidth=1.5)
        f_lines[name], = ax_femur.plot([], [], label=name, color=color, linewidth=1.5)
        t_lines[name], = ax_tibia.plot([], [], label=name, color=color, linewidth=1.5)
        h_lines[name], = ax_clearance.plot([], [], label=name, color=color, linewidth=1.5)

    ax_femur.legend(loc='upper right', ncol=3, facecolor='#181818', edgecolor='none', fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig, ax_coxa, time_hist, c_hist, f_hist, t_hist, h_hist, c_lines, f_lines, t_lines, h_lines

def main():
    body_id, btn_pause, btn_gait, btn_restart, btn_vacuum, btn_payload = init_simulation()
    ik_engine = HexapodKinematicEngine()
    leg_rigs = {name: LegRig(LEG_COLORS[name]) for name in LEG_HIPS_REL}

    hud_id = p.addUserDebugText(text="INITIALIZING...", textPosition=[0, 0, 0.2], textColorRGB=[1, 1, 1], textSize=2.0)
    poly_lines = []

    fig, ax_c, time_hist, c_hist, f_hist, t_hist, h_hist, c_lines, f_lines, t_lines, h_lines = init_telemetry_plot()

    step_count = 0
    walk_step = 0
    is_paused = False
    gait_idx = 0
    current_gait_name = GAIT_NAMES[gait_idx]
    stair_bodies = []
    
    body_z = BODY_MIN_HEIGHT
    body_x = SPAWN_X_START
    body_pitch = 0.0

    world_foot_anchors = {}
    is_swinging = {name: False for name in LEG_HIPS_REL}
    swing_start = {}
    swing_target = {}
    anchors_initialized = False

    last_pause_clicks = p.readUserDebugParameter(btn_pause)
    last_gait_clicks = p.readUserDebugParameter(btn_gait)
    last_restart_clicks = p.readUserDebugParameter(btn_restart)

    while p.isConnected() and plt.fignum_exists(fig.number):
        try:
            do_restart = False
            
            p_clicks = p.readUserDebugParameter(btn_pause)
            if p_clicks != last_pause_clicks:
                is_paused = not is_paused
                last_pause_clicks = p_clicks

            g_clicks = p.readUserDebugParameter(btn_gait)
            if g_clicks != last_gait_clicks:
                gait_idx = (gait_idx + 1) % len(GAIT_NAMES)
                current_gait_name = GAIT_NAMES[gait_idx]
                last_gait_clicks = g_clicks

            r_clicks = p.readUserDebugParameter(btn_restart)
            if r_clicks != last_restart_clicks:
                do_restart = True
                last_restart_clicks = r_clicks

            vacuum_toggle = p.readUserDebugParameter(btn_vacuum)
            vacuum_mass_kg = 0.18 if vacuum_toggle >= 0.5 else 0.0
            payload_mass = p.readUserDebugParameter(btn_payload)
            current_total_mass = BASE_MASS_KG + vacuum_mass_kg + payload_mass

            keys = p.getKeyboardEvents()
            if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):
                is_paused = not is_paused
            if ord('g') in keys and (keys[ord('g')] & p.KEY_WAS_TRIGGERED):
                gait_idx = (gait_idx + 1) % len(GAIT_NAMES)
                current_gait_name = GAIT_NAMES[gait_idx]
            if ord('z') in keys and (keys[ord('z')] & p.KEY_WAS_TRIGGERED):
                if not stair_bodies:
                    stair_bodies = spawn_absolute_stairs()
            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                do_restart = True

            if do_restart:
                step_count = 0
                walk_step = 0
                is_paused = False
                gait_idx = 0
                current_gait_name = GAIT_NAMES[gait_idx]
                body_z = BODY_MIN_HEIGHT
                body_x = SPAWN_X_START
                body_pitch = 0.0
                world_foot_anchors.clear()
                anchors_initialized = False
                is_swinging = {name: False for name in LEG_HIPS_REL}
                swing_start.clear()
                swing_target.clear()
                for sb in stair_bodies:
                    try: p.removeBody(sb)
                    except: pass
                stair_bodies.clear()
                time_hist.clear()
                for name in LEG_HIPS_REL:
                    c_hist[name].clear()
                    f_hist[name].clear()
                    t_hist[name].clear()
                    h_hist[name].clear()
                p.resetBasePositionAndOrientation(body_id, [SPAWN_X_START, 0, BODY_MIN_HEIGHT], [0, 0, 0, 1])
                continue

            if is_paused:
                time.sleep(0.05)
                continue

            if not anchors_initialized:
                for name, hip_data in LEG_HIPS_REL.items():
                    r_nom = L_COXA + NOMINAL_REACH
                    yaw_rad = np.radians(hip_data[3])
                    foot_x = SPAWN_X_START + hip_data[0] + r_nom * np.cos(yaw_rad)
                    foot_y = hip_data[1] + r_nom * np.sin(yaw_rad)
                    ground_z = get_surface_height_at(foot_x, foot_y, body_id)
                    world_foot_anchors[name] = np.array([foot_x, foot_y, ground_z])
                anchors_initialized = True

            if step_count < PRE_LIFTOFF_DELAY + LIFTOFF_STEPS:
                if step_count < PRE_LIFTOFF_DELAY:
                    body_z = BODY_MIN_HEIGHT
                    state_text = f"STANDBY | MASS: {current_total_mass:.2f}kg"
                else:
                    progress = (step_count - PRE_LIFTOFF_DELAY) / float(LIFTOFF_STEPS)
                    body_z = BODY_MIN_HEIGHT + (BODY_TARGET_HEIGHT - BODY_MIN_HEIGHT) * np.sin(progress * np.pi / 2.0)
                    state_text = f"LIFTING ({int(progress*100)}%)"
                
                body_pitch = 0.0
                target_pitch = 0.0
            else:
                state_text = f"ADAPTIVE GAIT: {current_gait_name}"
                body_x += FORWARD_SPEED
                
                cycle_time = (walk_step / float(TICKS_PER_CYCLE)) % 1.0
                active_gait = GAIT_CONFIGS[current_gait_name]
                swing_ratio = active_gait["swing_ratio"]
                # Use the taller stair clearance once stairs are in the scene, so the
                # foot arc actually clears the step edge instead of the flat-ground height.
                lift_height = STAIR_LIFT_HEIGHT if stair_bodies else LIFT_HEIGHT
                
                for name, hip_data in LEG_HIPS_REL.items():
                    leg_phase = (cycle_time + active_gait["offsets"][name]) % 1.0
                    
                    if leg_phase < swing_ratio:
                        if not is_swinging[name]:
                            is_swinging[name] = True
                            swing_start[name] = world_foot_anchors[name].copy()
                            
                            r_nom = L_COXA + NOMINAL_REACH
                            yaw_rad = np.radians(hip_data[3])
                            future_body_x = body_x + STRIDE_LENGTH * 0.5 
                            target_x = future_body_x + hip_data[0] + r_nom * np.cos(yaw_rad)
                            target_y = hip_data[1] + r_nom * np.sin(yaw_rad)
                            
                            target_z = get_surface_height_at(target_x, target_y, body_id)
                            swing_target[name] = np.array([target_x, target_y, target_z])
                            
                        s = leg_phase / swing_ratio
                        cur_x = swing_start[name][0] + (swing_target[name][0] - swing_start[name][0]) * s
                        cur_y = swing_start[name][1] + (swing_target[name][1] - swing_start[name][1]) * s
                        cur_z = swing_start[name][2] + (swing_target[name][2] - swing_start[name][2]) * s
                        
                        arc_z = lift_height * np.sin(s * np.pi)
                        world_foot_anchors[name] = np.array([cur_x, cur_y, cur_z + arc_z])
                        
                    else:
                        if is_swinging[name]:
                            is_swinging[name] = False
                            world_foot_anchors[name] = swing_target[name].copy()

                front_stance_zs = []
                rear_stance_zs = []
                all_stance_zs = []
                
                for name in LEG_HIPS_REL:
                    if not is_swinging[name]:
                        z = world_foot_anchors[name][2]
                        all_stance_zs.append(z)
                        if 'F' in name: front_stance_zs.append(z)
                        if 'R' in name: rear_stance_zs.append(z)
                        
                if all_stance_zs:
                    target_body_z = np.mean(all_stance_zs) + BODY_TARGET_HEIGHT
                else:
                    target_body_z = BODY_TARGET_HEIGHT
                
                if front_stance_zs and rear_stance_zs:
                    avg_front_z = np.mean(front_stance_zs)
                    avg_rear_z = np.mean(rear_stance_zs)
                    pitch_angle = -np.arctan2(avg_front_z - avg_rear_z, 0.18) 
                    target_pitch = pitch_angle
                else:
                    target_pitch = 0.0

                body_z += (target_body_z - body_z) * 0.08
                body_pitch += (target_pitch - body_pitch) * 0.05
                walk_step += 1

            body_pos_arr = np.array([body_x, 0.0, body_z])
            body_orn_quat = p.getQuaternionFromEuler([0.0, body_pitch, 0.0])
            body_rot_mat = np.array(p.getMatrixFromQuaternion(body_orn_quat)).reshape(3, 3)
            
            p.resetBasePositionAndOrientation(body_id, body_pos_arr, body_orn_quat)
            p.resetDebugVisualizerCamera(cameraDistance=0.60, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=body_pos_arr)

            time_hist.append(step_count)
            # Determine the FULL grounded set for this timestep before computing any
            # leg's torque — computing it incrementally inside the leg loop below
            # under-counted grounded legs for whichever legs were processed first
            # (FR, MR, RR), inflating their torque relative to legs processed later.
            grounded_feet = [name for name in LEG_HIPS_REL if not is_swinging.get(name, False)]
            n_grounded = max(1, len(grounded_feet))
            total_torque = 0.0
            foot_positions = {}

            for name, hip_data in LEG_HIPS_REL.items():
                hip_local = np.array([hip_data[0], hip_data[1], hip_data[2]])
                base_yaw = hip_data[3]
                target_foot = world_foot_anchors[name]

                angles, coxa_t, femur_t, tibia_t = ik_engine.solve_ik_cartesian(
                    body_pos_arr, body_rot_mat, hip_local, base_yaw, target_foot, current_total_mass, grounded_legs=n_grounded
                )
                
                pts = solve_fk_points(body_pos_arr, body_rot_mat, hip_local, base_yaw, angles)
                leg_rigs[name].update(*pts)

                foot_positions[name] = pts[3]
                c_hist[name].append(coxa_t)
                f_hist[name].append(femur_t)
                t_hist[name].append(tibia_t)
                h_hist[name].append(target_foot[2])
                total_torque += femur_t

            if step_count % 15 == 0:
                hud_text = f"{state_text}\nMASS: {current_total_mass:.2f}kg | PITCH: {np.degrees(body_pitch):.1f}°"
                hud_id = p.addUserDebugText(
                    text=hud_text,
                    textPosition=[body_pos_arr[0] - 0.12, body_pos_arr[1], body_pos_arr[2] + 0.16],
                    textColorRGB=[1.0, 1.0, 1.0],
                    textSize=1.8,
                    replaceItemUniqueId=hud_id
                )

                for line_id in poly_lines: p.removeUserDebugItem(line_id)
                poly_lines.clear()
                if len(grounded_feet) > 1:
                    for i in range(len(grounded_feet)):
                        f1, f2 = grounded_feet[i], grounded_feet[(i + 1) % len(grounded_feet)]
                        p1 = foot_positions[f1].tolist()
                        p2 = foot_positions[f2].tolist()
                        poly_lines.append(p.addUserDebugLine(p1, p2, lineColorRGB=[0, 1, 0.5], lineWidth=2.0))

            if step_count % 20 == 0:
                ax_c.set_xlim(max(0, step_count - HISTORY_LEN), max(HISTORY_LEN, step_count))
                for name in LEG_HIPS_REL:
                    c_lines[name].set_data(time_hist, c_hist[name])
                    f_lines[name].set_data(time_hist, f_hist[name])
                    t_lines[name].set_data(time_hist, t_hist[name])
                    h_lines[name].set_data(time_hist, h_hist[name])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            p.stepSimulation()
            step_count += 1

        except Exception as e:
            import traceback
            print(f"Physics Engine Reset or Error Handled: {e}")
            traceback.print_exc()
            break

    plt.close('all')
    if p.isConnected(): p.disconnect()

if __name__ == "__main__":
    main()