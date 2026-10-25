# hexapod-simulation
PyBullet simulation environment for an 18-DOF hexapod robot. Includes inverse kinematics (IK), Wave/Tripod/Ripple gait generation, and servo torque analysis.
18-DOF Hexapod Simulation 🕷️

A physics-based simulation environment built in PyBullet to develop, test, and visualize inverse kinematics, walking gaits, and stair-climbing capabilities for an 18-DOF PETG hexapod robot.

🚀 Overview

This repository contains the Python simulation code (wave_sim_fixed.py) used to validate the robot's movements before deploying to physical hardware. It features a live telemetry dashboard that tracks joint torque loads against the physical limits of MG996R servos, ensuring the physical build will not fail under load.

Key Features

Inverse Kinematics (IK) Engine: Calculates precise Cartesian coordinates for 3-DOF legs: Coxa (35mm), Femur (95mm), and Tibia (105mm).

Dynamic Gait Generation: Simulates and cycles between Wave, Tripod, and Ripple walking patterns.

Live Telemetry Dashboard: Uses Matplotlib to plot real-time torque loads for all 18 servos and compares them against the MG996R stall torque limit (11.0 kg·cm).

Terrain Tracking & Stair Climbing: Adaptive foot placement for navigating uneven terrain and a dedicated stair-climbing sequence.

Adjustable Mass Simulation: Dynamically calculates center of gravity and torque with a 2.2 kg base weight, an optional 180g vacuum system, and up to a 2.0 kg payload.

🛠️ Dependencies

Ensure you have Python installed (Anaconda recommended). You will need the following libraries:

pip install pybullet numpy matplotlib


💻 Setup & Execution

Clone this repository:

git clone https://github.com/zaeem-mohd/hexapod-simulation.git
cd hexapod-simulation


Run the simulation using your Anaconda prompt or standard terminal:

python wave_sim_fixed.py


🎮 Simulation Controls

While the PyBullet window is active, you can use the following keyboard controls and UI sliders:

SPACE : Pause / Resume the simulation.

G : Cycle through walking gaits (Wave -> Tripod -> Ripple).

Z : Spawn stairs and execute the sequential stair-climbing maneuver.

R : Restart the simulation (resets to Wave gait and start position).

UI Sliders: Toggle the Vacuum System weight or adjust the live Payload/Cargo mass.
