# AI-Powered Robot Arm Learning from Human Demonstration

## Overview

This project explores a vision-based approach to robot-arm motion generation using human demonstrations. It captures human hand movements through a webcam, extracts hand landmarks, processes the recorded trajectory, and generates suitable robot movements for simulation.

The system integrates Computer Vision, Machine Learning, Robotics, and Web Development.

## Key Features

- Real-time hand tracking using MediaPipe.
- Extraction of 21 hand landmarks.
- Human demonstration recording.
- Landmark data processing and trajectory generation.
- Behavioural Cloning using PyTorch.
- Robot inverse kinematics.
- ROS 2-based robot integration.
- Gazebo robot simulation.
- Web-based monitoring dashboard.

## Technology Stack

| Category | Technology |
|---|---|
| Operating System | Ubuntu 24.04 |
| Programming Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Machine Learning | PyTorch |
| Numerical Processing | NumPy |
| Backend | FastAPI |
| Frontend | React.js, Vite |
| Robotics Middleware | ROS 2 Jazzy |
| Simulation | Gazebo Sim 8 |
| Robot Description | URDF |

## System Workflow

1. Capture human hand movement through a webcam.
2. Detect 21 hand landmarks using MediaPipe.
3. Record the landmark coordinates over time.
4. Validate and process the demonstration.
5. Generate robot-reachable targets.
6. Calculate robot joint configurations using inverse kinematics.
7. Validate the robot trajectory.
8. Execute and visualize the movement in Gazebo.

## Project Structure

```text
human_action_cloning_bot/
├── backend/
├── frontend/
├── models/
│   ├── mediapipe/
│   └── behavior_cloning_model.pth
├── ros2_ws/
│   └── src/
│       └── human_robot_sim/
├── scripts/
├── src/
├── tests/
├── .gitignore
└── README.md
```

## Running the Project

### 1. Start the Robot Simulation

```bash
source /opt/ros/jazzy/setup.bash
cd ~/human_action_cloning_bot/ros2_ws
colcon build
source install/setup.bash
ros2 launch human_robot_sim robot_sim.launch.py
```

### 2. Start the Backend

```bash
cd ~/human_action_cloning_bot
source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### 3. Start the Frontend

```bash
cd ~/human_action_cloning_bot/frontend
npm run dev
```

Open the dashboard at:

`http://127.0.0.1:5173`

**Note:** These commands assume the ROS 2 package builds successfully in the included workspace. The backend and frontend commands assume their dependencies have already been installed.

## Project Objective

The objective is to explore how human demonstrations can be used to generate robot-arm movements, reducing dependence on manually specifying every robot action.

## Future Enhancements

- Improved camera calibration.
- More reliable human-to-robot coordinate mapping.
- Improved Behavioural Cloning training.
- Collision avoidance.
- Physical robot-arm integration.

## Developer

G. Muthulakshmi  
B.Tech – Artificial Intelligence and Machine Learning  
Sri Shakthi Institute of Engineering and Technology
