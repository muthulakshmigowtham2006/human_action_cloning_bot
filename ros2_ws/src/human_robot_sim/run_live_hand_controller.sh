#!/bin/bash

source /opt/ros/jazzy/setup.bash

source ~/human_action_cloning_bot/venv/bin/activate

source ~/robot_ws/install/setup.bash

exec python ~/robot_ws/install/human_robot_sim/lib/python3.12/site-packages/human_robot_sim/live_hand_robot_controller.py
