#!/usr/bin/env bash

PROJECT_ROOT="/home/muthulakshmi/human_action_cloning_bot"
ROBOT_WS="/home/muthulakshmi/robot_ws"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PATH="/home/muthulakshmi/.local/node/bin:$PATH"
export DISPLAY="${DISPLAY:-:0}"

echo "============================================================"
echo " AI-Powered Robot Arm Learning from Human Demonstration"
echo " Starting Demonstration Environment..."
echo "============================================================"

# 1. Source ROS 2 Jazzy & Workspace
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "[ERROR] /opt/ros/jazzy/setup.bash not found!"
    exit 1
fi

if [ -f "$ROBOT_WS/install/setup.bash" ]; then
    source "$ROBOT_WS/install/setup.bash"
else
    echo "[WARN] $ROBOT_WS/install/setup.bash not found. Building workspace..."
    (cd "$ROBOT_WS" && colcon build --packages-select human_robot_sim)
    source "$ROBOT_WS/install/setup.bash"
fi

# 2. Check / Start Gazebo Simulation (unless --no-gazebo passed)
if [[ "$1" != "--no-gazebo" ]]; then
    if pgrep -f "robot_sim.launch.py" >/dev/null 2>&1 || pgrep -f "gz sim" >/dev/null 2>&1; then
        echo "[OK] Gazebo simulation is already running."
    else
        echo "[INFO] Starting Gazebo Sim 8 & ROS 2 controllers (robot_sim.launch.py)..."
        setsid ros2 launch human_robot_sim robot_sim.launch.py > "$LOG_DIR/gazebo_sim.log" 2>&1 < /dev/null &
        echo "[INFO] Gazebo launch PID: $! (log: $LOG_DIR/gazebo_sim.log)"
        sleep 5
    fi
fi

# 3. Check / Start FastAPI Backend on Port 8000
if ss -tln | grep -q ":8000 "; then
    echo "[OK] FastAPI backend is already running on http://127.0.0.1:8000"
else
    echo "[INFO] Starting FastAPI backend on http://127.0.0.1:8000 ..."
    (
        cd "$PROJECT_ROOT"
        source /opt/ros/jazzy/setup.bash
        source "$ROBOT_WS/install/setup.bash"
        source "$PROJECT_ROOT/venv/bin/activate"
        setsid python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 > "$LOG_DIR/backend.log" 2>&1 < /dev/null &
        echo "[INFO] Backend PID: $! (log: $LOG_DIR/backend.log)"
    )
    sleep 3
fi

# 4. Check / Start Vite React Frontend on Port 5173
if ss -tln | grep -q ":5173 "; then
    echo "[OK] React frontend is already running on http://127.0.0.1:5173"
else
    echo "[INFO] Starting React frontend on http://127.0.0.1:5173 ..."
    (
        cd "$PROJECT_ROOT/frontend"
        setsid npm run dev > "$LOG_DIR/frontend.log" 2>&1 < /dev/null &
        echo "[INFO] Frontend PID: $! (log: $LOG_DIR/frontend.log)"
    )
    sleep 2
fi

echo ""
echo "============================================================"
echo " Demonstration Stack Status:"
echo " - Dashboard UI:   http://127.0.0.1:5173"
echo " - Backend Health: http://127.0.0.1:8000/api/health"
echo " - Webcam Stream:  http://127.0.0.1:8000/stream/webcam"
echo "============================================================"
