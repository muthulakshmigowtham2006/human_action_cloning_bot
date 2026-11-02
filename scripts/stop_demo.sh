#!/usr/bin/env bash

echo "============================================================"
echo " Stopping AI-Powered Robot Arm Demonstration Stack Safely"
echo "============================================================"

# 1. Safely stop robot motion and release webcam via API if backend is reachable
if curl -s -m 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[INFO] Sending safe Stop Robot Motion command..."
    curl -s -m 2 -X POST http://127.0.0.1:8000/api/robot/stop >/dev/null 2>&1 || true
    echo "[INFO] Releasing webcam /dev/video0..."
    curl -s -m 2 -X POST http://127.0.0.1:8000/api/camera/stop >/dev/null 2>&1 || true
fi

# 2. Stop FastAPI backend
if pgrep -f "uvicorn backend.app:app" >/dev/null 2>&1; then
    echo "[INFO] Stopping FastAPI backend..."
    pkill -TERM -f "uvicorn backend.app:app" || true
    sleep 1
else
    echo "[OK] FastAPI backend is not running."
fi

# 3. Stop Vite frontend
if pgrep -f "vite --host 127.0.0.1 --port 5173" >/dev/null 2>&1; then
    echo "[INFO] Stopping Vite frontend server..."
    pkill -TERM -f "vite --host 127.0.0.1 --port 5173" || true
else
    echo "[OK] Vite frontend server is not running."
fi

# 4. Optionally stop Gazebo simulation if --all or --stop-sim is passed
if [[ "$1" == "--all" || "$1" == "--stop-sim" ]]; then
    echo "[INFO] Stopping Gazebo simulation and ROS 2 launch processes..."
    pkill -TERM -f "robot_sim.launch.py" 2>/dev/null || true
    pkill -TERM -f "gz sim" 2>/dev/null || true
fi

echo "[OK] Shutdown complete."

