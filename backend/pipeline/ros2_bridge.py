import json
import math
import os
import subprocess
import threading
import time
from typing import Dict, Any, List, Optional

from backend.pipeline.bc_kinematics import bc_kinematics_engine

try:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState

    RCLPY_AVAILABLE = True
    RCLPY_IMPORT_ERROR = None
except Exception as import_exc:
    RCLPY_AVAILABLE = False
    RCLPY_IMPORT_ERROR = str(import_exc)
    Node = object


class ROS2RobotBridge:
    """
    Manages the ROS 2 Jazzy node connection to Gazebo Sim 8 and arm_controller.
    Subscribes to /joint_states for real robot telemetry and publishes validated
    JointTrajectory messages to /arm_controller/joint_trajectory.
    Supports Play, Pause/Resume, Stop, Reset to Home, and WebSocket state streaming.
    """

    LAUNCH_CMD_DISPLAY = (
        "source /opt/ros/jazzy/setup.bash && "
        "source ~/robot_ws/install/setup.bash && "
        "ros2 launch human_robot_sim robot_sim.launch.py"
    )

    def __init__(self):
        self.lock = threading.Lock()
        self.node: Optional[Any] = None
        self.publisher: Optional[Any] = None
        self.subscription: Optional[Any] = None
        self.spin_thread: Optional[threading.Thread] = None
        self.is_spinning = False
        self.ros2_init_error: Optional[str] = RCLPY_IMPORT_ERROR

        # Live Joint State Telemetry from /joint_states
        self.joint_names = list(bc_kinematics_engine.joint_names)
        self.latest_joint_states: Optional[Dict[str, float]] = None
        self.latest_joint_velocities: Optional[Dict[str, float]] = None
        self.latest_ee_xyz: Optional[List[float]] = None
        self.executed_ee_trail: List[List[float]] = []
        self.last_joint_state_time: Optional[float] = None
        self.joint_state_msg_count = 0

        # Command & Playback State
        # command_execution_status:
        # "ready" | "validating" | "executing" | "paused" | "completed" | "stopped" | "error"
        self.latest_joint_targets: Optional[List[float]] = None
        self.command_execution_status = "ready"
        self.active_demonstration_id: Optional[str] = None
        self.playback_progress_percent = 0.0
        self.current_waypoint_step = 0
        self.total_waypoint_steps = 0
        self.last_command_error: Optional[str] = None
        self.last_action_log: List[str] = []

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None

        # Optional managed simulation subprocess
        self.sim_process: Optional[subprocess.Popen] = None

    def _log_event(self, message: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        with self.lock:
            self.last_action_log.append(entry)
            if len(self.last_action_log) > 40:
                self.last_action_log = self.last_action_log[-40:]

    def start_bridge(self) -> bool:
        """Initializes the rclpy node and background spin thread if not already running."""
        if not RCLPY_AVAILABLE:
            self.ros2_init_error = (
                f"rclpy is not available in this Python process ({RCLPY_IMPORT_ERROR}). "
                "Start the backend with /opt/ros/jazzy/setup.bash sourced."
            )
            return False

        with self.lock:
            if self.is_spinning and self.node is not None:
                return True

            try:
                if not rclpy.ok():
                    rclpy.init(args=None)

                self.node = rclpy.create_node("human_action_cloning_dashboard_bridge")
                self.publisher = self.node.create_publisher(
                    JointTrajectory,
                    "/arm_controller/joint_trajectory",
                    10,
                )
                self.subscription = self.node.create_subscription(
                    JointState,
                    "/joint_states",
                    self._joint_state_callback,
                    10,
                )
                self.is_spinning = True
                self.ros2_init_error = None
            except Exception as exc:
                self.ros2_init_error = f"Failed to initialize ROS 2 node: {exc}"
                self.is_spinning = False
                return False

        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()
        self._log_event("ROS 2 bridge node initialized and listening on /joint_states.")
        return True

    def _spin_loop(self):
        while self.is_spinning and RCLPY_AVAILABLE and rclpy.ok():
            try:
                if self.node is not None:
                    rclpy.spin_once(self.node, timeout_sec=0.04)
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    def _joint_state_callback(self, msg: Any):
        now = time.time()
        pos_map: Dict[str, float] = {}
        vel_map: Dict[str, float] = {}

        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                val = float(msg.position[idx])
                pos_map[name] = 0.0 if abs(val) < 1e-9 else round(val, 4)
            if idx < len(msg.velocity):
                vval = float(msg.velocity[idx])
                vel_map[name] = 0.0 if abs(vval) < 1e-9 else round(vval, 4)

        j1 = pos_map.get("joint1", 0.0)
        j2 = pos_map.get("joint2", 0.0)
        j3 = pos_map.get("joint3", 0.0)
        ee_x, ee_y, ee_z = bc_kinematics_engine.forward_kinematics(j1, j2, j3)
        ee_pt = [round(ee_x, 4), round(ee_y, 4), round(ee_z, 4)]

        with self.lock:
            self.latest_joint_states = pos_map
            self.latest_joint_velocities = vel_map
            self.latest_ee_xyz = ee_pt
            self.last_joint_state_time = now
            self.joint_state_msg_count += 1

            # Record live end-effector trail when position changes noticeably
            if not self.executed_ee_trail:
                self.executed_ee_trail.append(ee_pt)
            else:
                prev = self.executed_ee_trail[-1]
                dist = math.sqrt(
                    (ee_pt[0] - prev[0]) ** 2
                    + (ee_pt[1] - prev[1]) ** 2
                    + (ee_pt[2] - prev[2]) ** 2
                )
                if dist > 0.008:
                    self.executed_ee_trail.append(ee_pt)
                    if len(self.executed_ee_trail) > 180:
                        self.executed_ee_trail = self.executed_ee_trail[-180:]

    def get_status(self) -> Dict[str, Any]:
        """
        Inspects real ROS 2 graph, subscriber counts, and /joint_states freshness.
        Never reports a green status unless verified.
        """
        if not self.is_spinning and RCLPY_AVAILABLE:
            self.start_bridge()

        now = time.time()
        with self.lock:
            ros2_connected = bool(
                RCLPY_AVAILABLE and self.is_spinning and self.node is not None and rclpy.ok()
            )
            js_age = (
                round(now - self.last_joint_state_time, 2)
                if self.last_joint_state_time is not None
                else None
            )
            joint_states_fresh = js_age is not None and js_age < 3.0

            traj_subscribers = 0
            clock_publishers = 0
            if ros2_connected and self.node is not None:
                try:
                    traj_subscribers = self.node.count_subscribers(
                        "/arm_controller/joint_trajectory"
                    )
                    clock_publishers = self.node.count_publishers("/clock")
                except Exception:
                    pass

            gazebo_running = bool(
                ros2_connected and (joint_states_fresh or clock_publishers > 0)
            )
            controller_available = bool(
                ros2_connected and traj_subscribers > 0 and joint_states_fresh
            )

            status_reason = None
            if not ros2_connected:
                status_reason = (
                    self.ros2_init_error
                    or "ROS 2 bridge node is not connected. Source /opt/ros/jazzy/setup.bash."
                )
            elif not gazebo_running:
                status_reason = (
                    "Gazebo simulation is not running (no active /clock or /joint_states messages). "
                    "Click 'Launch Simulation' or start robot_sim.launch.py."
                )
            elif not controller_available:
                status_reason = (
                    "Gazebo is running, but /arm_controller/joint_trajectory has no active subscribers "
                    "or /joint_states is not publishing yet."
                )

            ordered_positions = None
            ordered_velocities = None
            if self.latest_joint_states and joint_states_fresh:
                ordered_positions = [
                    self.latest_joint_states.get(jn, 0.0) for jn in self.joint_names
                ]
            if self.latest_joint_velocities and joint_states_fresh:
                ordered_velocities = [
                    self.latest_joint_velocities.get(jn, 0.0) for jn in self.joint_names
                ]

            exec_state = self.command_execution_status
            if exec_state in ("idle", "stopped") and controller_available:
                pass

            return {
                "ros2_connected": ros2_connected,
                "gazebo_running": gazebo_running,
                "controller_available": controller_available,
                "controller_name": "arm_controller (JointTrajectoryController)",
                "trajectory_topic": "/arm_controller/joint_trajectory",
                "trajectory_subscribers": traj_subscribers,
                "joint_states_topic": "/joint_states",
                "joint_states_fresh": joint_states_fresh,
                "joint_states_age_seconds": js_age,
                "joint_state_msg_count": self.joint_state_msg_count,
                "joint_names": self.joint_names,
                "joint_limits": bc_kinematics_engine.joint_limits,
                "latest_joint_positions_rad": ordered_positions,
                "latest_joint_positions_deg": (
                    [round(math.degrees(a), 2) for a in ordered_positions]
                    if ordered_positions is not None
                    else None
                ),
                "latest_joint_velocities_rad_s": ordered_velocities,
                "latest_ee_xyz": self.latest_ee_xyz if joint_states_fresh else None,
                "executed_ee_trail": list(self.executed_ee_trail[-120:]) if joint_states_fresh else [],
                "latest_joint_targets_rad": self.latest_joint_targets,
                "latest_joint_targets_deg": (
                    [round(math.degrees(a), 2) for a in self.latest_joint_targets]
                    if self.latest_joint_targets is not None
                    else None
                ),
                "command_execution_status": exec_state,
                "is_paused": self._pause_event.is_set(),
                "active_demonstration_id": self.active_demonstration_id,
                "playback_progress_percent": self.playback_progress_percent,
                "current_waypoint_step": self.current_waypoint_step,
                "total_waypoint_steps": self.total_waypoint_steps,
                "status_reason": status_reason,
                "launch_command": self.LAUNCH_CMD_DISPLAY,
                "last_command_error": self.last_command_error,
                "recent_logs": list(self.last_action_log[-15:]),
            }

    def launch_simulation_if_needed(self) -> Dict[str, Any]:
        """
        Safely launches robot_sim.launch.py if Gazebo is not already running.
        Cleans up any orphaned launch processes that stopped publishing /joint_states.
        """
        status = self.get_status()
        if status["gazebo_running"] and status["controller_available"]:
            return {
                "launched": False,
                "already_running": True,
                "message": "Gazebo simulation and arm_controller are already running and ready.",
                "status": status,
            }

        with self.lock:
            # Clean up any stale or half-dead ros2 launch / gz sim processes before relaunching
            try:
                subprocess.run(
                    ["pkill", "-f", "robot_sim.launch.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                subprocess.run(
                    ["pkill", "-f", "gz sim"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                time.sleep(0.6)
            except Exception:
                pass

            launch_cmd = (
                "source /opt/ros/jazzy/setup.bash && "
                "source /home/muthulakshmi/robot_ws/install/setup.bash && "
                "export DISPLAY=${DISPLAY:-:0} && "
                "exec ros2 launch human_robot_sim robot_sim.launch.py"
            )
            try:
                self.sim_process = subprocess.Popen(
                    ["/bin/bash", "-c", launch_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                pid = self.sim_process.pid
            except Exception as exc:
                self._log_event(f"Failed to launch simulation subprocess: {exc}")
                raise RuntimeError(f"Failed to launch simulation: {exc}") from exc

        self._log_event(f"Launched Gazebo simulation subprocess (PID {pid}).")
        return {
            "launched": True,
            "pid": pid,
            "message": "Started ros2 launch human_robot_sim robot_sim.launch.py. Waiting ~5-8 seconds for controllers to become active.",
        }

    def _publish_single_point(
        self, positions: List[float], duration_sec: float, velocities: Optional[List[float]] = None
    ):
        """Validates joint angles against URDF limits and publishes a JointTrajectory point."""
        if len(positions) != 3:
            raise ValueError(f"Expected 3 joint positions, got {len(positions)}")

        ok, violations = bc_kinematics_engine.check_joint_limits(
            positions[0], positions[1], positions[2]
        )
        if not ok:
            raise ValueError(f"Refusing to command joints outside URDF limits: {violations}")

        if self.publisher is None:
            raise RuntimeError("ROS 2 trajectory publisher is not initialized.")

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        if velocities is not None and len(velocities) == 3:
            pt.velocities = [float(v) for v in velocities]

        secs = int(math.floor(duration_sec))
        nanosecs = int(round((duration_sec - secs) * 1e9))
        pt.time_from_start.sec = secs
        pt.time_from_start.nanosec = nanosecs

        traj.points.append(pt)
        self.publisher.publish(traj)

        with self.lock:
            self.latest_joint_targets = [round(float(p), 4) for p in positions]

    def execute_test_motion(self) -> Dict[str, Any]:
        """Executes a safe, predefined 3-waypoint movement within verified URDF limits."""
        status = self.get_status()
        if not status["gazebo_running"] or not status["controller_available"]:
            reason = status.get("status_reason") or "Simulation or arm_controller is not ready."
            self._log_event(f"Rejected Test Motion: {reason}")
            raise RuntimeError(reason)

        with self.lock:
            if self.command_execution_status in ("executing", "testing", "playing"):
                raise RuntimeError(
                    f"Robot is currently busy ({self.command_execution_status}). Stop motion first."
                )
            self._stop_event.clear()
            self._pause_event.clear()
            self.executed_ee_trail = []
            self.command_execution_status = "executing"
            self.active_demonstration_id = "safe_test_motion"
            self.playback_progress_percent = 0.0
            self.current_waypoint_step = 0
            self.total_waypoint_steps = 3
            self.last_command_error = None

        test_waypoints = [
            {"positions": [0.40, 0.45, 0.60], "duration": 1.8, "label": "Pose A (+Yaw, Reach)"},
            {"positions": [-0.40, 0.35, 0.75], "duration": 2.0, "label": "Pose B (-Yaw, Flex)"},
            {"positions": [0.0, 0.0, 0.0], "duration": 1.8, "label": "Home Zero Pose"},
        ]
        for wp in test_waypoints:
            ok, violations = bc_kinematics_engine.check_joint_limits(*wp["positions"])
            if not ok:
                with self.lock:
                    self.command_execution_status = "error"
                raise ValueError(f"Test waypoint violated URDF limits: {violations}")

        def _run_test():
            self._log_event("Starting safe 3-step Test Motion in Gazebo...")
            try:
                for idx, wp in enumerate(test_waypoints):
                    if self._stop_event.is_set():
                        return

                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    if self._stop_event.is_set():
                        return

                    with self.lock:
                        self.current_waypoint_step = idx + 1
                        self.playback_progress_percent = round(
                            ((idx + 1) / len(test_waypoints)) * 100.0, 1
                        )

                    self._publish_single_point(wp["positions"], wp["duration"])
                    self._log_event(
                        f"Test Motion step {idx + 1}/3 ({wp['label']}): {wp['positions']} rad"
                    )

                    elapsed = 0.0
                    dt = 0.08
                    while elapsed < wp["duration"]:
                        if self._stop_event.is_set():
                            return
                        if self._pause_event.is_set():
                            while self._pause_event.is_set() and not self._stop_event.is_set():
                                time.sleep(0.1)
                            if self._stop_event.is_set():
                                return
                            # Re-issue current waypoint for remaining duration
                            rem = max(0.3, wp["duration"] - elapsed)
                            self._publish_single_point(wp["positions"], rem)
                        time.sleep(dt)
                        elapsed += dt

                with self.lock:
                    if not self._stop_event.is_set():
                        self.playback_progress_percent = 100.0
                        self.command_execution_status = "completed"
                self._log_event("Test Motion completed successfully.")
            except Exception as exc:
                with self.lock:
                    self.command_execution_status = "error"
                    self.last_command_error = str(exc)
                self._log_event(f"Test Motion error: {exc}")

        self._playback_thread = threading.Thread(target=_run_test, daemon=True)
        self._playback_thread.start()

        return {
            "started": True,
            "mode": "test_motion",
            "waypoints": test_waypoints,
            "message": "Safe 3-waypoint test motion started.",
        }

    def play_demonstration(self, demonstration_id: str) -> Dict[str, Any]:
        """
        Validates and executes a processed demonstration on the simulated robot in Gazebo.
        Updates execution state through Validating -> Executing -> Completed.
        """
        status = self.get_status()
        if not status["gazebo_running"] or not status["controller_available"]:
            reason = status.get("status_reason") or "Simulation or arm_controller is not ready."
            self._log_event(f"Rejected playback for {demonstration_id}: {reason}")
            raise RuntimeError(reason)

        with self.lock:
            if self.command_execution_status in ("executing", "testing", "playing"):
                raise RuntimeError(
                    f"Robot is currently executing '{self.active_demonstration_id}'. Stop motion first."
                )
            self.command_execution_status = "validating"

        demo_data = bc_kinematics_engine.get_demonstration(demonstration_id)
        summary = demo_data.get("processing_summary")
        if not summary or summary.get("status") != "processed":
            with self.lock:
                self.command_execution_status = "error"
            reason = (
                f"Demonstration '{demonstration_id}' has not been processed or was rejected."
            )
            self._log_event(f"Rejected playback: {reason}")
            raise ValueError(reason)

        waypoints = summary.get("waypoints", [])
        val_report = bc_kinematics_engine.validate_trajectory(waypoints)
        if not val_report["valid"]:
            with self.lock:
                self.command_execution_status = "error"
                self.last_command_error = "; ".join(val_report["violations"])
            raise ValueError(f"Trajectory validation failed: {val_report['violations']}")

        with self.lock:
            self._stop_event.clear()
            self._pause_event.clear()
            self.executed_ee_trail = []
            self.command_execution_status = "executing"
            self.active_demonstration_id = demonstration_id
            self.playback_progress_percent = 0.0
            self.current_waypoint_step = 0
            self.total_waypoint_steps = len(waypoints)
            self.last_command_error = None

        def _run_playback():
            self._log_event(
                f"Executing '{demonstration_id}' ({len(waypoints)} validated waypoints) in Gazebo..."
            )
            try:
                for idx, wp in enumerate(waypoints):
                    if self._stop_event.is_set():
                        self._log_event(f"Execution of '{demonstration_id}' stopped by user.")
                        return

                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    if self._stop_event.is_set():
                        return

                    positions = wp["joints_rad"]
                    duration = float(wp.get("duration_sec", 0.7))

                    with self.lock:
                        self.current_waypoint_step = idx + 1
                        self.playback_progress_percent = round(
                            ((idx + 1) / len(waypoints)) * 100.0, 1
                        )

                    self._publish_single_point(positions, duration)
                    self._log_event(
                        f"Waypoint {idx + 1}/{len(waypoints)}: joints={positions} rad"
                    )

                    elapsed = 0.0
                    dt = 0.08
                    while elapsed < duration:
                        if self._stop_event.is_set():
                            return
                        if self._pause_event.is_set():
                            while self._pause_event.is_set() and not self._stop_event.is_set():
                                time.sleep(0.1)
                            if self._stop_event.is_set():
                                return
                            rem = max(0.25, duration - elapsed)
                            self._publish_single_point(positions, rem)
                        time.sleep(dt)
                        elapsed += dt

                with self.lock:
                    if not self._stop_event.is_set():
                        self.playback_progress_percent = 100.0
                        self.command_execution_status = "completed"

                self._update_demo_playback_status(demonstration_id, "completed")
                self._log_event(f"Execution of '{demonstration_id}' completed successfully.")
            except Exception as exc:
                with self.lock:
                    self.command_execution_status = "error"
                    self.last_command_error = str(exc)
                self._update_demo_playback_status(demonstration_id, f"error: {exc}")
                self._log_event(f"Execution error on '{demonstration_id}': {exc}")

        self._playback_thread = threading.Thread(target=_run_playback, daemon=True)
        self._playback_thread.start()

        return {
            "started": True,
            "demonstration_id": demonstration_id,
            "waypoint_count": len(waypoints),
            "target_generation_mode": summary.get("target_generation_mode"),
            "target_generation_label": summary.get("target_generation_label"),
            "estimated_duration_sec": summary.get("estimated_playback_duration_sec"),
        }

    def pause_motion(self) -> Dict[str, Any]:
        """Pauses active trajectory playback and holds current joint angles."""
        with self.lock:
            if self.command_execution_status not in ("executing", "playing", "testing"):
                raise RuntimeError("No active trajectory execution to pause.")
            self._pause_event.set()
            self.command_execution_status = "paused"
            hold_positions = (
                [float(self.latest_joint_states.get(jn, 0.0)) for jn in self.joint_names]
                if self.latest_joint_states
                else [0.0, 0.0, 0.0]
            )

        try:
            self._publish_single_point(hold_positions, duration_sec=0.15, velocities=[0.0, 0.0, 0.0])
        except Exception:
            pass

        self._log_event(f"Execution paused at step {self.current_waypoint_step}/{self.total_waypoint_steps}.")
        return {
            "paused": True,
            "command_execution_status": "paused",
            "step": self.current_waypoint_step,
            "message": "Robot execution paused.",
        }

    def resume_motion(self) -> Dict[str, Any]:
        """Resumes paused trajectory playback."""
        with self.lock:
            if not self._pause_event.is_set() or self.command_execution_status != "paused":
                raise RuntimeError("Robot is not currently paused.")
            self._pause_event.clear()
            self.command_execution_status = "executing"

        self._log_event(f"Execution resumed from step {self.current_waypoint_step}/{self.total_waypoint_steps}.")
        return {
            "resumed": True,
            "command_execution_status": "executing",
            "step": self.current_waypoint_step,
            "message": "Robot execution resumed.",
        }

    def reset_simulation(self) -> Dict[str, Any]:
        """
        Safely stops any active trajectory, clears trails, and commands the robot arm
        back to its Home zero configuration [0.0, 0.0, 0.0] rad in Gazebo.
        """
        self._stop_event.set()
        self._pause_event.clear()

        status = self.get_status()
        if not status["gazebo_running"] or not status["controller_available"]:
            raise RuntimeError(
                status.get("status_reason") or "Cannot reset: Gazebo or arm_controller is offline."
            )

        self._publish_single_point([0.0, 0.0, 0.0], duration_sec=1.5, velocities=[0.0, 0.0, 0.0])

        with self.lock:
            self.command_execution_status = "ready"
            self.active_demonstration_id = None
            self.playback_progress_percent = 0.0
            self.current_waypoint_step = 0
            self.total_waypoint_steps = 0
            self.last_command_error = None
            self.executed_ee_trail = []

        self._log_event("Reset Robot to Home Pose [0.0, 0.0, 0.0] rad.")
        return {
            "reset": True,
            "home_positions_rad": [0.0, 0.0, 0.0],
            "command_execution_status": "ready",
            "message": "Robot arm reset to Home pose [0.0, 0.0, 0.0] rad.",
        }

    def _update_demo_playback_status(self, demonstration_id: str, playback_status: str):
        try:
            safe_id = os.path.basename(demonstration_id)
            meta_path = os.path.join(
                bc_kinematics_engine.demonstrations_dir, safe_id, "metadata.json"
            )
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["playback_status"] = playback_status
                meta["last_played_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        except Exception:
            pass

    def stop_motion(self) -> Dict[str, Any]:
        """
        Safely cancels active playback and sends an immediate zero-velocity hold
        trajectory at the robot's current joint positions to /arm_controller/joint_trajectory.
        """
        self._stop_event.set()
        self._pause_event.clear()

        hold_positions = [0.0, 0.0, 0.0]
        with self.lock:
            active_id = self.active_demonstration_id
            if self.latest_joint_states:
                hold_positions = [
                    float(self.latest_joint_states.get(jn, 0.0)) for jn in self.joint_names
                ]
            elif self.latest_joint_targets:
                hold_positions = list(self.latest_joint_targets)

            self.command_execution_status = "stopped"

        clamped_hold = []
        for idx, jn in enumerate(self.joint_names):
            lim = bc_kinematics_engine.joint_limits.get(jn, {"lower": -1.57, "upper": 1.57})
            clamped_hold.append(
                max(lim["lower"], min(lim["upper"], hold_positions[idx]))
            )

        published_hold = False
        if self.is_spinning and self.publisher is not None:
            try:
                self._publish_single_point(
                    clamped_hold, duration_sec=0.15, velocities=[0.0, 0.0, 0.0]
                )
                published_hold = True
            except Exception:
                pass

        if active_id and active_id != "safe_test_motion":
            self._update_demo_playback_status(active_id, "stopped")

        self._log_event(
            f"Stop Robot Motion triggered. Hold command sent at {clamped_hold} rad."
        )
        return {
            "stopped": True,
            "hold_positions_rad": [round(v, 4) for v in clamped_hold],
            "published_hold_command": published_hold,
            "command_execution_status": "stopped",
            "message": "Active robot motion cancelled and safe hold command sent.",
        }

    def shutdown(self):
        self._stop_event.set()
        self._pause_event.clear()
        with self.lock:
            self.is_spinning = False
        if self.spin_thread and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        if self.node is not None and RCLPY_AVAILABLE:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None


ros2_bridge = ROS2RobotBridge()
