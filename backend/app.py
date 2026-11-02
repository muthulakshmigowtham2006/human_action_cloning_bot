import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.pipeline.camera_streamer import camera_streamer
from backend.pipeline.bc_kinematics import bc_kinematics_engine
from backend.pipeline.ros2_bridge import ros2_bridge

server_start_time = time.time()


class StopRecordingRequest(BaseModel):
    auto_process: bool = True
    auto_execute: bool = False


class PlayDemonstrationRequest(BaseModel):
    demonstration_id: str = Field(..., min_length=1, max_length=128)


class ValidateTrajectoryRequest(BaseModel):
    demonstration_id: Optional[str] = None
    waypoints: Optional[List[Dict[str, Any]]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Backend] Initializing camera capture streamer & MediaPipe HandLandmarker...")
    cam_ok = camera_streamer.start()
    if not cam_ok:
        print("[Backend] Warning: Could not open camera device 0 initially.")
    else:
        print("[Backend] Camera streamer initialized successfully.")

    print("[Backend] Initializing ROS 2 bridge node...")
    ros2_bridge.start_bridge()

    yield

    print("[Backend] Shutting down camera streamer & ROS 2 bridge...")
    camera_streamer.stop()
    ros2_bridge.shutdown()
    print("[Backend] Clean shutdown complete.")


app = FastAPI(
    title="Human Action Cloning Dashboard API",
    description="End-to-End API for AI-Powered Robot Arm Learning from Human Demonstration",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "project": "AI-Powered Robot Arm Learning from Human Demonstration",
        "service": "FastAPI Backend",
        "stage": "Two-Panel Integrated Demonstration Ready",
    }


@app.get("/api/health")
def health_check():
    camera_status = camera_streamer.get_status()
    rec_status = camera_streamer.get_recording_status()
    model_status = bc_kinematics_engine.get_status()
    robot_status = ros2_bridge.get_status()
    uptime_seconds = round(time.time() - server_start_time, 1)

    return {
        "status": "online",
        "uptime_seconds": uptime_seconds,
        "camera": camera_status,
        "recording": rec_status,
        "model_kinematics": model_status,
        "robot": robot_status,
        "subsystems": {
            "webcam_capture": "active" if camera_status["connected"] else "disconnected",
            "hand_tracking": (
                "active" if camera_status["landmarker_loaded"] else "error"
            ),
            "behavior_cloning_model": (
                "loaded" if model_status["model_loaded"] else "error"
            ),
            "ros2_bridge": (
                "connected" if robot_status["ros2_connected"] else "disconnected"
            ),
            "gazebo_sim": (
                "running" if robot_status["gazebo_running"] else "offline"
            ),
            "arm_controller": (
                "ready" if robot_status["controller_available"] else "unavailable"
            ),
        },
    }


# ============================================================
# CAMERA ENDPOINTS
# ============================================================

@app.get("/api/camera/status")
def get_camera_status():
    return camera_streamer.get_status()


@app.post("/api/camera/start")
def start_camera():
    status = camera_streamer.get_status()
    if status["connected"]:
        return {
            "started": True,
            "already_active": True,
            "message": "Camera capture is already active.",
            "camera": status,
        }

    ok = camera_streamer.start()
    new_status = camera_streamer.get_status()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=new_status.get("error") or "Failed to open camera /dev/video0",
        )
    return {
        "started": True,
        "already_active": False,
        "message": "Camera capture started successfully.",
        "camera": new_status,
    }


@app.post("/api/camera/stop")
def stop_camera():
    result = camera_streamer.stop()
    return {
        "stopped": True,
        "saved_active_recording": result.get("saved_active_recording"),
        "message": "Camera capture stopped and /dev/video0 released.",
        "camera": camera_streamer.get_status(),
    }


async def frame_generator():
    """Streams multipart MJPEG frames with frame pacing while camera is active."""
    while True:
        if not camera_streamer.is_running:
            break
        jpeg_bytes = camera_streamer.get_latest_jpeg()
        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        await asyncio.sleep(0.033)


@app.get("/stream/webcam")
def webcam_stream():
    status = camera_streamer.get_status()
    if not status["connected"]:
        raise HTTPException(
            status_code=503,
            detail="Camera capture is not active. Click 'Start Camera' first.",
        )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# RECORDING & AUTOMATIC PIPELINE ENDPOINTS
# ============================================================

@app.get("/api/recording/status")
def get_recording_status():
    return camera_streamer.get_recording_status()


@app.post("/api/recording/start")
def start_recording():
    try:
        return camera_streamer.start_recording()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recording/stop")
async def stop_recording(req: Optional[StopRecordingRequest] = None):
    """
    Stops active recording, verifies that valid hand landmarks were captured,
    automatically executes the Left-to-Right Processing Pipeline on the recorded
    trajectory, and optionally triggers robot execution if auto_execute=True.
    """
    opts = req or StopRecordingRequest()
    try:
        metadata = await asyncio.to_thread(camera_streamer.stop_recording)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    demo_id = metadata["demonstration_id"]
    processing_summary = None
    execution_result = None

    if opts.auto_process:
        camera_streamer.set_recording_state("processing")
        try:
            processing_summary = await asyncio.to_thread(
                bc_kinematics_engine.process_demonstration, demo_id
            )
            if processing_summary.get("status") == "processed":
                camera_streamer.set_recording_state("complete")
                metadata["processing_status"] = "processed"
                metadata["processing_summary"] = processing_summary

                if opts.auto_execute:
                    rob_status = ros2_bridge.get_status()
                    if rob_status["gazebo_running"] and rob_status["controller_available"]:
                        execution_result = ros2_bridge.play_demonstration(demo_id)
            else:
                camera_streamer.set_recording_state(
                    "error", processing_summary.get("error")
                )
        except Exception as proc_exc:
            camera_streamer.set_recording_state("error", str(proc_exc))
            raise HTTPException(
                status_code=400,
                detail=f"Recording saved as '{demo_id}', but processing failed: {proc_exc}",
            ) from proc_exc

    return {
        **metadata,
        "processing_summary": processing_summary,
        "auto_execution": execution_result,
    }


@app.get("/api/demonstrations")
def list_demonstrations():
    demos = bc_kinematics_engine.list_demonstrations()
    return {
        "count": len(demos),
        "demonstrations": demos,
    }


@app.get("/api/demonstrations/{demonstration_id}")
def get_demonstration_details(demonstration_id: str):
    try:
        return bc_kinematics_engine.get_demonstration(demonstration_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ============================================================
# PROCESSING & VALIDATION ENDPOINTS
# ============================================================

@app.post("/api/demonstrations/{demonstration_id}/process")
async def process_demonstration(demonstration_id: str):
    try:
        camera_streamer.set_recording_state("processing")
        summary = await asyncio.to_thread(
            bc_kinematics_engine.process_demonstration, demonstration_id
        )
        camera_streamer.set_recording_state(
            "complete" if summary.get("status") == "processed" else "error",
            summary.get("error"),
        )
        return summary
    except FileNotFoundError as exc:
        camera_streamer.set_recording_state("error", str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        camera_streamer.set_recording_state("error", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/demonstrations/{demonstration_id}/processing-status")
def get_processing_status(demonstration_id: str):
    try:
        demo = bc_kinematics_engine.get_demonstration(demonstration_id)
        return {
            "demonstration_id": demonstration_id,
            "processing_status": demo.get("processing_status", "unprocessed"),
            "model_inference_status": demo.get("model_inference_status", "not_run"),
            "processing_summary": demo.get("processing_summary"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/robot/validate-trajectory")
def validate_robot_trajectory(req: ValidateTrajectoryRequest):
    if req.waypoints is not None:
        return bc_kinematics_engine.validate_trajectory(req.waypoints)
    if req.demonstration_id:
        try:
            demo = bc_kinematics_engine.get_demonstration(req.demonstration_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        summary = demo.get("processing_summary") or {}
        waypoints = summary.get("waypoints", [])
        return bc_kinematics_engine.validate_trajectory(waypoints)
    raise HTTPException(
        status_code=400, detail="Provide either 'demonstration_id' or 'waypoints'."
    )


# ============================================================
# ROBOT SIMULATION, URDF & LIVE JOINT STREAMING ENDPOINTS
# ============================================================

@app.get("/api/robot/urdf")
def get_robot_urdf():
    return bc_kinematics_engine.get_urdf_model_spec()


@app.get("/api/robot/status")
def get_robot_status():
    return ros2_bridge.get_status()


@app.get("/api/robot/joint-states")
def get_live_joint_states():
    status = ros2_bridge.get_status()
    return {
        "ros2_connected": status["ros2_connected"],
        "gazebo_running": status["gazebo_running"],
        "controller_available": status["controller_available"],
        "joint_states_fresh": status["joint_states_fresh"],
        "joint_names": status["joint_names"],
        "positions_rad": status["latest_joint_positions_rad"],
        "positions_deg": status["latest_joint_positions_deg"],
        "velocities_rad_s": status["latest_joint_velocities_rad_s"],
        "targets_rad": status["latest_joint_targets_rad"],
        "ee_xyz": status["latest_ee_xyz"],
        "executed_ee_trail": status["executed_ee_trail"],
        "command_execution_status": status["command_execution_status"],
        "is_paused": status["is_paused"],
        "playback_progress_percent": status["playback_progress_percent"],
        "current_waypoint_step": status["current_waypoint_step"],
        "total_waypoint_steps": status["total_waypoint_steps"],
    }


@app.websocket("/ws/robot-state")
@app.websocket("/api/ws/robot-state")
async def websocket_robot_state(websocket: WebSocket):
    """Streams live ROS 2 /joint_states and execution telemetry at 25 Hz to the browser 3D viewport."""
    await websocket.accept()
    try:
        while True:
            status = ros2_bridge.get_status()
            await websocket.send_json({
                "ros2_connected": status["ros2_connected"],
                "gazebo_running": status["gazebo_running"],
                "controller_available": status["controller_available"],
                "joint_states_fresh": status["joint_states_fresh"],
                "joint_state_msg_count": status["joint_state_msg_count"],
                "positions_rad": status["latest_joint_positions_rad"],
                "positions_deg": status["latest_joint_positions_deg"],
                "velocities_rad_s": status["latest_joint_velocities_rad_s"],
                "targets_rad": status["latest_joint_targets_rad"],
                "ee_xyz": status["latest_ee_xyz"],
                "executed_ee_trail": status["executed_ee_trail"],
                "command_execution_status": status["command_execution_status"],
                "is_paused": status["is_paused"],
                "active_demonstration_id": status["active_demonstration_id"],
                "playback_progress_percent": status["playback_progress_percent"],
                "current_waypoint_step": status["current_waypoint_step"],
                "total_waypoint_steps": status["total_waypoint_steps"],
            })
            await asyncio.sleep(0.04)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/api/robot/connect")
def connect_robot():
    ros2_bridge.start_bridge()
    status = ros2_bridge.get_status()
    return {
        "connected": status["ros2_connected"],
        "gazebo_running": status["gazebo_running"],
        "controller_available": status["controller_available"],
        "status": status,
    }


@app.post("/api/robot/launch-sim")
def launch_robot_simulation():
    try:
        return ros2_bridge.launch_simulation_if_needed()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/robot/test-motion")
def trigger_test_motion():
    try:
        return ros2_bridge.execute_test_motion()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/robot/play-demonstration")
@app.post("/api/robot/execute")
def trigger_play_demonstration(req: PlayDemonstrationRequest):
    try:
        return ros2_bridge.play_demonstration(req.demonstration_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/robot/pause")
def trigger_pause_robot():
    try:
        return ros2_bridge.pause_motion()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/robot/resume")
def trigger_resume_robot():
    try:
        return ros2_bridge.resume_motion()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/robot/reset")
def trigger_reset_robot():
    try:
        return ros2_bridge.reset_simulation()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/robot/stop")
def trigger_stop_robot():
    return ros2_bridge.stop_motion()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
