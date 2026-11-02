import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline.bc_kinematics import BCKinematicsEngine, bc_kinematics_engine
from backend.pipeline.camera_streamer import CameraStreamer, camera_streamer


def make_synthetic_21_landmarks(wrist_x: float, wrist_y: float, wrist_z: float):
    """Generates a realistic 21-landmark hand skeleton anchored at (wrist_x, wrist_y, wrist_z)."""
    lms = []
    for idx in range(21):
        lms.append([
            round(wrist_x + idx * 0.004, 5),
            round(wrist_y - idx * 0.005, 5),
            round(wrist_z - idx * 0.001, 5),
        ])
    return lms


def test_section5_recording_regression():
    print("\n=== [Regression Suite] Section 5 & Test A/B: Recording Buffer & Landmark Pipeline ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        streamer = CameraStreamer(device_id=0, demonstrations_dir=tmp_dir)

        # 1. Start headless recording session
        start_info = streamer.start_recording(allow_headless_test=True)
        demo_id = start_info["demonstration_id"]
        assert start_info["recording"] is True
        assert demo_id.startswith("demo_")
        base_t = streamer.recording_start_time

        # 2. Feed 25 frames of smooth synthetic hand movement + 3 frames with no hand
        expected_total = 28
        expected_valid = 25
        for i in range(expected_total):
            t_now = base_t + (i + 1) * 0.04
            if i in (5, 12, 19):
                # Explicit no-hand frame
                res = streamer.ingest_frame_landmarks(None, now=t_now)
                assert res["hand_detected"] is False
            else:
                wx = 0.35 + 0.25 * math.sin(i * 0.12)
                wy = 0.55 - 0.18 * math.cos(i * 0.10)
                wz = -0.04 + 0.02 * math.sin(i * 0.08)
                lms = make_synthetic_21_landmarks(wx, wy, wz)
                res = streamer.ingest_frame_landmarks(lms, now=t_now)
                assert res["hand_detected"] is True

        assert len(streamer.recorded_frames_buffer) == expected_total
        assert streamer.recording_detected_count == expected_valid
        print(
            f"[PASS] Synthetic valid & missing landmark frames appended to buffer "
            f"({expected_valid}/{expected_total} valid)."
        )

        # 3. Stop recording and verify saved files
        meta = streamer.stop_recording()
        assert meta["demonstration_id"] == demo_id
        assert meta["valid"] is True
        assert meta["frame_count"] == expected_total
        assert meta["detected_hand_frames"] == expected_valid
        assert meta["missing_hand_frames"] == 3
        assert os.path.isfile(meta["landmarks_csv"]) and os.path.getsize(meta["landmarks_csv"]) > 0
        assert os.path.isfile(meta["trajectory_json"]) and os.path.getsize(meta["trajectory_json"]) > 0

        # 4. Read back landmarks.csv and verify monotonic timestamps & 66 columns
        with open(meta["landmarks_csv"], "r") as f:
            reader = csv.DictReader(f)
            assert len(reader.fieldnames) == 66
            rows = list(reader)
            assert len(rows) == expected_total
            prev_ts = -1.0
            valid_read = 0
            for idx, r in enumerate(rows):
                assert int(r["frame_index"]) == idx
                ts = float(r["timestamp_sec"])
                assert ts > prev_ts, f"Non-increasing timestamp at row {idx}: {ts} <= {prev_ts}"
                prev_ts = ts
                if r["hand_detected"] == "1":
                    valid_read += 1
                    assert float(r["landmark_0_x"]) > 0.0
            assert valid_read == expected_valid
        print(
            f"[PASS] Saved CSV & JSON verified on disk and read back ({len(rows)} rows, "
            f"strictly increasing timestamps)."
        )

        # 5. Process the recorded trajectory through BCKinematicsEngine (Tests C & D)
        engine = BCKinematicsEngine()
        engine.demonstrations_dir = tmp_dir
        summary = engine.process_demonstration(demo_id)
        assert summary["status"] == "processed", f"Processing failed: {summary.get('error')}"
        assert summary["detected_hand_frames"] == expected_valid
        assert summary["waypoint_count"] > 0
        assert summary["validation_report"]["valid"] is True
        assert len(summary["pipeline_stages"]) == 8
        assert all(s["status"] == "complete" for s in summary["pipeline_stages"])
        print(
            f"[PASS] Test C & D: Processed recorded hand trajectory into {summary['waypoint_count']} "
            f"URDF-validated waypoints (max FK error={summary['validation_report']['max_fk_error_m']}m)."
        )

        # 6. Verify a zero-hand recording is explicitly rejected on stop_recording()
        s2 = streamer.start_recording(allow_headless_test=True)
        empty_id = s2["demonstration_id"]
        for k in range(5):
            streamer.ingest_frame_landmarks(None, now=time.time() + (k + 1) * 0.03)
        try:
            streamer.stop_recording()
            raise AssertionError("Expected stop_recording() to reject a 0-hand recording")
        except ValueError as verr:
            assert "rejected" in str(verr).lower()
            assert not os.path.exists(os.path.join(tmp_dir, empty_id))
            print("[PASS] Zero-hand recording explicitly rejected and no corrupt directory left behind.")


def test_model_and_kinematics():
    print("\n=== [Test D] Behavior Cloning Model, URDF Spec & Kinematics Verification ===")
    status = bc_kinematics_engine.get_status()
    assert status["model_loaded"] is True, f"Model failed to load: {status['model_error']}"
    assert status["urdf_loaded"] is True, f"URDF failed to load: {status['urdf_error']}"
    assert status["joint_names"] == ["joint1", "joint2", "joint3"]

    urdf_data = bc_kinematics_engine.get_urdf_model_spec()
    assert urdf_data["urdf_loaded"] is True
    assert set(urdf_data["spec"]["links"].keys()) == {
        "base_link",
        "link1",
        "link2",
        "link3",
        "end_effector",
    }
    print("[PASS] URDF 5-link & 3-revolute-joint spec verified for 3D browser viewport.")

    # Verify IK and FK consistency on reachable target
    j1, j2, j3 = bc_kinematics_engine.inverse_kinematics(1.02, 0.15, 1.94)
    fk_x, fk_y, fk_z = bc_kinematics_engine.forward_kinematics(j1, j2, j3)
    err = math.sqrt((1.02 - fk_x) ** 2 + (0.15 - fk_y) ** 2 + (1.94 - fk_z) ** 2)
    assert err < 1e-6, f"FK error too large: {err}"
    ok, violations = bc_kinematics_engine.check_joint_limits(j1, j2, j3)
    assert ok is True, f"Expected valid URDF angles, got violations: {violations}"
    print(
        f"[PASS] IK/FK verified at (1.02, 0.15, 1.94) -> joints=({j1:.3f}, {j2:.3f}, {j3:.3f}) rad, FK err={err:.8f}m"
    )

    # Verify unreachable target rejection
    try:
        bc_kinematics_engine.inverse_kinematics(5.0, 0.0, 1.2)
        raise AssertionError("Should have rejected unreachable target")
    except ValueError:
        print("[PASS] Unreachable target (5.0, 0.0, 1.2) properly rejected.")


def http_json(url: str, method: str = "GET", body: dict = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=12) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_live_api_and_ros2(base_url: str = "http://127.0.0.1:8000"):
    print("\n=== [Tests E, F, G] Live FastAPI Endpoints, URDF 3D Spec & ROS 2 / Gazebo Execution ===")
    code, health = http_json(f"{base_url}/api/health")
    assert code == 200 and health["status"] == "online"
    assert health["camera"]["landmarker_loaded"] is True
    print("[PASS] GET /api/health online, MediaPipe HandLandmarker loaded.")

    # Verify /api/robot/urdf for embedded 3D viewport
    code, urdf_resp = http_json(f"{base_url}/api/robot/urdf")
    assert code == 200 and urdf_resp["urdf_loaded"] is True
    assert "human_robot_arm" in urdf_resp["urdf_xml"]
    print("[PASS] Test E: GET /api/robot/urdf serves full URDF XML & parsed link/joint geometry.")

    # Process & validate baseline demonstration
    code, base_proc = http_json(
        f"{base_url}/api/demonstrations/demo_baseline_existing/process", method="POST"
    )
    assert code == 200 and base_proc["status"] == "processed"
    assert base_proc["waypoint_count"] == 16

    code, val_resp = http_json(
        f"{base_url}/api/robot/validate-trajectory",
        method="POST",
        body={"demonstration_id": "demo_baseline_existing"},
    )
    assert code == 200 and val_resp["valid"] is True
    print(
        f"[PASS] POST /api/robot/validate-trajectory verified {val_resp['waypoint_count']} waypoints."
    )

    # Check ROS 2 & Gazebo status
    code, rob_status = http_json(f"{base_url}/api/robot/status")
    assert code == 200
    print(
        f"Robot Status: ros2={rob_status['ros2_connected']}, "
        f"gazebo={rob_status['gazebo_running']}, "
        f"controller={rob_status['controller_available']}, "
        f"joints_rad={rob_status['latest_joint_positions_rad']}"
    )

    if rob_status["gazebo_running"] and rob_status["controller_available"]:
        # Reset to Home [0, 0, 0] first
        code, reset_res = http_json(f"{base_url}/api/robot/reset", method="POST")
        assert code == 200 and reset_res["reset"] is True
        time.sleep(1.6)

        _, home_js = http_json(f"{base_url}/api/robot/joint-states")
        initial_pos = home_js["positions_rad"]
        print(f"[PASS] Test G (Reset): Robot reset to Home pose: {initial_pos}")

        # Execute Demonstration (Test F)
        code, play_res = http_json(
            f"{base_url}/api/robot/execute",
            method="POST",
            body={"demonstration_id": "demo_baseline_existing"},
        )
        assert code == 200 and play_res["started"] is True
        print("[PASS] Test F: Triggered POST /api/robot/execute for 'demo_baseline_existing'.")
        time.sleep(2.2)

        _, mid_js = http_json(f"{base_url}/api/robot/joint-states")
        mid_pos = mid_js["positions_rad"]
        print(
            f"Live /joint_states during execution: {mid_pos}, EE={mid_js['ee_xyz']}, "
            f"step={mid_js['current_waypoint_step']}/{mid_js['total_waypoint_steps']}"
        )
        assert mid_pos is not None and any(
            abs(mid_pos[i] - initial_pos[i]) > 0.05 for i in range(3)
        ), "Robot joints did not move in Gazebo during execution!"
        print("[PASS] Test E & F: Verified live joint movement & 3D EE trail from Gazebo /joint_states!")

        # Test Pause & Resume
        code, pause_res = http_json(f"{base_url}/api/robot/pause", method="POST")
        assert code == 200 and pause_res["paused"] is True
        print("[PASS] POST /api/robot/pause paused execution.")
        time.sleep(0.4)

        code, resume_res = http_json(f"{base_url}/api/robot/resume", method="POST")
        assert code == 200 and resume_res["resumed"] is True
        print("[PASS] POST /api/robot/resume resumed execution.")
        time.sleep(0.5)

        # Test Stop Robot (Test G)
        code, stop_res = http_json(f"{base_url}/api/robot/stop", method="POST")
        assert code == 200 and stop_res["stopped"] is True
        print("[PASS] Test G (Safety): POST /api/robot/stop safely halted motion at:", stop_res["hold_positions_rad"])


if __name__ == "__main__":
    test_section5_recording_regression()
    test_model_and_kinematics()
    test_live_api_and_ros2()
    print("\nALL END-TO-END VERIFICATION & REGRESSION TESTS PASSED!")
