import csv
import json
import os
import shutil
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Anatomical hand connections for visual overlay (21 landmarks)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),                   # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                   # Index
    (5, 9), (9, 10), (10, 11), (11, 12),              # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),            # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky & palm
]

# Palm reference indices: Wrist (0) + Index/Middle/Ring/Pinky MCP knuckles (5, 9, 13, 17)
PALM_INDICES: Tuple[int, ...] = (0, 5, 9, 13, 17)


def compute_palm_center(landmarks: List[List[float]]) -> List[float]:
    """Computes the 3D palm centre from wrist (0) and MCP knuckles (5, 9, 13, 17)."""
    if not landmarks or len(landmarks) < 21:
        return [0.0, 0.0, 0.0]
    px = sum(landmarks[idx][0] for idx in PALM_INDICES) / len(PALM_INDICES)
    py = sum(landmarks[idx][1] for idx in PALM_INDICES) / len(PALM_INDICES)
    pz = sum(landmarks[idx][2] for idx in PALM_INDICES) / len(PALM_INDICES)
    return [round(px, 5), round(py, 5), round(pz, 5)]


class CameraStreamer:
    """
    Thread-safe single-owner webcam capture, MediaPipe Tasks HandLandmarker runner,
    and bounded demonstration recorder.
    Guarantees that the exact landmarks displayed on the live video feed are
    appended to the active recording buffer from a single unified source.
    """

    def __init__(
        self,
        device_id: int = 0,
        project_root: Optional[str] = None,
        demonstrations_dir: Optional[str] = None,
    ):
        self.device_id = device_id
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.model_asset_path = os.path.join(
            self.project_root, "models", "mediapipe", "hand_landmarker.task"
        )
        self.demonstrations_dir = demonstrations_dir or os.path.join(
            self.project_root, "data", "demonstrations"
        )
        os.makedirs(self.demonstrations_dir, exist_ok=True)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # MediaPipe HandLandmarker state (VIDEO mode primary + IMAGE mode fallback)
        self.landmarker: Optional[vision.HandLandmarker] = None
        self.image_landmarker: Optional[vision.HandLandmarker] = None
        self.landmarker_loaded = False
        self.landmarker_error: Optional[str] = None
        self.last_mp_timestamp_ms = 0

        # Frame & tracking state
        self.latest_frame = None
        self.latest_jpeg: Optional[bytes] = None
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.last_error: Optional[str] = None

        # Live hand detection telemetry (single source of truth)
        self.hand_detected = False
        self.hand_count = 0
        self.total_hand_detections = 0
        self.latest_landmarks: Optional[List[List[float]]] = None
        self.latest_wrist: Optional[List[float]] = None
        self.latest_palm_center: Optional[List[float]] = None
        self.latest_landmark_time: float = 0.0

        # Bounded Recording State
        # recording_state: "idle" | "recording" | "processing" | "complete" | "error"
        self.recording_state = "idle"
        self.is_recording = False
        self.current_demo_id: Optional[str] = None
        self.recording_start_time: Optional[float] = None
        self.recording_start_iso: Optional[str] = None
        self.recorded_frames_buffer: List[Dict[str, Any]] = []
        self.live_trajectory_points: List[Dict[str, Any]] = []
        self.recording_detected_count = 0
        self.last_saved_demo: Optional[Dict[str, Any]] = None
        self.recording_error: Optional[str] = None

    def _init_landmarker(self) -> bool:
        """
        Initializes MediaPipe HandLandmarker in VIDEO mode (with responsive 0.35 thresholds)
        plus a secondary IMAGE-mode detector so fast hand movement never drops frames
        due to temporal tracking loss.
        """
        if self.landmarker is not None and self.landmarker_loaded:
            return True

        if not os.path.isfile(self.model_asset_path):
            self.landmarker_loaded = False
            self.landmarker_error = f"MediaPipe model not found at {self.model_asset_path}"
            return False

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
            video_options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(video_options)

            image_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_asset_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
            )
            self.image_landmarker = vision.HandLandmarker.create_from_options(image_options)

            self.landmarker_loaded = True
            self.landmarker_error = None
            self.last_mp_timestamp_ms = 0
            return True
        except Exception as exc:
            self.landmarker = None
            self.image_landmarker = None
            self.landmarker_loaded = False
            self.landmarker_error = f"Failed to initialize MediaPipe HandLandmarker: {exc}"
            return False

    def _close_landmarker(self):
        for attr in ("landmarker", "image_landmarker"):
            inst = getattr(self, attr, None)
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.landmarker_loaded = False

    def start(self) -> bool:
        with self.lock:
            if self.is_running and self.cap is not None and self.cap.isOpened():
                return True

            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            self._init_landmarker()

            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                self.last_error = f"Cannot open video device /dev/video{self.device_id}"
                self.cap = None
                self.is_running = False
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.fps = 0.0
            self.last_error = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _detect_landmarks_on_rgb(
        self, rgb_frame, now: float
    ) -> Optional[List[List[float]]]:
        """
        Single unified MediaPipe HandLandmarker extraction function used by both
        the live display overlay and the recording buffer.
        Uses VIDEO mode first and falls back to IMAGE mode if temporal tracking slips.
        """
        if not self.landmarker_loaded or self.landmarker is None:
            return None

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts_ms = int((now - self.start_time) * 1000)
            if ts_ms <= self.last_mp_timestamp_ms:
                ts_ms = self.last_mp_timestamp_ms + 1
            self.last_mp_timestamp_ms = ts_ms

            result = self.landmarker.detect_for_video(mp_image, ts_ms)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand = result.hand_landmarks[0]
                if len(hand) == 21:
                    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand]

            # Fallback to single-frame IMAGE detection if VIDEO tracker lost temporal lock
            if self.image_landmarker is not None:
                img_res = self.image_landmarker.detect(mp_image)
                if img_res.hand_landmarks and len(img_res.hand_landmarks) > 0:
                    hand = img_res.hand_landmarks[0]
                    if len(hand) == 21:
                        return [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand]
        except Exception as mp_exc:
            with self.lock:
                self.landmarker_error = f"MediaPipe inference error: {mp_exc}"

        return None

    def _append_frame_to_recording_locked(
        self, timestamp_sec: float, detected_landmarks: Optional[List[List[float]]]
    ) -> Tuple[int, float]:
        """
        Appends a single frame (with or without detected hand) to the active recording buffer.
        Must be called with self.lock held.
        """
        frame_idx = len(self.recorded_frames_buffer)
        has_hand = (
            detected_landmarks is not None
            and isinstance(detected_landmarks, list)
            and len(detected_landmarks) == 21
        )
        # Ensure strictly non-decreasing timestamps
        if self.recorded_frames_buffer:
            prev_ts = self.recorded_frames_buffer[-1]["timestamp_sec"]
            if timestamp_sec <= prev_ts:
                timestamp_sec = round(prev_ts + 0.001, 4)

        wrist = (
            [round(v, 5) for v in detected_landmarks[0]]
            if has_hand
            else None
        )
        palm = compute_palm_center(detected_landmarks) if has_hand else None

        if has_hand:
            self.recording_detected_count += 1

        entry = {
            "frame_index": frame_idx,
            "timestamp_sec": round(max(0.0, timestamp_sec), 4),
            "hand_detected": 1 if has_hand else 0,
            "wrist": wrist,
            "palm_center": palm,
            "landmarks": detected_landmarks if has_hand else None,
        }
        self.recorded_frames_buffer.append(entry)

        if has_hand:
            self.live_trajectory_points.append({
                "frame_index": frame_idx,
                "timestamp_sec": entry["timestamp_sec"],
                "wrist": wrist,
                "palm_center": palm,
            })
            # Cap in-memory live preview points to 600 to keep status payload lightweight
            if len(self.live_trajectory_points) > 600:
                self.live_trajectory_points = self.live_trajectory_points[-600:]

        return len(self.recorded_frames_buffer), entry["timestamp_sec"]

    def ingest_frame_landmarks(
        self,
        detected_landmarks: Optional[List[List[float]]],
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates live hand telemetry and (if recording is active) appends to the
        active recording buffer in a single atomic operation under self.lock.
        Used by _capture_loop and directly testable by regression tests.
        """
        now_ts = now if now is not None else time.time()
        with self.lock:
            self.frame_count += 1
            self.last_error = None

            has_hand = (
                detected_landmarks is not None
                and isinstance(detected_landmarks, list)
                and len(detected_landmarks) == 21
            )
            if has_hand:
                self.hand_detected = True
                self.hand_count = 1
                self.total_hand_detections += 1
                self.latest_landmarks = detected_landmarks
                self.latest_wrist = [round(v, 4) for v in detected_landmarks[0]]
                self.latest_palm_center = compute_palm_center(detected_landmarks)
                self.latest_landmark_time = now_ts
            else:
                self.hand_detected = False
                self.hand_count = 0
                self.latest_landmarks = None
                self.latest_wrist = None
                self.latest_palm_center = None

            is_rec = self.is_recording
            rec_len = 0
            rec_valid = 0
            rec_dur = 0.0
            if is_rec and self.recording_start_time is not None:
                rec_dur = max(0.0, now_ts - self.recording_start_time)
                rec_len, rec_dur = self._append_frame_to_recording_locked(
                    rec_dur, detected_landmarks if has_hand else None
                )
                rec_valid = self.recording_detected_count

            return {
                "is_recording": is_rec,
                "recorded_frames": rec_len,
                "detected_hand_frames": rec_valid,
                "recording_duration_sec": rec_dur,
                "hand_detected": has_hand,
            }

    def _draw_landmarks_and_hud(
        self,
        frame,
        landmarks_norm: Optional[List[List[float]]],
        is_recording: bool,
        rec_frames: int,
        rec_valid_frames: int,
        rec_duration: float,
        trajectory_pts: List[Dict[str, Any]],
    ):
        """Draws actual detected 21 hand landmarks, skeleton, recorded path trail, and HUD."""
        h, w, _ = frame.shape

        # Draw recorded wrist path trail directly on the camera feed when available
        if len(trajectory_pts) >= 2:
            trail = trajectory_pts[-120:]
            for idx in range(1, len(trail)):
                w0 = trail[idx - 1].get("wrist")
                w1 = trail[idx].get("wrist")
                if w0 and w1:
                    p0 = (int(w0[0] * w), int(w0[1] * h))
                    p1 = (int(w1[0] * w), int(w1[1] * h))
                    cv2.line(frame, p0, p1, (250, 204, 21), 2, cv2.LINE_AA)

        if landmarks_norm and len(landmarks_norm) == 21:
            pts_px = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks_norm]

            # Draw skeleton connections
            for idx_a, idx_b in HAND_CONNECTIONS:
                cv2.line(frame, pts_px[idx_a], pts_px[idx_b], (212, 182, 6), 2, cv2.LINE_AA)

            # Draw all 21 landmarks
            for idx, (px, py) in enumerate(pts_px):
                if idx == 0:
                    cv2.circle(frame, (px, py), 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 4, (129, 185, 16), -1, cv2.LINE_AA)

            wx, wy, wz = landmarks_norm[0]
            cv2.putText(
                frame,
                f"Wrist(0): ({wx:.2f}, {wy:.2f}, {wz:.2f})",
                (14, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (56, 239, 125),
                1,
                cv2.LINE_AA,
            )

        # Draw recording banner if recording is active
        if is_recording:
            cv2.rectangle(frame, (12, 44), (390, 78), (20, 20, 45), -1)
            cv2.circle(frame, (28, 61), 6, (50, 50, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"REC {rec_duration:.1f}s | {rec_frames}f ({rec_valid_frames} valid hand)",
                (42, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (140, 140, 255),
                1,
                cv2.LINE_AA,
            )

    def _capture_loop(self):
        fps_frames = 0
        last_fps_calc = time.time()

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            success, frame = self.cap.read()
            if not success:
                with self.lock:
                    self.last_error = "Failed to read frame from webcam"
                time.sleep(0.03)
                continue

            now = time.time()
            # Mirror view for natural user demonstration
            frame = cv2.flip(frame, 1)

            # 1. Extract 21 hand landmarks from current RGB frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_landmarks = self._detect_landmarks_on_rgb(rgb_frame, now)

            # 2. Update FPS counter
            fps_frames += 1
            if now - last_fps_calc >= 1.0:
                measured_fps = round(fps_frames / (now - last_fps_calc), 1)
                fps_frames = 0
                last_fps_calc = now
                with self.lock:
                    self.fps = measured_fps

            # 3. Ingest landmarks into both live state and active recording buffer atomically
            ingest_info = self.ingest_frame_landmarks(detected_landmarks, now=now)
            with self.lock:
                trail_snapshot = list(self.live_trajectory_points)

            # 4. Draw exact same landmarks & HUD onto display copy
            display_frame = frame.copy()
            self._draw_landmarks_and_hud(
                display_frame,
                detected_landmarks,
                ingest_info["is_recording"],
                ingest_info["recorded_frames"],
                ingest_info["detected_hand_frames"],
                ingest_info["recording_duration_sec"],
                trail_snapshot,
            )

            ret, jpeg = cv2.imencode(
                ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ret:
                with self.lock:
                    self.latest_frame = display_frame
                    self.latest_jpeg = jpeg.tobytes()

            time.sleep(0.008)

    def start_recording(self, allow_headless_test: bool = False) -> Dict[str, Any]:
        """Begins a new bounded user-controlled demonstration recording session."""
        with self.lock:
            if not allow_headless_test and (
                not self.is_running or self.cap is None or not self.cap.isOpened()
            ):
                raise RuntimeError("Cannot start recording: Camera capture is not active.")

            if self.is_recording:
                raise RuntimeError(
                    f"Recording session '{self.current_demo_id}' is already active."
                )

            now = time.time()
            dt_str = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
            ms_suffix = f"{int((now % 1) * 1000):03d}"
            demo_id = f"demo_{dt_str}_{ms_suffix}"

            self.is_recording = True
            self.recording_state = "recording"
            self.current_demo_id = demo_id
            self.recording_start_time = now
            self.recording_start_iso = datetime.fromtimestamp(now).isoformat()
            self.recorded_frames_buffer = []
            self.live_trajectory_points = []
            self.recording_detected_count = 0
            self.recording_error = None

            # If live camera is actively running and a hand is currently visible
            # (within the last 0.35s), capture it immediately as frame 0 so the
            # landmarks visible when clicking Start Recording are never missed.
            if (
                self.is_running
                and self.hand_detected
                and self.latest_landmarks is not None
                and len(self.latest_landmarks) == 21
                and 0.0 <= (now - self.latest_landmark_time) <= 0.35
            ):
                self._append_frame_to_recording_locked(0.0, list(self.latest_landmarks))

            return {
                "recording": True,
                "recording_state": self.recording_state,
                "demonstration_id": demo_id,
                "start_time": self.recording_start_iso,
                "initial_frames": len(self.recorded_frames_buffer),
                "message": f"Started recording demonstration {demo_id}",
            }

    def set_recording_state(self, state: str, error: Optional[str] = None):
        with self.lock:
            self.recording_state = state
            if error is not None:
                self.recording_error = error

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stops the active recording, validates that valid hand frames were captured,
        writes structured CSV + JSON + metadata, verifies the saved files on disk,
        and rejects empty/no-hand recordings explicitly.
        """
        with self.lock:
            if not self.is_recording or not self.current_demo_id:
                raise RuntimeError("No active recording session to stop.")

            demo_id = self.current_demo_id
            start_time = self.recording_start_time or time.time()
            start_iso = self.recording_start_iso or datetime.now().isoformat()
            stop_time = time.time()
            stop_iso = datetime.fromtimestamp(stop_time).isoformat()
            duration_sec = round(max(0.0, stop_time - start_time), 3)
            frames_copy = list(self.recorded_frames_buffer)
            trajectory_copy = list(self.live_trajectory_points)
            detected_count = self.recording_detected_count
            measured_fps = self.fps

            # Reset active recording flags immediately
            self.is_recording = False
            self.current_demo_id = None
            self.recording_start_time = None
            self.recording_start_iso = None
            self.recorded_frames_buffer = []
            self.recording_detected_count = 0

        total_frames = len(frames_copy)
        missing_count = total_frames - detected_count

        # Explicit rejection of empty or zero-hand recordings (Requirements 2.11, 2.12, 5)
        if total_frames == 0 or detected_count == 0:
            err_msg = (
                f"Recording '{demo_id}' rejected: captured {detected_count} valid hand "
                f"frames out of {total_frames} total frames ({duration_sec:.2f}s). "
                "Please ensure your hand is clearly visible with 21 MediaPipe landmarks "
                "in the camera view while recording."
            )
            with self.lock:
                self.recording_state = "error"
                self.recording_error = err_msg
            raise ValueError(err_msg)

        # Write files for valid recording
        demo_dir = os.path.join(self.demonstrations_dir, demo_id)
        os.makedirs(demo_dir, exist_ok=True)

        landmarks_csv_path = os.path.join(demo_dir, "landmarks.csv")
        raw_compat_csv_path = os.path.join(demo_dir, "raw_hand_landmarks.csv")
        trajectory_json_path = os.path.join(demo_dir, "trajectory.json")
        metadata_json_path = os.path.join(demo_dir, "metadata.json")

        try:
            # 1. Write full structured landmarks.csv (66 columns)
            header = ["frame_index", "timestamp_sec", "hand_detected"]
            for i in range(21):
                header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

            with open(landmarks_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for item in frames_copy:
                    row = [
                        item["frame_index"],
                        item["timestamp_sec"],
                        item["hand_detected"],
                    ]
                    lms = item["landmarks"]
                    if lms and len(lms) == 21:
                        for lm in lms:
                            row.extend([lm[0], lm[1], lm[2]])
                    else:
                        row.extend(["", "", ""] * 21)
                    writer.writerow(row)

            # 2. Write raw_hand_landmarks.csv (64 columns matching src/collect_data.py)
            compat_header = ["frame"]
            for i in range(21):
                compat_header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

            with open(raw_compat_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(compat_header)
                valid_idx = 0
                for item in frames_copy:
                    lms = item["landmarks"]
                    if lms and len(lms) == 21:
                        row = [valid_idx]
                        for lm in lms:
                            row.extend([lm[0], lm[1], lm[2]])
                        writer.writerow(row)
                        valid_idx += 1

            # 3. Write structured trajectory.json
            with open(trajectory_json_path, mode="w") as tj:
                json.dump(
                    {
                        "demonstration_id": demo_id,
                        "duration_seconds": duration_sec,
                        "frame_count": total_frames,
                        "detected_hand_frames": detected_count,
                        "missing_hand_frames": missing_count,
                        "frames": frames_copy,
                    },
                    tj,
                    indent=2,
                )

            # 4. Read back landmarks.csv to verify non-empty valid data on disk
            if not os.path.isfile(landmarks_csv_path) or os.path.getsize(landmarks_csv_path) == 0:
                raise RuntimeError("Saved landmarks.csv is missing or empty.")

            verified_rows = 0
            verified_hand_rows = 0
            with open(landmarks_csv_path, mode="r") as vf:
                reader = csv.DictReader(vf)
                if not reader.fieldnames or len(reader.fieldnames) != 66:
                    raise RuntimeError(
                        f"Saved landmarks.csv header invalid: got {len(reader.fieldnames or [])} cols"
                    )
                for r in reader:
                    verified_rows += 1
                    if str(r.get("hand_detected", "0")).strip() == "1":
                        verified_hand_rows += 1

            if verified_rows != total_frames or verified_hand_rows != detected_count:
                raise RuntimeError(
                    f"Read-back verification mismatch: wrote {total_frames}/{detected_count}, "
                    f"read back {verified_rows}/{verified_hand_rows}."
                )

            metadata = {
                "demonstration_id": demo_id,
                "valid": True,
                "start_time": start_iso,
                "stop_time": stop_iso,
                "duration_seconds": duration_sec,
                "frame_count": total_frames,
                "detected_hand_frames": detected_count,
                "missing_hand_frames": missing_count,
                "camera_fps": measured_fps,
                "landmarks_csv": landmarks_csv_path,
                "raw_compatible_csv": raw_compat_csv_path,
                "trajectory_json": trajectory_json_path,
                "processing_status": "unprocessed",
                "model_inference_status": "not_run",
                "playback_status": "not_played",
                "trajectory_preview": trajectory_copy,
                "error": None,
            }

            with open(metadata_json_path, mode="w") as f:
                json.dump(metadata, f, indent=2)

            with self.lock:
                self.last_saved_demo = metadata
                self.recording_state = "complete"
                self.recording_error = None

            return metadata

        except Exception as exc:
            if os.path.isdir(demo_dir):
                shutil.rmtree(demo_dir, ignore_errors=True)
            err_msg = f"Failed to save demonstration {demo_id}: {exc}"
            with self.lock:
                self.recording_state = "error"
                self.recording_error = err_msg
            raise RuntimeError(err_msg) from exc

    def get_recording_status(self) -> Dict[str, Any]:
        with self.lock:
            duration = (
                round(time.time() - self.recording_start_time, 2)
                if self.is_recording and self.recording_start_time
                else (
                    self.last_saved_demo["duration_seconds"]
                    if self.last_saved_demo
                    else 0.0
                )
            )
            return {
                "is_recording": self.is_recording,
                "recording_state": self.recording_state,
                "current_demonstration_id": self.current_demo_id,
                "recording_duration_seconds": duration,
                "recorded_frames": len(self.recorded_frames_buffer)
                if self.is_recording
                else (self.last_saved_demo["frame_count"] if self.last_saved_demo else 0),
                "detected_hand_frames": self.recording_detected_count
                if self.is_recording
                else (
                    self.last_saved_demo["detected_hand_frames"]
                    if self.last_saved_demo
                    else 0
                ),
                "live_trajectory": list(self.live_trajectory_points),
                "latest_demonstration": self.last_saved_demo,
                "error": self.recording_error,
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            connected = self.is_running and (
                self.cap is not None and self.cap.isOpened()
            )
            uptime = round(time.time() - self.start_time, 1) if connected else 0.0
            rec_duration = (
                round(time.time() - self.recording_start_time, 1)
                if self.is_recording and self.recording_start_time
                else 0.0
            )
            return {
                "connected": connected,
                "device_id": self.device_id,
                "fps": self.fps if connected else 0.0,
                "frame_count": self.frame_count,
                "uptime_seconds": uptime,
                "hand_detected": self.hand_detected if connected else False,
                "hand_count": self.hand_count if connected else 0,
                "total_hand_detections": self.total_hand_detections,
                "latest_wrist": self.latest_wrist if connected else None,
                "latest_palm_center": self.latest_palm_center if connected else None,
                "landmarker_loaded": self.landmarker_loaded,
                "landmarker_error": self.landmarker_error,
                "is_recording": self.is_recording,
                "recording_state": self.recording_state,
                "recording_duration_seconds": rec_duration,
                "recording_frames": len(self.recorded_frames_buffer)
                if self.is_recording
                else 0,
                "recording_valid_frames": self.recording_detected_count
                if self.is_recording
                else 0,
                "latest_demonstration_id": (
                    self.current_demo_id
                    or (
                        self.last_saved_demo["demonstration_id"]
                        if self.last_saved_demo
                        else None
                    )
                ),
                "error": self.last_error,
            }

    def stop(self) -> Dict[str, Any]:
        """
        Safely stops camera capture. If a recording is active with valid frames,
        finalizes and saves it first before releasing /dev/video0.
        """
        saved_demo = None
        if self.is_recording:
            try:
                saved_demo = self.stop_recording()
            except Exception as exc:
                with self.lock:
                    self.recording_error = str(exc)

        with self.lock:
            self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self._close_landmarker()
            self.latest_jpeg = None
            self.latest_frame = None
            self.hand_detected = False
            self.hand_count = 0
            self.fps = 0.0

        return {
            "stopped": True,
            "saved_active_recording": saved_demo,
        }


# Global singleton instance for single process camera access
camera_streamer = CameraStreamer(device_id=0)
