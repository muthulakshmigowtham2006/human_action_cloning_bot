import cv2
import math
import time
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# BEHAVIOR CLONING MODEL
# ============================================================

class BehaviorCloningModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD BC MODEL
# ============================================================

model_path = "models/behavior_cloning_model.pth"

model = BehaviorCloningModel()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()

print("Behavior Cloning model loaded successfully!")


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

model_file = "models/mediapipe/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_file
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print("\nCamera started!")
print("Move your hand in front of the camera.")
print("Press Q to quit.")


# ============================================================
# NORMALIZATION
# ============================================================

initial_wrist = None


# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()

frame_number = 0


while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read frame.")

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # If hand detected
    # --------------------------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        wrist = hand[0]


        # ----------------------------------------------------
        # Raw MediaPipe wrist coordinates
        # ----------------------------------------------------

        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z


        # ----------------------------------------------------
        # Set first wrist position as origin
        # ----------------------------------------------------

        if initial_wrist is None:

            initial_wrist = (
                wrist_x,
                wrist_y,
                wrist_z
            )

            print("\nInitial wrist position captured!")


        # ----------------------------------------------------
        # Normalize trajectory
        # ----------------------------------------------------

        normalized_x = (
            wrist_x -
            initial_wrist[0]
        )

        normalized_y = (
            wrist_y -
            initial_wrist[1]
        )

        normalized_z = (
            wrist_z -
            initial_wrist[2]
        )


        # ----------------------------------------------------
        # Prepare BC input
        # ----------------------------------------------------

        model_input = torch.tensor(
            [[
                normalized_x,
                normalized_y,
                normalized_z
            ]],
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # BC prediction
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = model(
                model_input
            )


        robot_x = prediction[0][0].item()
        robot_y = prediction[0][1].item()
        robot_z = prediction[0][2].item()


        # ----------------------------------------------------
        # Display values
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Human: "
            f"{normalized_x:.3f}, "
            f"{normalized_y:.3f}, "
            f"{normalized_z:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Robot: "
            f"{robot_x:.3f}, "
            f"{robot_y:.3f}, "
            f"{robot_z:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Draw wrist
        # ----------------------------------------------------

        h, w, _ = frame.shape

        wrist_pixel_x = int(
            wrist_x * w
        )

        wrist_pixel_y = int(
            wrist_y * h
        )


        cv2.circle(
            frame,
            (
                wrist_pixel_x,
                wrist_pixel_y
            ),
            8,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            print(
                f"Human: "
                f"[{normalized_x:.3f}, "
                f"{normalized_y:.3f}, "
                f"{normalized_z:.3f}] "
                f"→ Robot: "
                f"[{robot_x:.3f}, "
                f"{robot_y:.3f}, "
                f"{robot_z:.3f}]"
            )


    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # Show camera
    # --------------------------------------------------------

    cv2.imshow(
        "Live Human Action -> Behavior Cloning",
        frame
    )


    frame_number += 1


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()

print("\nLive BC test completed.")
