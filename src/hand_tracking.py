import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe Hand Landmarker Setup
# -----------------------------

model_path = "models/mediapipe/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Open Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


# -----------------------------
# Process Camera Frames
# -----------------------------

timestamp_ms = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Error: Cannot read frame")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp
    timestamp_ms += 33

    # Detect hands
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # -----------------------------
    # Draw hand landmarks and
    # extract coordinates
    # -----------------------------

    if result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            print(
                f"\nHand {hand_index + 1} detected"
            )

            height, width, _ = frame.shape

            for landmark_index, landmark in enumerate(
                hand_landmarks
            ):

                # Normalized MediaPipe coordinates
                x_norm = landmark.x
                y_norm = landmark.y
                z_norm = landmark.z

                # Pixel coordinates for drawing
                x = int(x_norm * width)
                y = int(y_norm * height)

                # Print landmark coordinates
                print(
                    f"Landmark {landmark_index}: "
                    f"x={x_norm:.3f}, "
                    f"y={y_norm:.3f}, "
                    f"z={z_norm:.3f}"
                )

                # Draw landmark
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


    # -----------------------------
    # Display Output
    # -----------------------------

    cv2.imshow(
        "Human Action Cloning - Hand Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()
