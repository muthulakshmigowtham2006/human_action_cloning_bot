import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os


# -----------------------------
# MediaPipe Hand Landmarker Setup
# -----------------------------

model_path = "models/mediapipe/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Create Dataset Folder
# -----------------------------

os.makedirs("data/raw", exist_ok=True)

csv_file = "data/raw/hand_landmarks.csv"


# -----------------------------
# Create CSV Header
# -----------------------------

header = ["frame"]

for i in range(21):
    header.extend([
        f"landmark_{i}_x",
        f"landmark_{i}_y",
        f"landmark_{i}_z"
    ])


# -----------------------------
# Open Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


# -----------------------------
# Open CSV File
# -----------------------------

file_exists = os.path.exists(csv_file)

csvfile = open(
    csv_file,
    mode="a",
    newline=""
)

writer = csv.writer(csvfile)

if not file_exists:
    writer.writerow(header)


# -----------------------------
# Process Frames
# -----------------------------

timestamp_ms = 0
frame_number = 0

print("Recording started!")
print("Press 'q' to stop.")


while True:

    success, frame = cap.read()

    if not success:
        print("Error: Cannot read frame")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Update timestamp
    timestamp_ms += 33

    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    height, width, _ = frame.shape


    # -----------------------------
    # Extract and Save Coordinates
    # -----------------------------

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        row = [frame_number]

        for landmark_index, landmark in enumerate(
            hand_landmarks
        ):

            # Normalized coordinates
            x = landmark.x
            y = landmark.y
            z = landmark.z

            # Add coordinates to CSV row
            row.extend([x, y, z])

            # Pixel coordinates for drawing
            pixel_x = int(x * width)
            pixel_y = int(y * height)

            # Draw landmark
            cv2.circle(
                frame,
                (pixel_x, pixel_y),
                5,
                (0, 255, 0),
                -1
            )

        # Save one complete hand frame
        writer.writerow(row)

        frame_number += 1

        print(
            f"Saved frame {frame_number}"
        )


    # -----------------------------
    # Display Frame
    # -----------------------------

    cv2.imshow(
        "Human Action Data Collection",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

csvfile.close()

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Recording stopped.")
print(f"Data saved to: {csv_file}")
