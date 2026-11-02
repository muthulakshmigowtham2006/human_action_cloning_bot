import csv
import os


# -----------------------------
# File Paths
# -----------------------------

input_file = "data/raw/hand_landmarks.csv"
output_file = "data/processed/wrist_trajectory.csv"


# -----------------------------
# Create Output Directory
# -----------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


# -----------------------------
# Read Raw Data
# -----------------------------

trajectory = []

with open(
    input_file,
    mode="r"
) as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        frame = row["frame"]

        # Landmark 0 = Wrist
        x = row["landmark_0_x"]
        y = row["landmark_0_y"]
        z = row["landmark_0_z"]

        trajectory.append([
            frame,
            x,
            y,
            z
        ])


# -----------------------------
# Save Processed Trajectory
# -----------------------------

with open(
    output_file,
    mode="w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    # Header
    writer.writerow([
        "frame",
        "wrist_x",
        "wrist_y",
        "wrist_z"
    ])

    # Trajectory data
    writer.writerows(trajectory)


print("Trajectory processing completed!")
print(
    f"Processed data saved to: {output_file}"
)

print(
    f"Total trajectory frames: {len(trajectory)}"
)
