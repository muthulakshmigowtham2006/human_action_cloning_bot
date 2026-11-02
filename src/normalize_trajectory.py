import csv
import os


# -----------------------------
# File Paths
# -----------------------------

input_file = "data/processed/wrist_trajectory.csv"

output_file = (
    "data/processed/"
    "normalized_wrist_trajectory.csv"
)


# -----------------------------
# Read Trajectory
# -----------------------------

trajectory = []

with open(
    input_file,
    mode="r"
) as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        trajectory.append({
            "frame": int(row["frame"]),
            "x": float(row["wrist_x"]),
            "y": float(row["wrist_y"]),
            "z": float(row["wrist_z"])
        })


# -----------------------------
# Check Data
# -----------------------------

if not trajectory:
    print("No trajectory data found!")
    exit()


# -----------------------------
# Get Starting Position
# -----------------------------

start_x = trajectory[0]["x"]
start_y = trajectory[0]["y"]
start_z = trajectory[0]["z"]


# -----------------------------
# Normalize Trajectory
# -----------------------------

normalized_trajectory = []

for point in trajectory:

    normalized_x = point["x"] - start_x
    normalized_y = point["y"] - start_y
    normalized_z = point["z"] - start_z

    normalized_trajectory.append([
        point["frame"],
        normalized_x,
        normalized_y,
        normalized_z
    ])


# -----------------------------
# Save Normalized Data
# -----------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

with open(
    output_file,
    mode="w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "frame",
        "normalized_x",
        "normalized_y",
        "normalized_z"
    ])

    writer.writerows(
        normalized_trajectory
    )


print("Trajectory normalization completed!")
print(
    f"Normalized data saved to: {output_file}"
)

print(
    f"Total frames: "
    f"{len(normalized_trajectory)}"
)
