import csv
import matplotlib.pyplot as plt


# -----------------------------
# File Path
# -----------------------------

input_file = (
    "data/processed/"
    "normalized_wrist_trajectory.csv"
)


# -----------------------------
# Read Trajectory Data
# -----------------------------

x_values = []
y_values = []

with open(
    input_file,
    mode="r"
) as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        x = float(row["normalized_x"])
        y = float(row["normalized_y"])

        x_values.append(x)
        y_values.append(y)


# -----------------------------
# Create Trajectory Plot
# -----------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    x_values,
    y_values,
    marker="o",
    markersize=3
)

# Starting point
plt.scatter(
    x_values[0],
    y_values[0],
    s=100,
    label="Start"
)

# Ending point
plt.scatter(
    x_values[-1],
    y_values[-1],
    s=100,
    label="End"
)

plt.xlabel("Normalized X")
plt.ylabel("Normalized Y")

plt.title(
    "Human Wrist Movement Trajectory"
)

plt.legend()

plt.grid(True)

plt.axis("equal")

plt.show()
