import csv

input_file = "data/processed/normalized_wrist_trajectory.csv"
output_file = "data/processed/behavior_cloning_dataset.csv"

with open(input_file, "r") as infile:
    reader = csv.DictReader(infile)

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)

        # Dataset header
        writer.writerow([
            "human_x",
            "human_y",
            "human_z",
            "robot_x",
            "robot_y",
            "robot_z"
        ])

        count = 0

        for row in reader:
            human_x = float(row["normalized_x"])
            human_y = float(row["normalized_y"])
            human_z = float(row["normalized_z"])

            # Same mapping used in human_to_robot.py
            robot_x = 0.6 + human_x * 0.5
            robot_y = human_y * 0.5
            robot_z = 0.5 - human_y * 0.5

            writer.writerow([
                human_x,
                human_y,
                human_z,
                robot_x,
                robot_y,
                robot_z
            ])

            count += 1

print("Behavior Cloning dataset created!")
print("Total training samples:", count)
print("Saved to:", output_file)
