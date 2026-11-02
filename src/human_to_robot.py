import csv
import pybullet as p
import pybullet_data
import time

# -----------------------------
# Load human trajectory
# -----------------------------

trajectory_file = "data/processed/normalized_wrist_trajectory.csv"

trajectory = []

with open(trajectory_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x = float(row["normalized_x"])
        y = float(row["normalized_y"])
        z = float(row["normalized_z"])

        trajectory.append([x, y, z])

print("Human trajectory loaded!")
print("Total trajectory points:", len(trajectory))


# -----------------------------
# Start PyBullet
# -----------------------------

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")

p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF(
    "kuka_iiwa/model.urdf",
    useFixedBase=True
)

print("PyBullet simulation started!")
print("Robot ID:", robot_id)


# -----------------------------
# Robot movement
# -----------------------------

for point in trajectory:

    human_x, human_y, human_z = point

    # Convert normalized human coordinates
    # into robot workspace coordinates

    robot_x = 0.6 + human_x * 0.5
    robot_y = human_y * 0.5
    robot_z = 0.5 - human_y * 0.5

    target_position = [
        robot_x,
        robot_y,
        robot_z
    ]

    # Calculate inverse kinematics
    joint_positions = p.calculateInverseKinematics(
        robot_id,
        6,
        target_position
    )

    # Move robot joints
    for joint_index in range(7):
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.POSITION_CONTROL,
            targetPosition=joint_positions[joint_index]
        )

    # Let simulation run
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1 / 240)


print("Human trajectory replay completed!")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)
