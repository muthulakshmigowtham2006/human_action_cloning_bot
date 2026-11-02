import csv
import time
import torch
import torch.nn as nn
import pybullet as p
import pybullet_data


# --------------------------------------------------
# 1. Behavior Cloning Model
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

model = BehaviorCloningModel()

model.load_state_dict(
    torch.load(
        "models/behavior_cloning_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Behavior Cloning model loaded!")


# --------------------------------------------------
# 3. Load human trajectory
# --------------------------------------------------

trajectory_file = (
    "data/processed/normalized_wrist_trajectory.csv"
)

trajectory = []

with open(trajectory_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        human_position = [
            float(row["normalized_x"]),
            float(row["normalized_y"]),
            float(row["normalized_z"])
        ]

        trajectory.append(human_position)


print("Human trajectory loaded!")
print("Total frames:", len(trajectory))


# --------------------------------------------------
# 4. Start PyBullet
# --------------------------------------------------

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.loadURDF("plane.urdf")

p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF(
    "kuka_iiwa/model.urdf",
    useFixedBase=True
)

print("PyBullet simulation started!")
print("Robot ID:", robot_id)


# --------------------------------------------------
# 5. Replay trajectory using BC model
# --------------------------------------------------

print("\nStarting BC-controlled robot movement...\n")

for frame_number, human_position in enumerate(trajectory):

    # Convert human position to tensor
    human_input = torch.tensor(
        [human_position],
        dtype=torch.float32
    )

    # Predict robot position
    with torch.no_grad():
        prediction = model(human_input)

    robot_position = prediction[0].tolist()

    # Keep target inside a safe KUKA workspace
    target_position = [
        max(0.35, min(0.85, robot_position[0])),
        max(-0.45, min(0.45, robot_position[1])),
        max(0.15, min(0.85, robot_position[2]))
    ]

    # Inverse kinematics
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

    # Run simulation
    for _ in range(20):

        p.stepSimulation()
        time.sleep(1 / 240)

    # Display progress
    if frame_number % 10 == 0:

        print(
            f"Frame {frame_number}: "
            f"Human={human_position} "
            f"→ Robot={target_position}"
        )


print("\nBC-controlled trajectory replay completed!")


# --------------------------------------------------
# 6. Keep simulation open
# --------------------------------------------------

print("PyBullet simulation is still running.")
print("Press Ctrl+C in the terminal to stop.")

while True:

    p.stepSimulation()
    time.sleep(1 / 240)
