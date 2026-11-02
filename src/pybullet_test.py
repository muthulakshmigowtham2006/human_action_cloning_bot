import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
physics_client = p.connect(p.GUI)

# Load PyBullet default models
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create ground
p.loadURDF("plane.urdf")

# Set gravity
p.setGravity(0, 0, -9.81)

# Load KUKA robot
robot_id = p.loadURDF(
    "kuka_iiwa/model.urdf",
    useFixedBase=True
)

print("PyBullet simulation started!")
print("Robot ID:", robot_id)

# Run simulation
while True:
    p.stepSimulation()
    time.sleep(1 / 240)
