import torch
import torch.nn as nn
import math


# ============================================================
# 1. BEHAVIOR CLONING MODEL
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
# 2. LOAD TRAINED MODEL
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
# 3. ROBOT GEOMETRY
# ============================================================

# Distance from joint 2 to joint 3
L1 = 0.9

# Distance from joint 3 to end-effector joint
L2 = 0.8

# World Z position of joint 2
JOINT2_Z = 1.2


# ============================================================
# 4. INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(x, y, z):

    # --------------------------------------------------------
    # Joint 1
    # Rotates the arm around the vertical Z axis.
    # --------------------------------------------------------

    joint1 = math.atan2(y, x)

    # Horizontal distance from the Z axis
    r = math.sqrt(x * x + y * y)

    # Vertical distance from joint 2
    z_relative = z - JOINT2_Z

    # Distance from joint 2 to target
    distance = math.sqrt(
        r * r +
        z_relative * z_relative
    )

    # --------------------------------------------------------
    # Reachability check
    # --------------------------------------------------------

    if distance > L1 + L2:

        raise ValueError(
            f"Target is outside workspace. "
            f"Distance={distance:.4f}, "
            f"Maximum={L1 + L2:.4f}"
        )

    if distance < abs(L1 - L2):

        raise ValueError(
            f"Target is inside minimum workspace. "
            f"Distance={distance:.4f}, "
            f"Minimum={abs(L1 - L2):.4f}"
        )

    # --------------------------------------------------------
    # Joint 3
    #
    # Law of cosines
    # --------------------------------------------------------

    cos_joint3 = (
        distance ** 2
        - L1 ** 2
        - L2 ** 2
    ) / (2 * L1 * L2)

    # Numerical protection
    cos_joint3 = max(
        -1.0,
        min(1.0, cos_joint3)
    )

    joint3 = math.acos(cos_joint3)

    # --------------------------------------------------------
    # Joint 2
    #
    # IMPORTANT:
    # Our links extend along the Z direction.
    # Therefore the reference angle is measured from Z.
    # --------------------------------------------------------

    alpha = math.atan2(
        r,
        z_relative
    )

    beta = math.atan2(
        L2 * math.sin(joint3),
        L1 + L2 * math.cos(joint3)
    )

    joint2 = alpha - beta

    return joint1, joint2, joint3


# ============================================================
# 5. FORWARD KINEMATICS
# ============================================================

def forward_kinematics(joint1, joint2, joint3):

    # --------------------------------------------------------
    # Planar position
    #
    # Link 1 contribution
    # --------------------------------------------------------

    radial_distance = (
        L1 * math.sin(joint2)
    )

    vertical_distance = (
        L1 * math.cos(joint2)
    )

    # --------------------------------------------------------
    # Link 2 contribution
    # --------------------------------------------------------

    radial_distance += (
        L2 * math.sin(joint2 + joint3)
    )

    vertical_distance += (
        L2 * math.cos(joint2 + joint3)
    )

    # --------------------------------------------------------
    # Convert radial distance back to X/Y
    # --------------------------------------------------------

    x = radial_distance * math.cos(joint1)

    y = radial_distance * math.sin(joint1)

    z = JOINT2_Z + vertical_distance

    return x, y, z


# ============================================================
# 6. TEST HUMAN INPUT
# ============================================================

test_input = torch.tensor(
    [[0.0, 0.0, 0.0]],
    dtype=torch.float32
)


# ============================================================
# 7. BEHAVIOR CLONING PREDICTION
# ============================================================

with torch.no_grad():

    prediction = model(test_input)


robot_x = prediction[0][0].item()
robot_y = prediction[0][1].item()
robot_z = prediction[0][2].item()


print("\nHuman input:")
print(
    "Human XYZ:",
    test_input.numpy()[0]
)


print("\nBehavior Cloning prediction:")
print(
    "Robot XYZ:",
    [
        robot_x,
        robot_y,
        robot_z
    ]
)


# ============================================================
# 8. INVERSE KINEMATICS
# ============================================================

try:

    joint1, joint2, joint3 = inverse_kinematics(
        robot_x,
        robot_y,
        robot_z
    )

    print("\nInverse Kinematics result:")

    print(
        f"Joint 1: {joint1:.6f} rad"
    )

    print(
        f"Joint 2: {joint2:.6f} rad"
    )

    print(
        f"Joint 3: {joint3:.6f} rad"
    )

    print("\nJoint angles in degrees:")

    print(
        f"Joint 1: {math.degrees(joint1):.3f}°"
    )

    print(
        f"Joint 2: {math.degrees(joint2):.3f}°"
    )

    print(
        f"Joint 3: {math.degrees(joint3):.3f}°"
    )


    # ========================================================
    # 9. FORWARD KINEMATICS VERIFICATION
    # ========================================================

    fk_x, fk_y, fk_z = forward_kinematics(
        joint1,
        joint2,
        joint3
    )

    print("\nForward Kinematics verification:")

    print(
        f"Reconstructed X: {fk_x:.6f}"
    )

    print(
        f"Reconstructed Y: {fk_y:.6f}"
    )

    print(
        f"Reconstructed Z: {fk_z:.6f}"
    )


    # ========================================================
    # 10. POSITION ERROR
    # ========================================================

    error_x = abs(robot_x - fk_x)
    error_y = abs(robot_y - fk_y)
    error_z = abs(robot_z - fk_z)

    total_error = math.sqrt(
        error_x ** 2 +
        error_y ** 2 +
        error_z ** 2
    )

    print("\nPosition error:")

    print(
        f"X error: {error_x:.8f}"
    )

    print(
        f"Y error: {error_y:.8f}"
    )

    print(
        f"Z error: {error_z:.8f}"
    )

    print(
        f"Total position error: {total_error:.8f} m"
    )


    # ========================================================
    # 11. FINAL CHECK
    # ========================================================

    if total_error < 0.001:

        print(
            "\n✅ IK + Forward Kinematics verification PASSED!"
        )

    else:

        print(
            "\n⚠️ IK verification needs further checking."
        )


except ValueError as error:

    print("\n❌ IK ERROR:")
    print(error)
