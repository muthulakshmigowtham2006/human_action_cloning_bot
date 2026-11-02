import math
import os

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


# ============================================================
# BEHAVIOR CLONING MODEL
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
# ROS 2 NODE
# ============================================================

class BCRobotController(Node):

    def __init__(self):

        super().__init__(
            "bc_robot_controller"
        )

        # ----------------------------------------------------
        # Publisher
        # ----------------------------------------------------

        self.publisher = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10
        )

        # ----------------------------------------------------
        # Load BC model
        # ----------------------------------------------------

        project_path = os.path.expanduser(
            "~/human_action_cloning_bot"
        )

        model_path = os.path.join(
            project_path,
            "models",
            "behavior_cloning_model.pth"
        )

        self.model = BehaviorCloningModel()

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        self.model.eval()

        self.get_logger().info(
            "Behavior Cloning model loaded successfully!"
        )

        # ----------------------------------------------------
        # Test human input
        # ----------------------------------------------------

        self.human_input = torch.tensor(
            [[0.0, 0.0, 0.0]],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Run once after 2 seconds
        # ----------------------------------------------------

        self.timer = self.create_timer(
            2.0,
            self.execute_action
        )

        self.executed = False


    # ========================================================
    # BEHAVIOR CLONING
    # ========================================================

    def predict_robot_position(self):

        with torch.no_grad():

            prediction = self.model(
                self.human_input
            )

        x = prediction[0][0].item()
        y = prediction[0][1].item()
        z = prediction[0][2].item()

        return x, y, z


    # ========================================================
    # INVERSE KINEMATICS
    # ========================================================

    def inverse_kinematics(
        self,
        x,
        y,
        z
    ):

        L1 = 0.9
        L2 = 0.8

        JOINT2_Z = 1.2

        # ----------------------------------------------------
        # Joint 1
        # ----------------------------------------------------

        joint1 = math.atan2(
            y,
            x
        )

        # ----------------------------------------------------
        # Horizontal distance
        # ----------------------------------------------------

        r = math.sqrt(
            x * x +
            y * y
        )

        # ----------------------------------------------------
        # Vertical distance
        # ----------------------------------------------------

        z_relative = z - JOINT2_Z

        # ----------------------------------------------------
        # Distance from joint 2
        # ----------------------------------------------------

        distance = math.sqrt(
            r * r +
            z_relative * z_relative
        )

        # ----------------------------------------------------
        # Reachability
        # ----------------------------------------------------

        if distance > L1 + L2:

            raise ValueError(
                "Target is outside robot workspace."
            )

        if distance < abs(L1 - L2):

            raise ValueError(
                "Target is inside minimum workspace."
            )

        # ----------------------------------------------------
        # Joint 3
        # ----------------------------------------------------

        cos_joint3 = (
            distance ** 2
            - L1 ** 2
            - L2 ** 2
        ) / (
            2 * L1 * L2
        )

        cos_joint3 = max(
            -1.0,
            min(1.0, cos_joint3)
        )

        joint3 = math.acos(
            cos_joint3
        )

        # ----------------------------------------------------
        # Joint 2
        # ----------------------------------------------------

        alpha = math.atan2(
            r,
            z_relative
        )

        beta = math.atan2(
            L2 * math.sin(joint3),
            L1 + L2 * math.cos(joint3)
        )

        joint2 = alpha - beta

        return (
            joint1,
            joint2,
            joint3
        )


    # ========================================================
    # SEND JOINT TRAJECTORY
    # ========================================================

    def send_joint_command(
        self,
        joint1,
        joint2,
        joint3
    ):

        trajectory = JointTrajectory()

        trajectory.joint_names = [
            "joint1",
            "joint2",
            "joint3"
        ]

        point = JointTrajectoryPoint()

        point.positions = [
            joint1,
            joint2,
            joint3
        ]

        point.time_from_start.sec = 3

        trajectory.points.append(
            point
        )

        self.publisher.publish(
            trajectory
        )

        self.get_logger().info(
            "Joint trajectory published!"
        )


    # ========================================================
    # MAIN ACTION
    # ========================================================

    def execute_action(self):

        if self.executed:
            return

        self.executed = True

        try:

            # ------------------------------------------------
            # BC
            # ------------------------------------------------

            robot_x, robot_y, robot_z = (
                self.predict_robot_position()
            )

            self.get_logger().info(
                f"BC Robot XYZ: "
                f"[{robot_x:.4f}, "
                f"{robot_y:.4f}, "
                f"{robot_z:.4f}]"
            )

            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            joint1, joint2, joint3 = (
                self.inverse_kinematics(
                    robot_x,
                    robot_y,
                    robot_z
                )
            )

            self.get_logger().info(
                f"Joint angles: "
                f"[{joint1:.4f}, "
                f"{joint2:.4f}, "
                f"{joint3:.4f}]"
            )

            self.get_logger().info(
                "Sending command to Gazebo..."
            )

            # ------------------------------------------------
            # ROS 2
            # ------------------------------------------------

            self.send_joint_command(
                joint1,
                joint2,
                joint3
            )

        except Exception as error:

            self.get_logger().error(
                f"Controller error: {error}"
            )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = BCRobotController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
