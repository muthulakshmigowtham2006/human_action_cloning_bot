import math
import os
import time

import cv2
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# LIVE HUMAN → ROBOT CONTROLLER
# ============================================================

class LiveHandRobotController(Node):

    def __init__(self):

        super().__init__("live_hand_robot_controller")

        # ----------------------------------------------------
        # ROS publisher
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
            "Behavior Cloning model loaded!"
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        model_file = os.path.join(
            project_path,
            "models",
            "mediapipe",
            "hand_landmarker.task"
        )

        base_options = python.BaseOptions(
            model_asset_path=model_file
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.get_logger().info(
            "MediaPipe loaded!"
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )

        self.get_logger().info(
            "Camera opened!"
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        self.initial_wrist = None

        # ----------------------------------------------------
        # Position smoothing
        # ----------------------------------------------------

        self.previous_robot_position = None

        self.smoothing_factor = 0.15

        # ----------------------------------------------------
        # Joint smoothing
        # ----------------------------------------------------

        self.previous_joints = None

        self.joint_smoothing = 0.20

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        self.start_time = time.time()

        self.last_command_time = 0.0

        self.command_interval = 0.10

        # ----------------------------------------------------
        # ROS timer
        #
        # 30 Hz camera processing
        # ----------------------------------------------------

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.process_frame
        )


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

        # Joint 1
        joint1 = math.atan2(
            y,
            x
        )

        # Horizontal distance
        r = math.sqrt(
            x * x +
            y * y
        )

        # Vertical distance
        z_relative = z - JOINT2_Z

        distance = math.sqrt(
            r * r +
            z_relative * z_relative
        )

        # Workspace check
        if distance > L1 + L2:
            return None

        if distance < abs(L1 - L2):
            return None

        # Joint 3
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

        # Joint 2
        alpha = math.atan2(
            r,
            z_relative
        )

        beta = math.atan2(
            L2 * math.sin(joint3),
            L1 + L2 * math.cos(joint3)
        )

        joint2 = alpha - beta

        # ----------------------------------------------------
        # Joint limits
        # ----------------------------------------------------

        joint1 = max(
            -3.14,
            min(3.14, joint1)
        )

        joint2 = max(
            -1.57,
            min(1.57, joint2)
        )

        joint3 = max(
            -2.5,
            min(2.5, joint3)
        )

        return (
            joint1,
            joint2,
            joint3
        )


    # ========================================================
    # SMOOTH ROBOT POSITION
    # ========================================================

    def smooth_position(
        self,
        position
    ):

        if self.previous_robot_position is None:

            self.previous_robot_position = position

            return position

        result = []

        for old, new in zip(
            self.previous_robot_position,
            position
        ):

            value = (
                old * (1.0 - self.smoothing_factor)
                +
                new * self.smoothing_factor
            )

            result.append(value)

        self.previous_robot_position = result

        return result


    # ========================================================
    # SMOOTH JOINT ANGLES
    # ========================================================

    def smooth_joints(
        self,
        joints
    ):

        if self.previous_joints is None:

            self.previous_joints = joints

            return joints

        result = []

        for old, new in zip(
            self.previous_joints,
            joints
        ):

            value = (
                old * (1.0 - self.joint_smoothing)
                +
                new * self.joint_smoothing
            )

            result.append(value)

        self.previous_joints = result

        return result


    # ========================================================
    # SEND TRAJECTORY
    # ========================================================

    def publish_joint_command(
        self,
        joints
    ):

        trajectory = JointTrajectory()

        trajectory.joint_names = [
            "joint1",
            "joint2",
            "joint3"
        ]

        point = JointTrajectoryPoint()

        point.positions = list(joints)

        # Give the controller time to smoothly reach
        # the updated target.

        point.time_from_start.sec = 0

        point.time_from_start.nanosec = 200000000

        trajectory.points.append(
            point
        )

        self.publisher.publish(
            trajectory
        )


    # ========================================================
    # CAMERA PROCESSING
    # ========================================================

    def process_frame(self):

        success, frame = self.camera.read()

        if not success:
            return

        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - self.start_time) * 1000
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # No hand
        # ----------------------------------------------------

        if not result.hand_landmarks:

            cv2.putText(
                frame,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "Human Action Cloning - Live Robot",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()

            return

        # ----------------------------------------------------
        # Wrist
        # ----------------------------------------------------

        wrist = result.hand_landmarks[0][0]

        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z

        # ----------------------------------------------------
        # Initial position
        # ----------------------------------------------------

        if self.initial_wrist is None:

            self.initial_wrist = (
                wrist_x,
                wrist_y,
                wrist_z
            )

            self.get_logger().info(
                "Initial wrist position captured."
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized_x = (
            wrist_x -
            self.initial_wrist[0]
        )

        normalized_y = (
            wrist_y -
            self.initial_wrist[1]
        )

        normalized_z = (
            wrist_z -
            self.initial_wrist[2]
        )

        # ----------------------------------------------------
        # BC input
        # ----------------------------------------------------

        model_input = torch.tensor(
            [[
                normalized_x,
                normalized_y,
                normalized_z
            ]],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # BC prediction
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = self.model(
                model_input
            )

        robot_x = prediction[0][0].item()
        robot_y = prediction[0][1].item()
        robot_z = prediction[0][2].item()

        # ----------------------------------------------------
        # Smooth XYZ
        # ----------------------------------------------------

        robot_x, robot_y, robot_z = (
            self.smooth_position(
                [
                    robot_x,
                    robot_y,
                    robot_z
                ]
            )
        )

        # ----------------------------------------------------
        # IK
        # ----------------------------------------------------

        joints = self.inverse_kinematics(
            robot_x,
            robot_y,
            robot_z
        )

        if joints is not None:

            # ------------------------------------------------
            # Smooth joints
            # ------------------------------------------------

            joints = self.smooth_joints(
                joints
            )

            current_time = time.time()

            # ------------------------------------------------
            # Controlled command rate
            # ------------------------------------------------

            if (
                current_time -
                self.last_command_time
                >=
                self.command_interval
            ):

                self.publish_joint_command(
                    joints
                )

                self.last_command_time = (
                    current_time
                )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.putText(
                frame,
                "ROBOT ACTIVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Robot XYZ: "
                f"{robot_x:.2f}, "
                f"{robot_y:.2f}, "
                f"{robot_z:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Joints: "
                f"{math.degrees(joints[0]):.0f}, "
                f"{math.degrees(joints[1]):.0f}, "
                f"{math.degrees(joints[2]):.0f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "TARGET OUT OF RANGE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Wrist visualization
        # ----------------------------------------------------

        height, width, _ = frame.shape

        px = int(
            wrist_x * width
        )

        py = int(
            wrist_y * height
        )

        cv2.circle(
            frame,
            (px, py),
            8,
            (0, 0, 255),
            -1
        )

        cv2.imshow(
            "Human Action Cloning - Live Robot",
            frame
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            self.get_logger().info(
                "Stopping live controller..."
            )

            rclpy.shutdown()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = LiveHandRobotController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.camera.release()

        cv2.destroyAllWindows()

        node.landmarker.close()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()
