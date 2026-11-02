import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    package_path = get_package_share_directory("human_robot_sim")

    controller_config = os.path.join(
        package_path,
        "config",
        "controllers.yaml"
    )

    urdf_file = os.path.join(
        package_path,
        "urdf",
        "human_robot_arm.urdf"
    )

    world_file = os.path.join(
        package_path,
        "worlds",
        "robot_world.sdf"
    )

    with open(urdf_file, "r") as file:
        robot_description = file.read()
    robot_description = robot_description.replace(
        "CONTROLLER_CONFIG_PATH",
        controller_config
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                "/opt/ros/jazzy/share/ros_gz_sim/launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": f"-r {world_file}"
        }.items()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description
            }
        ]
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "human_robot_arm",
            "-topic",
            "robot_description"
        ],
        output="screen"
    )
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager"
        ],
        output="screen"
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster,
        arm_controller
    ])
