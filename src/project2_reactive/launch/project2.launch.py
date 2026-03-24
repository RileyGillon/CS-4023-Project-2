# Copyright 2026 Riley
#
# Licensed under the MIT License.
#
# project2.launch.py
# ------------------
# ROS 2 launch file for CS 4023 Project 2 – Reactive Robotics.
#
# Nodes launched:
#   1. reactive_controller  (project2_reactive) – priority-based reactive
#      controller that subscribes to /scan, /hazard_detection, /key_vel,
#      and /odom, then publishes velocity commands to /cmd_vel.
#
#   2. teleop_twist_keyboard – keyboard tele-operation node whose output
#      is REMAPPED from /cmd_vel to /key_vel so the controller can include
#      keyboard commands in its priority arbitration rather than letting
#      them bypass the Halt behaviour.
#
# How to launch (on a CSN Linux machine, after sourcing both the ROS 2
# overlay and the built workspace):
#
#   source /opt/ros/jazzy/setup.bash
#   source ~/ros2_ws/install/setup.bash
#   ros2 launch project2_reactive project2.launch.py
#
# A separate xterm window opens for the keyboard tele-op.  Click that
# window and follow the on-screen instructions.  Use conservative speeds
# (e.g. speed=0.1, turn=0.2) as recommended in the TurtleBot 4 manual.

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the LaunchDescription for Project 2."""

    # ------------------------------------------------------------------
    # 1. Reactive controller node
    #    Runs on the desktop computer; communicates with the TurtleBot 4
    #    via the ROS_DISCOVERY_SERVER connection established by robot-setup.sh
    # ------------------------------------------------------------------
    reactive_controller_node = Node(
        package='project2_reactive',
        executable='reactive_controller',
        name='reactive_controller',
        output='screen',
        emulate_tty=True,
    )

    # ------------------------------------------------------------------
    # 2. Keyboard tele-operation node
    #    teleop_twist_keyboard normally publishes to /cmd_vel, but here
    #    we remap its output to /key_vel so the reactive controller can
    #    place keyboard commands at priority level 2 in its arbitration
    #    pipeline (below Halt, above all autonomous behaviours).
    #
    #    stamped:=true  – publish TwistStamped (required by TurtleBot 4).
    #    prefix='xterm -e'  – open a dedicated terminal window for input.
    # ------------------------------------------------------------------
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_keyboard',
        output='screen',
        prefix='xterm -e',
        parameters=[{'stamped': True}],
        remappings=[('/cmd_vel', '/key_vel')],
    )

    return LaunchDescription([
        reactive_controller_node,
        teleop_node,
    ])
