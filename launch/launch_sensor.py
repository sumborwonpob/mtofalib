from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mtofalib',
            executable='uvc_publisher.py',
            name='uvc_publisher',
            output='screen',
        ),
        Node(
            package='mtofalib',
            executable='mtof_publisher',
            name='mtof_publisher',
            output='screen',
        ),
])