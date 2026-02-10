from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    environment = LaunchConfiguration('environment')

    declare_environment = DeclareLaunchArgument(
        'environment',
        default_value='marina',
        description='Environment configuration to use'
    )

    # Package paths
    gpcoctomap_path = get_package_share_directory('gpcoctomap')
    config_file = os.path.join(
        gpcoctomap_path,
        'config',
        'gpcoctomap_' + environment.perform({}) + '.yaml'
    )

    gpcoctomap_node = Node(
        package='gpcoctomap',
        executable='gpcoctomap_server',
        name='gpcoctomap_server',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            config_file
        ]
    )

    return LaunchDescription([
        declare_environment,
        gpcoctomap_node
    ])
