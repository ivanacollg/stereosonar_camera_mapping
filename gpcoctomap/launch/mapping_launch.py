from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch arguments
    environment = LaunchConfiguration('environment')

    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='marina',
        description='Environment configuration to use'
    )

    # Package paths
    pkg_path = get_package_share_directory('gpcoctomap')

    # The function that runs *after* context is available
    def launch_setup(context):
        # Resolve the environment variable properly here
        env = environment.perform(context)

        # Parameter file path depends on environment
        config_file = os.path.join(pkg_path, 'config', f'gpcoctomap_{env}.yaml')

        # If file doesn’t exist, fall back to default
        if not os.path.exists(config_file):
            print(f"[WARN] Param file for '{env}' not found — using default gpcoctomap_marina.yaml")
            config_file = os.path.join(pkg_path, 'config', 'gpcoctomap_marina.yaml')
        else:
            print(f"[INFO] Using config file for '{env}'")

        gpcoctomap_node = Node(
            package='gpcoctomap',
            executable='gpcoctomap_server',
            name='gpcoctomap_server',
            parameters=[
                config_file,
                {
                    'use_sim_time': True
                }
            ],
            output='screen',
        )

        return [gpcoctomap_node]

    return LaunchDescription([
        environment_arg,
        OpaqueFunction(function=launch_setup),
    ])
