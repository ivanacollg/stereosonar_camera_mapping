import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # --------------------
    # Launch arguments
    # --------------------
    rectification_arg = DeclareLaunchArgument(
        'rectification',
        default_value='true'
    )

    mapping_arg = DeclareLaunchArgument(
        'mapping',
        default_value='true'
    )

    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='marina',
        description='Environment name (e.g. marina, tank_disks)'
    )

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true'
    )

    environment = LaunchConfiguration('environment')
    mapping = LaunchConfiguration('mapping')
    sim_time = LaunchConfiguration('use_sim_time')

    # --------------------
    # Package paths
    # --------------------
    merge_pkg = get_package_share_directory('stereosonar_camera_merge')
    mapping_pkg = get_package_share_directory('gpcoctomap')

    merge_launch = os.path.join(merge_pkg, 'launch', 'merge_node_launch.py')
    mapping_launch = os.path.join(mapping_pkg, 'launch', 'mapping_launch.py')

    # --------------------
    # Include launch files
    # --------------------
    merge_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(merge_launch),
        launch_arguments={
            'environment': environment,
            'odom_topic': '/odom',
            'use_sim_time': sim_time,
        }.items(),
    )

    mapping_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mapping_launch),
        condition=IfCondition(mapping),
        launch_arguments={
            'environment': environment,
            'use_sim_time': sim_time,
        }.items(),
    )

    # --------------------
    # Launch description
    # --------------------
    return LaunchDescription([
        rectification_arg,
        mapping_arg,
        environment_arg,
        use_sim_time,

        merge_node_launch,
        mapping_node_launch,
    ])
