import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare the argument
    environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='marina',
        description='Environment name (e.g., marina, tank_disks)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo/Bag) clock if true'
    )

    environment = LaunchConfiguration('environment')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Path setup
    pkg_path = get_package_share_directory('stereosonar_camera_merge')
    rviz_file = os.path.join(pkg_path, 'rviz', 'stereo_merge.rviz')

    # The function that runs *after* context is available
    def launch_setup(context):
        # Resolve the environment variable properly here
        env = environment.perform(context)

        # Parameter file path depends on environment
        monocular_param_file = os.path.join(pkg_path, 'config', f'monocular_camera_params_{env}.yaml')
        sonar_param_file = os.path.join(pkg_path, 'config', f'sonar_params.yaml')
        merge_param_file = os.path.join(pkg_path, 'config', f'stereo_merge_params_{env}.yaml')


        # If file doesn’t exist, fall back to default
        if not os.path.exists(monocular_param_file):
            print(f"[WARN] Param file for '{env}' not found — using default monocular_camera_params_marina.yaml")
            monocular_param_file = os.path.join(pkg_path, 'config', 'monocular_camera_params_marina.yaml')
        else:
            print(f"[INFO] Using monocular_param file for '{env}'")
        if not os.path.exists(merge_param_file):
            print(f"[WARN] Param file for '{env}' not found — using default stereo_merge_params_marina.yaml")
            merge_param_file = os.path.join(pkg_path, 'config', 'stereo_merge_params_marina.yaml')
        else:
            print(f"[INFO] Using stereo_merge_param file for '{env}'")


        # Nodes
        merge_node = Node(
            package='stereosonar_camera_merge',
            executable='stereo_merge_node.py',
            name='stereo_merge_node',
            parameters=[
                monocular_param_file,
                sonar_param_file,
                merge_param_file,
                {
                    'use_sim_time': use_sim_time,
                    'publish_rate': 5,
                    'horizontal_sonar_sub': '/sonar_oculus_node/M750d/ping',
                    'vertical_sonar_sub': '/sonar_oculus_node/M1200d/ping',
                    'odom_sub': '/odom',
                    'image_sub': '/camera/image_raw/compressed',
                    'segmented_image_pub': '/sonar_camera_merge/segmented_img/compressed',
                    'merge_cloud_pub': '/sonar_camera_merge/cloud',
                    'horizontal_feature_image_pub': '/sonar_camera_merge/horizontal_feature_img/compressed',
                    'vertical_feature_image_pub': '/sonar_camera_merge/vertical_feature_img/compressed',
                    
                    'environment': 'marina'
                }
            ],
            output='screen',
        )

        tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world2baselink',
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        )

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        )

        return [merge_node, tf_node, rviz_node]

    # Return launch description
    return LaunchDescription([
        environment_arg,
        use_sim_time_arg,
        OpaqueFunction(function=launch_setup),
    ])