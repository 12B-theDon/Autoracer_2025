from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def _load_csv_path(param_file: str) -> str:
    try:
        with open(param_file, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {param_file}: {exc}") from exc

    return (
        data
        .get('mpc_bicycle_node', {})
        .get('ros__parameters', {})
        .get('path_csv', '/opp_detect_ws/src/object_detector/object_detector/rotate_path/rotated_trajectory_100.csv')
    )

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('object_detector'), 'config')
    config_path = os.path.join(config_dir, 'params.yaml')
    mpc_param_path = os.path.join(config_dir, 'mpc_params.yaml')

    return LaunchDescription([
        Node(
            package='object_detector',
            executable='scan_processor_node',
            name='scan_processor_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='knn_tracker_node',
            name='knn_tracker_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='publish_opp_odom',
            name='publish_opp_odom',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='mpc_bicycle_node',
            name='mpc_bicycle_node',
            output='screen',
            parameters=[mpc_param_path],
        )
    ])
