from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


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
        .get('path_csv', '/dodger_ws/mpc_bicycle/global_path/centered_trajectory_100.csv')
    )


def generate_launch_description():
    pkg_share = get_package_share_directory('mpc_bicycle')
    param_file = os.path.join(pkg_share, 'config', 'mpc_params.yaml')
    csv_path = _load_csv_path(param_file)

    return LaunchDescription([
        # MPC
        Node(
            package='mpc_bicycle',
            executable='mpc_bicycle_node',
            name='mpc_bicycle_node',
            output='screen',
            parameters=[param_file],
        ),
        
        # CSV -> Path + Marker
        Node(
            package='mpc_bicycle',
            executable='csv_trajectory_viz_node',
            name='csv_trajectory_visualizer',
            parameters=[{
                'csv_path': csv_path,
                'frame_id': 'map',
                'publish_rate_hz': 1.0,
            }],
            arguments=['--ros-args', '--log-level', 'warn'],
            output='screen'
        ),

        # TF broadcaster so RViz sees map -> base_link
        Node(
            package='mpc_bicycle',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            parameters=[{
                'odom_topic': '/ego_racecar/odom',
                'frame_id': 'map',
                'child_frame_id': 'ego_racecar/base_link',
            }],
            arguments=['--ros-args', '--log-level', 'warn'],
            output='screen'
        )
    ])

# radius 5m, 600 waypoints, 1.2 m/s -> csv file
