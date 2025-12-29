# MPC Bicycle Controller

ROS 2 package for following CSV waypoints with a discrete-time MPC bicycle model.  
Designed based on F1TENTH ROS2 simulator, but adaptable to real vehicles. (But have to configure TF trees)

---

## Features

- **CSV-driven reference**  
  Loads `timestep, x, y, yaw, velocity` or `x, y, v` CSVs recorded via `odom_to_csv_node`.  
  Coordinates are interpreted in the same frame as the odometry topic  
  (default `/ego_racecar/odom`, typically `map`).

- **Ackermann output**  
  Publishes `AckermannDriveStamped` on `/drive` (or any topic via parameter)   
  with steering saturation and acceleration limits(i.e. constraints)

- **Nominal MPC**  
  Finite-horizon LTI MPC with optional curvature feed-forward, per-step constraints,   
  and diagnostics (`trace_flow`, saturation logging).

- **Visualization utilities**  
  `csv_trajectory_visualizer` publishes `nav_msgs/Path` + RViz markers straight from the CSV.

- **TF helper**  
  `odom_tf_broadcaster` mirrors `/ego_racecar/odom` into `/tf` so RViz sees `map → ego_racecar/base_link`.

- **CSV toolkit**  
  - `odom_to_csv_node` for logging odometry to CSV  
  - `csv_downsampler` for reducing dense files to ≈ N waypoints

---

## 📁 Package layout
```text
mpc_bicycle/
├─ launch/                # launch files
│   └─ mpc_bicycle.launch.py
├─ config/
│   └─ mpc_params.yaml    # MPC parameters
├─ global_path/           # example CSVs
├─ src/
│   ├─ mpc_bicycle_node.cpp
│   ├─ odom_to_csv.cpp
│   ├─ csv_trajectory_visualizer.cpp
│   ├─ csv_downsampler.cpp
│   └─ odom_tf_broadcaster.cpp
└─ scripts/
    └─ downsample_csv.py
```

## Building

This package is build on ROS2 workspace, `dodger_ws` by using `colcon build --symlink-install` 

### 1. Move to work space
```bash
cd /dodger_ws
```

### 2. Build
```bash
colcon build --symlink-install 
```

### 3. Environment setting

```bash
source install/setup.bash
```

## Launching
`mpc_bicycle.launch.py` use csv file as a reference path which is constitute with  `timestamp, x(global),y(global), yaw(rad/s), linear velocity(x, m/s)`
Command is like below:

```bash
ros2 launch mpc_bicycle mpc_bicycle.launch.py
```
This command automatically runs the nodes below: 
- `mpc_bicycle_node` : MPC 제어기를 실행하여 /drive 토픽으로 Ackermann 명령 퍼블리시

- `csv_trajectory_viz_node` : CSV 궤적을 시각화 (nav_msgs/Path 및 RViz Marker)

- `odom_tf_broadcaster` : /ego_racecar/odom → /tf 변환 브로드캐스트

## Coordinate Frames
**The written reference global trajectory csv file and robot's odometrty coordinate must be identical. **
The given code uses `frame_id = map` from  `/ego_racecar/odom`. So, its csv's coordinate uses `map`, so both uses same frame. 

If we use ordinary frame architecture which is
```text
map → odom → base_link → <SENSORS>
```
then have to follow the description belowl. 
```text
(a) map 프레임 기반 위치를 퍼블리시하는 토픽을 구독하거나

(b) mpc_bicycle_node 내부에서 CSV 좌표계와 odometry 좌표계를 변환하도록 수정해야 합니다.
```
 
## 📝 Parameters
All parameters is declared at `config/mpc_params.yaml`. 
Main compositions are like below: 

### Parameter	Description
- `path_csv`: absolute path of `CSV` file, (e.g.: `/dodger_ws/mpc_bicycle/global_path/centered_trajectory_100.csv`)
- `wheelbase`, `dt`, `horizon`: vehicle model parameters(wheel base), sampling time, prediction horizaon
- `v_ref`, `use_csv_speed`:	reference path or whether using csv-given velocity value.
- `delta_max_deg`, `a_max`:	steering angle, acceleration limit.
- `q_y`, `q_psi`, `q_v`: state weight(position, orientation, and velocity)
- `r_kappa`, `r_a`:	input command weight (sterring rate, acceleration)
- `use_curvature_ff`, `debug`, `trace_flow`, `cmd_topic`:	(extra options for debugging)

#### c.f.:
`path_csv` can be override when you start launch, or you can modify `FindPackageShare()` to automatically reference internal direcroties.


## CSV Utilities
This package is for reading/making trajectory based on csv files. 다.

### 1. `odom_to_csv_node`, Real-time trajectory recording. 

Write Odometry topic(`/ego_racecar/odom`) to `CSV` file. Run command is like below:

``` bash
ros2 run mpc_bicycle odom_to_csv_node \
  --ros-args -p odom_topic:=/ego_racecar/odom \
             -p output_path:=/tmp/path.csv
```

Result format of csv file:
``` text
 at /tmp/path.csv
timestep, x, y, yaw, velocity
```
### 2. `csv_downsampler`: sampling the written csv file
re-sample written csv file at 1 to certain number(`--count`). 

``` bash
ros2 run mpc_bicycle csv_downsampler \
  --input /tmp/path.csv \
  --output /tmp/path_100.csv \
  --count 100
```
This reduces computational cost for MPC when there is too much of csv points. 


# Conclusion
Therefore, the given tools framework is like below: 

1. `odom_to_csv_node`: Record trajectoy(based on teleop command) to `csv`
2. `csv_down_sampler`: Sample to lower resolution to lower computation cost. 
3. `mpc_bicycle.launch.py`: Conduct simple bicycle MPC control 
