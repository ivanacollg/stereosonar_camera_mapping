# stereosonar_camera_mapping

This repo contains the code derived from the paper **"Towards Versatile Opti-Acoustic Sensor Fusion and Volumetric Mapping for Safe Underwater Navigation" (2026)**, presents a volumetric mapping framework that fuses a stereo sonar pair with a monocular camera to enable safe underwater navigation under varying visibility conditions. 
  
You are viewing the ROS2 , ROS1 version can be viwed [Here]

# Dependencies
This codebase is ROS native and will require a ROS installation. It can be used without ROS, but will require some work.

- ROS Jazzy
- Python3

Dependencies:

    sudo pip install empy catkin_pkg "numpy<2.0" lark scipy opencv-python PyYAML ultralytics open3d

    sudo apt install ros-jazzy-pcl-ros ros-jazzy-image-transport-plugins


# Set Up
```
    mkdir -p catkin_ws/src
    cd catkin_ws/src
    git clone git@github.com:ivanacollg/stereosonar_camera_mapping.git
    cd ..
    catkin build
    source devel/setup.bash
```

# Data
This [data folder](https://drive.google.com/drive/folders/1qGcltj2GMxAta1ByrptC76vznj0ym_UQ?usp=sharing) contains  data used for the paper "Towards Versatile Opti-Acoustic Sensor Fusion and Volumetric Mapping for Safe Underwater Navigation" (2026), which presents a volumetric mapping framework that fuses a stereo sonar pair with a monocular camera to enable safe underwater navigation under varying visibility conditions. 

Each folder contains sensor data from each scenario shown in the paper: 
- tank_disks
- marina

Each folder contains original ROS1 .bag data and converted data to ROS2 folder. 

Additionally, trained image segmentation models for each scenario can be found in in the `yolo_models` folder. Download and place the model files in the `stereosonar_camera_merge/models` folder. 

# Running Code
Download segmentation yolo model and sesnor data before running the code. 
Then run the code:
```
    roslaunch stereosonar_camera_merge merge.launch
```
After starting the code, run the sesor data:
```
    rosbag play [sample].bag --clock
```
### Running code for different scenarious
Different scenarious contain slightly different parameters for sonar range, monocular camera calibration, etc. 

Change the `environment` argument in the `merge.launch file` to launch different parameters:
- marina (default)
- tank_disks (for tank tests)

Example:
```
    roslaunch sonar_camera_reconstruction merge.launch environment:=tank_disks
```


# Citations
If you use this repo or any of the data provided please cite the following work:
```
@misc{collado2025,
      title={Towards Versatile Opti-Acoustic Sensor Fusion and Volumetric Mapping for Safe Underwater Navigation}, 
      author={Ivana Collado-Gonzalez and John McConnell and Brendan Englot},
      year={2026},
}
```

# Documentation
This repo contains three packages:
### stereosonar_camer_merge
This package takes in sensor infomation (orthogonal sonars, camera and odometry), performs sensor fussion and outputs a pointcloud with confidence values. 
### gpcoctomap
This packages takes in a pointcloud with confidence values and performes confidence driven Gaussian Pocess Volumetric Mapping. 
### sonar_oculus
This package declares the oculus_sonar msg type used in this work. 


