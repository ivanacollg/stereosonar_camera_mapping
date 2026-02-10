# stereosonar_camera_merge
 This package takes in sensor infomation (orthogonal sonars, camera and odometry), performs sensor fussion and outputs a pointcloud with confidence values.

### Subscriber Topics:
- #### Camera image topic:  
    - Default Name: /camera/image_raw/compressed
    - Type: sensor_msgs/msg/CompressedImage
- #### Horizontal Sonar topic:
    - Default Name: /sonar_oculus_node/M750d/ping
    - Type: sonar_oculus/msg/OculusPing
- #### Vertical Sonar topic:
    - Default Name: /sonar_oculus_node/M1200d/ping
    - Type: sonar_oculus/msg/OculusPing
- #### Robot Odometry topic:
    - Default Name: /odom
    - Type: nav_msgs/msg/Odometry

### Publisher Topics:

- #### Segmented Image topic
    - Default Name: /sonar_camera_reconstruction/segmented_img/compressed
    - Type: sensor_msgs/msg/CompressedImage
- #### Horizontal Sonar features Image topic
    - Default Name: "/sonar_camera_merge/horizontal_feature_img/compressed"
    - Type: sensor_msgs/msg/CompressedImage
- #### Vertical Sonar features Image topic
    - Default Name: "/sonar_camera_merge/vertical_feature_img/compressed"
    - Type: sensor_msgs/msg/CompressedImage
- #### Merged Point Cloud topic
    - Default Name: /sonar_camera_merge/cloud
    - Type: sensor_msgs/msg/PointCloud2

### Parameter Files:
#### Monocular Camera parameters
- image_width: width of the image in pixels
- image_height: height of the image in pixels
- camera_matrix/data:
  - fx - focal length in pixels along the x axis
  - fy - focal length in pixels along the y axis
  - s - skew term (normally 0 unless your camera pixels are not perfectly rectangular)
  - (cx, cy) - coordinates of the principle point (optical center) in pixels
  - Complete camera matrix:
    <pre>
    [fx,  s,  cx,  
      0, fy,  cy,  
      0,  0,   1 ]
    </pre>
- distortion_coefficients/data: by default uses the plumb bob model, can be found through camera calibration
- Ts_c - Transformation Matrix from sonar to camera frame:
  - contains a rotation matrix R, a transformation matrix T, and a 4th row to make it homogeneous (never changes)
  - R is a 3x3 matrix with each column denoting the camera's x, y, and z axis directions in relation to the sonar axes in unit vector form.
  - T is a 1x3 matrix with each row denoting the x, y, and z coordinates of the origin of the camera frame in relation to the origin of the sonar frame in meters.
  - Complete transformation matrix:
    <pre>
     [R<sub>xx</sub>, R<sub>yx</sub>, R<sub>zx</sub>, T<sub>x</sub>,
      R<sub>xy</sub>, R<sub>yy</sub>, R<sub>zy</sub>, T<sub>y</sub>,
      R<sub>xz</sub>, R<sub>yz</sub>, R<sub>zz</sub>, T<sub>z</sub>,
       0,   0,   0,  1,]
    </pre>
  - model_name: Name of the trained yolov11 model found in folder models

#### Sonar parameters
- sonarRange: The maximum detection range in meters
- horizontalFOV: sonar horizontal FOV in degrees
- verticalAperture: The vertical angular coverage in degrees
- transformation: Trasform from horizontal to vertical sonar in meters

- threshold: detection strength threshold for the CFAR processing, higher thresholds remove both noise as well as weaker targets in favor of fewer, stronger targets
- CFAR/Ntc: number of training cells
- CFAR/Ngc: number of guard cells
- CFAR/Pfa: false alarm rate
- CFAR/rank: matrix rank

#### Stereo Merge parameters
Merge confidence values:
- conf_ss: 20.0 # stereo sonar pointcloud 
- conf_s:  2.0 # sonar-to-image pointcloud 
- conf_e:  1.0 # sonar expansion pointclo