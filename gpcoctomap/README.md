# gpcoctomap
This packages takes in a pointcloud with confidence values and performes confidence driven Gaussian Pocess Volumetric Mapping. 

### Subscriber Topics:
- #### Merged Point Cloud topic
    - Default Name: /sonar_camera_merge/cloud
    - Type: sensor_msgs/msg/PointCloud2

### Published Topics:
- #### Octupied Voxels
    - Default Name: /occupied_cells_vis_array
    - Type: gpcoctomap/MarkerArrayPub