# Python libraries
import numpy as np
from scipy.interpolate import interp1d
import threading
import warnings
#from line_profiler import LineProfiler

# OpenCV
import cv2
import open3d as o3d

# Custom libraries
from imaging_sonar import ImagingSonar
from monocular_camera import MonocularCamera
from stereosonarRGB import StereoSonarRGB

# Ros 
from tf.transformations import euler_from_quaternion, euler_matrix

class MergeFunctions:
    """
    A class to handle merging of sonar and camera data for underwater robotics applications.

    Attributes:
        Ts_c (np.ndarray): Transformation matrix from sonar to camera frame.
        minpixnum (int): Minimum number of pixels required for a valid segmented region.
        threshold_inv (int): Threshold value for image segmentation.
        sonar_msg (object): Holds sonar message data.
        pose (object): Holds robot pose data.
        image (np.ndarray): Holds the captured image data.
        color_map (np.ndarray): Predefined color mapping for visualizing clusters.
        xyz_aggregated (np.ndarray): Stores the aggregated 3D point cloud data.
        lock (threading.RLock): Lock for handling concurrency.
    """
    def __init__(self, Ts_c_horizontal, Ts_c_vertical, minpixnum, threshold_inv, boundry, yolo_segmentation, conf_ss, conf_s, conf_e):
        """
        Initializes the MergeFunctions class.

        Args:
            Ts_c (np.ndarray): Transformation matrix from sonar to camera frame.
            minpixnum (int): Minimum number of pixels for valid segmentation.
            threshold_inv (int): Threshold for image preprocessing.
            boundry (int): Boundry image threshold
            yolo_segmentation (bool): activate or not yolo segmentation (resomended True)
            conf_ss (float): confidence value of stee sonar pointcloud
            conf_s (float): confidence value of sonar-to-image pointcloud
            conf_e (float): confidence vaue of sonar expanded pointcloud
        """
        # sonar to camera transfromation
        self.translation_horizontal = Ts_c_horizontal[:3, 3]
        self.rotation_horizontal = Ts_c_horizontal[:3,:3]
        self.translation_vertical = Ts_c_vertical[:3, 3]
        self.rotation_vertical = Ts_c_vertical[:3,:3]
        # point confidence values
        self.conf_s = 1/conf_s
        self.conf_ss = 1/conf_ss
        self.conf_e = 1/conf_e
        # Initialize sensor information
        self.sonar_msg = None
        self.pose = None
        self.image = None

        self.minpixnum=minpixnum
        self.threshold_inv = threshold_inv
        self.boundry = boundry
        self.yolo_segmentation = yolo_segmentation

        self.color_map = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.25,0.75,0.25],
                    [1.0,1.0,0.0],
                    [.44,.62,.8118],
                    [0.8118,0.44,0.62],
                    [0.62,0.8118,0.44],
                    [0.75,0.25,0.25],
                    [0.25,0.75,0.25],
                    [0.25,0.25,0.75],
                    [0.1,0.45,0.7],
                    [0.7,0.45,0.1],
                    [0.45,0.7,0.1],
                    [0.45,0.1,0.7],
                    [0.1,0.7,0.45]])
                
                
        self.xyz_aggregated = np.zeros(0)
        # the threading lock
        self.lock = threading.Lock()
        self.stero_sonar = StereoSonarRGB()

    
    def set_camera_params(self, K, D, rgb_width, rgb_height, model_path):
        """
        Sets camera parameters.

        Args:
            K (np.ndarray): Intrinsic camera matrix.
            D (np.ndarray): Distortion coefficients.
            rgb_width (int): Width of the RGB image.
            rgb_height (int): Height of the RGB image.
        """
        self.monocular_camera = MonocularCamera(K, D, rgb_width, rgb_height, model_path)
    
    def set_horizontal_sonar_params(self, sonar_range, detector_threshold, vertical_FOV, sonar_features, fast_performance):
        """
        Sets sonar parameters.

        Args:
            sonar_range (float): Maximum range of the sonar.
            detector_threshold (float): Detection threshold for sonar processing.
            vertical_FOV (float): Vertical field of view of the sonar.
            sonar_features (bool): Determines if sonar features are highlited n the image.
            fast_performance (bool): Determines if fast or detailed perfomanc paramterers are used.
        """
        self.horizontal_sonar = ImagingSonar(sonar_range, detector_threshold, vertical_FOV, sonar_features, fast_performance)
        #self.lp = LineProfiler()
        #self.lp.add_function(self.imaging_sonar.get_sonar_scanline)

    def set_vertical_sonar_params(self, sonar_range, detector_threshold, vertical_FOV, sonar_features, fast_performance):
        """
        Sets sonar parameters.

        Args:
            sonar_range (float): Maximum range of the sonar.
            detector_threshold (float): Detection threshold for sonar processing.
            vertical_FOV (float): Vertical field of view of the sonar.
            sonar_features (bool): Determines if sonar features are highlited n the image.
            fast_performance (bool): Determines if fast or detailed perfomanc paramterers are used.
        """
        self.vertical_sonar = ImagingSonar(sonar_range, detector_threshold, vertical_FOV, sonar_features, fast_performance)

    def init_horizontal_CFAR(self, Ntc, Ngc, Pfa, rank):
        """
        Initializes CFAR (Constant False Alarm Rate) detection for sonar.

        Args:
            Ntc (int): Number of training cells.
            Ngc (int): Number of guard cells.
            Pfa (float): Probability of false alarm.
            rank (int): Ranking order for CFAR detection.
        """
        self.horizontal_sonar.init_CFAR(Ntc, Ngc, Pfa, rank)

    def init_vertical_CFAR(self, Ntc, Ngc, Pfa, rank):
        """
        Initializes CFAR (Constant False Alarm Rate) detection for sonar.

        Args:
            Ntc (int): Number of training cells.
            Ngc (int): Number of guard cells.
            Pfa (float): Probability of false alarm.
            rank (int): Ranking order for CFAR detection.
        """
        self.vertical_sonar.init_CFAR(Ntc, Ngc, Pfa, rank)

    '''
    def set_sensor_info(self, image, pose, sonar_msg):
        """
        Updates sensor information.

        Args:
            image (np.ndarray): Camera image.
            pose (object): Pose information of the robot.
            sonar_msg (object): Sonar message data.
        """
        with self.lock:
            self.image= image
            self.pose = pose
            self.sonar_msg = sonar_msg
    '''

    def rotate_cloud(self, t, R, new_cloud):
        """
        Rotates and transforms a point cloud from the body frame to the map frame.

        Args:
            t (np.ndarray): Translation vector (3x1).
            R (np.ndarray): Rotation matrix (3x3).
            new_cloud (np.ndarray): Input point cloud (Nx3).

        Returns:
            np.ndarray: Transformed point cloud (Nx3).
        """
        # Build Homogeneous tranform matrix
        H = np.row_stack((np.column_stack((R, t.T)), np.array([0, 0, 0, 1])))

        # Change of cloud points to homogeneous
        x = new_cloud[:, 0]
        z = new_cloud[:, 2]
        y = new_cloud[:, 1]
        xyzw = np.column_stack((x, y, z, np.ones_like(x)))
       
        # Transform points to map reference frame
        xyzw_map = np.matmul(H, xyzw.T).T
        xyzw_map[:,0] = np.divide(xyzw_map[:, 0], xyzw_map[:, 3])
        xyzw_map[:,1] = np.divide(xyzw_map[:, 1], xyzw_map[:, 3])
        xyzw_map[:,2] = np.divide(xyzw_map[:, 2], xyzw_map[:, 3])

        return xyzw_map[:, 0:3]


    def merge_data(self, image, pose, horizontal_sonar, vertical_sonar):
        """
        Merges sonar and camera data to generate a 3D point cloud.

        Returns:
            tuple:
                - np.ndarray: Aggregated 3D point cloud.
                - np.ndarray: Processed image with depth overlay.
                - object: Timestamp of the sonar message.
                - np.ndarray: Feature image from sonar processing.
        """

        if horizontal_sonar is not None and vertical_sonar is not None and image is not None and pose is not None:
            stamp = horizontal_sonar.header.stamp
            
            # Apply Yolo segmentation on the RGB image
            if self.yolo_segmentation:
                labels, labeled_image = self.monocular_camera.yolo_segment(image)
                #print("confidences")
                #print(confidences)
                thresholded_image = np.ones((self.monocular_camera.height, self.monocular_camera.width)).astype(np.uint8)*255
            else:
                # Filter image -> returns black and white image segmenting foreground and background
                thresholded_image = self.monocular_camera.preprocess(image, self.threshold_inv)
                # Segment Image -> returns labeled segemnted image and the labels
                labels, labeled_image = self.monocular_camera.segment_image(thresholded_image)

            #print(labels)
            
            # decode the compressed horizontal image
            imgHorizontal = np.frombuffer(horizontal_sonar.ping.data, np.uint8)
            imgHorizontal = np.array(cv2.imdecode(imgHorizontal, cv2.IMREAD_COLOR)).astype(
                np.uint8
            )
            imgHorizontal = cv2.cvtColor(imgHorizontal, cv2.COLOR_BGR2GRAY)

            # decode the compressed vertical image
            imgVertical = np.frombuffer(vertical_sonar.ping.data, np.uint8)
            imgVertical = np.array(cv2.imdecode(imgVertical, cv2.IMREAD_COLOR)).astype(
                np.uint8
            )
            imgVertical = cv2.cvtColor(imgVertical, cv2.COLOR_BGR2GRAY)
            
            if (labels>1).any():
                # Get Stereo Sonar matched points, and filtered sonar images
                stereo_pointcloud, horizontal_feature_image, vertical_feature_image, close_pointcloud, close_points, close_pointcloud_v, close_points_v = self.stero_sonar.run_stereo(imgHorizontal, imgVertical, stamp, horizontal_sonar)
    
                if stereo_pointcloud.shape[1] > 0:

                    # Initialize variables
                    depth_img_color = image.copy()

                    # Take only the area that is segmented in a binary image 
                    area_image = (labeled_image > 1).astype(np.uint8)

                    # Perform contour extraction on the created binary image
                    contours, hierarchy = cv2.findContours(
                        area_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    #rospy.loginfo(n_white_pix)
                    contours_list = contours

                    # Initialize variables to iterate through clusters
                    chosen_cluster = None
                    chosen_extended_coordinates = None
                    chosen_in_bound_indx = None
                    chosen_indx_coord = None

                    # Add stereo sonar points ############################################################################
                    #extended_coordinates = self.horizontal_sonar.get_extended_coordinates(close_points[:, 0], -np.radians(close_points[:,1]))
                    #pointcloud = extended_coordinates

                    # Transform points from sonar to camera reference frame
                    extended_coordinates = np.matmul(self.rotation_horizontal, stereo_pointcloud.T).T + self.translation_horizontal

                    # Get 2D coordinates
                    xyw = np.matmul(self.monocular_camera.K, extended_coordinates.T).T
                    #xyw[:,0] = np.divide(xyw[:, 0], xyw[:, 2])
                    #xyw[:,1] = np.divide(xyw[:, 1], xyw[:, 2])
                    #xyw[:,2] = np.divide(xyw[:, 2], xyw[:, 2])
                    xyw[:, :2] /= xyw[:, 2:3]  # Vectorized division
                    xy = np.round(xyw)[:,0:2].astype(np.int32)

                    # Change 2D coordinates to pixel coordinates
                    uv = np.column_stack((xy[:,1], xy[:,0]))

                    #uv coordenate indx that are valid
                    in_bound_indx = np.where((uv[:, 0] >= 0) & (uv[:,1]>=0) & (uv[:,0]<self.monocular_camera.height) & (uv[:,1]<self.monocular_camera.width))[0]

                    # uv coordinates that are valid
                    indx_coord = uv[in_bound_indx].astype(int)

                    # Valid 3D points
                    valid_3D_points = stereo_pointcloud[in_bound_indx]

                    # Create depth map 
                    depth_map = np.full((self.monocular_camera.height, self.monocular_camera.width, 3), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    depth_map[indx_coord[:,0], indx_coord[:,1]] = valid_3D_points  # (y, x) indexing

                    # Get overlap
                    sonar_points_image = np.zeros((self.monocular_camera.height, self.monocular_camera.width))
                    sonar_points_image[indx_coord[:,0], indx_coord[:,1]] = 255
                    sonar_points_image= sonar_points_image.astype(np.uint8)

                    #depth_img_color[np.where(sonar_points_image==255)]=self.color_map[0]*255

                    # Draw the outline
                    cv2.drawContours(depth_img_color, contours_list, -1, color=(0, 23, 223), thickness=3)
                    #depth_img_color[np.where(overlap_image==255)]=self.color_map[color_indx]*255

                    #print(indx_coord[:,1].shape[0])
                    #for i in range(indx_coord[:,1].shape[0]):
                    #    cv2.circle(depth_img_color,(indx_coord[i,1],indx_coord[i,0]), 3, (0,255,0), -1)

                    overlap_image = cv2.bitwise_and(sonar_points_image, area_image*255)

                    overlap_indices = np.argwhere(overlap_image==255)
                    #print(len(overlap_indices))
                    # Get Overlap 3D values from depthamp using ovalp indices
                    filtered_pointcloud = depth_map[overlap_indices[:,0], overlap_indices[:,1]]
                    # Get confidence scores at those pixel coordinates
                    conf = np.full((filtered_pointcloud.shape[0], 1), self.conf_ss)
                    #conf = confidence_image[overlap_indices[:,0], overlap_indices[:,1]]
                    # Append as 4th column to each point (x, y, z, confidence)
                    filtered_pointcloud_with_conf = np.hstack((filtered_pointcloud, conf))

                    ##################################################################

                    # Add close horizontal range points with expanded coordintes ####################################################
                    close_extended_coordinates = self.horizontal_sonar.get_extended_coordinates(close_points[:, 0], -np.radians(close_points[:,1]))

                    # Transform points from sonar to camera reference frame
                    extended_coordinates = np.matmul(self.rotation_horizontal, close_extended_coordinates.T).T + self.translation_horizontal

                    # Get 2D coordinates
                    xyw = np.matmul(self.monocular_camera.K, extended_coordinates.T).T
                    #xyw[:,0] = np.divide(xyw[:, 0], xyw[:, 2])
                    #xyw[:,1] = np.divide(xyw[:, 1], xyw[:, 2])
                    #xyw[:,2] = np.divide(xyw[:, 2], xyw[:, 2])
                    xyw[:, :2] /= xyw[:, 2:3]  # Vectorized division
                    xy = np.round(xyw)[:,0:2].astype(np.int32)

                    # Change 2D coordinates to pixel coordinates
                    uv = np.column_stack((xy[:,1], xy[:,0]))

                    #uv coordenate indx that are valid
                    in_bound_indx_prior = in_bound_indx
                    in_bound_indx = np.where((uv[:, 0] >= 0) & (uv[:,1]>=0) & (uv[:,0]<self.monocular_camera.height) & (uv[:,1]<self.monocular_camera.width))[0]
                    
                    # uv coordinates that are valid
                    indx_coord = uv[in_bound_indx].astype(int)
                    indx_coord_prior = indx_coord

                    # Valid 3D points
                    valid_3D_points = close_extended_coordinates[in_bound_indx]

                    # Create depth map 
                    depth_map = np.full((self.monocular_camera.height, self.monocular_camera.width, 3), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    depth_map[indx_coord[:,0], indx_coord[:,1]] = valid_3D_points  # (y, x) indexing

                    # Get overlap
                    sonar_points_image = np.zeros((self.monocular_camera.height, self.monocular_camera.width))
                    sonar_points_image[indx_coord[:,0], indx_coord[:,1]] = 255
                    sonar_points_image= sonar_points_image.astype(np.uint8)

                    #depth_img_color[np.where(sonar_points_image==255)]=self.color_map[0]*255

                    #print(indx_coord[:,1].shape[0])
                    #for i in range(indx_coord[:,1].shape[0]):
                    #    cv2.circle(depth_img_color,(indx_coord[i,1],indx_coord[i,0]), 3, (255,0,0), -1)

                    overlap_image = cv2.bitwise_and(sonar_points_image, area_image*255)
                    overlap_indices = np.argwhere(overlap_image==255)
                    
                    #print(len(overlap_indices))
                    # Get Overlap 3D values from depthamp using ovalp indices
                    filtered_pointcloud3 = depth_map[overlap_indices[:,0], overlap_indices[:,1]]
                    # Create a column of 70s
                    conf = np.full((filtered_pointcloud3.shape[0], 1), self.conf_s)

                    # Concatenate to form a (100, 4) array
                    filtered_pointcloud3_with_conf = np.hstack((filtered_pointcloud3, conf))
                    
                    ################
                    # Valid 3D points
                    valid_3D_points = extended_coordinates[in_bound_indx]

                    # Create depth map 
                    depth_map = np.full((self.monocular_camera.height, self.monocular_camera.width, 3), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    depth_map[indx_coord[:,0], indx_coord[:,1]] = valid_3D_points  # (y, x) indexing

                    
                    # Create depth map 
                    distance_map = np.full((self.monocular_camera.height, self.monocular_camera.width), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    distance_map[overlap_indices[:,0], overlap_indices[:,1]] = depth_map[overlap_indices[:,0], overlap_indices[:,1], 2] 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        column_means = np.nanmean(distance_map, axis=0)
                        column_means = np.nan_to_num(column_means, nan=0)  # Replace NaN with 0 or another default value
                    # Mean depth values are assigned to the entire column in the image
                    mean_distance_image = np.tile(column_means, (self.monocular_camera.height, 1))
                    mean_distance_mask = (mean_distance_image > 0).astype(np.uint8) * 255
                    overlap_image = cv2.bitwise_and(mean_distance_mask, area_image)*255

                    depth_image = np.zeros((self.monocular_camera.height, self.monocular_camera.width))
                    # Apply condition: If img1 == 255, assign value from img2; otherwise, assign 0
                    depth_image[overlap_image == 255] = mean_distance_image[overlap_image == 255]

                    # Add all points from RGB area found
                    indx = np.array(np.where(depth_image!=0)).T
                    
                    # REMOVE INDICES that have already been procesed before the expansion
                    ## Ensure arrays are contiguous
                    indx_contig = np.ascontiguousarray(indx)
                    indx_coord_prior_contig = np.ascontiguousarray(indx_coord_prior)
                    # Use structured arrays to treat each row as a single element
                    indx_struct = indx_contig.view([('', indx_contig.dtype)] * indx_contig.shape[1])
                    indx_coord_prior_struct = indx_coord_prior_contig.view([('', indx_coord_prior_contig.dtype)] * indx_coord_prior_contig.shape[1])
                    # Subtract B rows from A
                    result_struct = np.setdiff1d(indx_struct, indx_coord_prior_struct)
                    # Convert result back to regular ndarray
                    result = result_struct.view(indx.dtype).reshape(-1, indx.shape[1])
                    # Transpose back if needed
                    indx = tuple(result.T)
                    #print(indx.shape)
                    
                    # Det distance values of expanded points
                    final_distance_values = depth_image[indx]
                    final_distance_values = final_distance_values[:, np.newaxis] # reshape 
                    xyw = np.array([indx[1], indx[0], np.ones(indx[1].shape)])
                    s = 1
                    coord_3d = (1/s)*(final_distance_values)*(np.matmul(np.linalg.inv(self.monocular_camera.K),xyw).T)
                    
                    rows, cols = indx# indx = (row_indices, col_indices)

                    #for i in range(len(rows)):
                    #    cv2.circle(depth_img_color, (cols[i], rows[i]), 3, (0, 0, 255), -1)
                    
                    xyz_cloud = np.matmul((coord_3d-self.translation_horizontal), self.rotation_horizontal)    
                    
                    # Create a column of 60s
                    conf = np.full((xyz_cloud.shape[0], 1), self.conf_e)

                    # Concatenate to form a (100, 4) array
                    xyz_cloud_with_conf = np.hstack((xyz_cloud, conf))
                    
                    ################
                    ##################################################################

                    # Add close vertical range points with expanded coordintes ####################################################
                    close_extended_coordinates = self.vertical_sonar.get_extended_coordinates(close_points_v[:, 0], -np.radians(close_points_v[:,1]))
                    
                    # Transform points from sonar to camera reference frame
                    extended_coordinates = np.matmul(self.rotation_vertical, close_extended_coordinates.T).T + self.translation_vertical

                    # Get 2D coordinates
                    xyw = np.matmul(self.monocular_camera.K, extended_coordinates.T).T
                    #xyw[:,0] = np.divide(xyw[:, 0], xyw[:, 2])
                    #xyw[:,1] = np.divide(xyw[:, 1], xyw[:, 2])
                    #xyw[:,2] = np.divide(xyw[:, 2], xyw[:, 2])
                    xyw[:, :2] /= xyw[:, 2:3]  # Vectorized division
                    xy = np.round(xyw)[:,0:2].astype(np.int32)

                    # Change 2D coordinates to pixel coordinates
                    uv = np.column_stack((xy[:,1], xy[:,0]))

                    #uv coordenate indx that are valid
                    in_bound_indx_prior = in_bound_indx
                    in_bound_indx = np.where((uv[:, 0] >= 0) & (uv[:,1]>=0) & (uv[:,0]<self.monocular_camera.height) & (uv[:,1]<self.monocular_camera.width))[0]
                    
                    # uv coordinates that are valid
                    indx_coord = uv[in_bound_indx].astype(int)
                    indx_coord_prior = indx_coord

                    # Valid 3D points
                    valid_3D_points = close_extended_coordinates[in_bound_indx]

                    # Create depth map 
                    depth_map = np.full((self.monocular_camera.height, self.monocular_camera.width, 3), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    depth_map[indx_coord[:,0], indx_coord[:,1]] = valid_3D_points  # (y, x) indexing

                    # Get overlap
                    sonar_points_image = np.zeros((self.monocular_camera.height, self.monocular_camera.width))
                    sonar_points_image[indx_coord[:,0], indx_coord[:,1]] = 255
                    sonar_points_image= sonar_points_image.astype(np.uint8)

                    #depth_img_color[np.where(sonar_points_image==255)]=self.color_map[0]*255

                    #print(indx_coord[:,1].shape[0])
                    #for i in range(indx_coord[:,1].shape[0]):
                    #    cv2.circle(depth_img_color,(indx_coord[i,1],indx_coord[i,0]), 3, (255,255,0), -1)
                    
                    overlap_image = cv2.bitwise_and(sonar_points_image, area_image*255)
                    overlap_indices = np.argwhere(overlap_image==255)
                    
                    #print(len(overlap_indices))
                    # Get Overlap 3D values from depthamp using ovalp indices
                    filtered_pointcloud4 = depth_map[overlap_indices[:,0], overlap_indices[:,1]]
                    
                    new = np.column_stack((filtered_pointcloud4[:, 0], filtered_pointcloud4[:, 2], filtered_pointcloud4[:, 1]))
                    filtered_pointcloud4 = new
                    # Create a column of 70s
                    conf = np.full((filtered_pointcloud4.shape[0], 1), self.conf_s)
                    
                    # Concatenate to form a (100, 4) array
                    filtered_pointcloud4_with_conf = np.hstack((filtered_pointcloud4, conf))
        
                    #####################################
                    # Valid 3D points
                    valid_3D_points = extended_coordinates[in_bound_indx]

                    # Create depth map 
                    depth_map = np.full((self.monocular_camera.height, self.monocular_camera.width, 3), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    depth_map[indx_coord[:,0], indx_coord[:,1]] = valid_3D_points  # (y, x) indexing

                    # Create depth map 
                    distance_map = np.full((self.monocular_camera.height, self.monocular_camera.width), np.nan)  # initialize with NaNs
                    # Fill in the 3D values
                    distance_map[overlap_indices[:,0], overlap_indices[:,1]] = depth_map[overlap_indices[:,0], overlap_indices[:,1], 2] 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        row_means = np.nanmean(distance_map, axis=1)
                        row_means = np.nan_to_num(row_means, nan=0)  # Replace NaN with 0 or another default value
                    # Mean depth values are assigned to the entire column in the image
                    # Tile row means across the image width
                    mean_distance_image = np.tile(row_means[:, np.newaxis], (1, self.monocular_camera.width))
                    mean_distance_mask = (mean_distance_image > 0).astype(np.uint8) * 255
                    
                    xyz_cloud2 = np.array([[0.0, 0.0, 0.0]])
                    for label in labels[labels > 1]:
                        
                        #print(label)
                        # Take only the area that is segmented in a binary image 
                        area_image = (labeled_image == label).astype(np.uint8)

                        overlap_image = cv2.bitwise_and(sonar_points_image, area_image*255)
                        overlap_indices = np.argwhere(overlap_image==255)
                        
                        if len(overlap_indices>1):
                    
                            overlap_image = cv2.bitwise_and(mean_distance_mask, area_image)*255
                            depth_image = np.zeros((self.monocular_camera.height, self.monocular_camera.width))

                            # Apply condition: If img1 == 255, assign value from img2; otherwise, assign 0
                            depth_image[overlap_image == 255] = mean_distance_image[overlap_image == 255]

                            # Add all points from RGB area found
                            indx = np.array(np.where(depth_image!=0)).T

                            # REMOVE INDICES that have already been procesed before the expansion
                            ## Ensure arrays are contiguous
                            indx_contig = np.ascontiguousarray(indx)
                            indx_coord_prior_contig = np.ascontiguousarray(indx_coord_prior)
                            # Use structured arrays to treat each row as a single element
                            indx_struct = indx_contig.view([('', indx_contig.dtype)] * indx_contig.shape[1])
                            indx_coord_prior_struct = indx_coord_prior_contig.view([('', indx_coord_prior_contig.dtype)] * indx_coord_prior_contig.shape[1])
                            # Subtract B rows from A
                            result_struct = np.setdiff1d(indx_struct, indx_coord_prior_struct)
                            # Convert result back to regular ndarray
                            result = result_struct.view(indx.dtype).reshape(-1, indx.shape[1])
                            # Transpose back if needed
                            indx = tuple(result.T)
                            #print(indx.shape)

                            # Det distance values of expanded points
                            final_distance_values = depth_image[indx]
                            final_distance_values = final_distance_values[:, np.newaxis] # reshape 
                            xyw = np.array([indx[1], indx[0], np.ones(indx[1].shape)])
                            s = 1
                            coord_3d = (1/s)*(final_distance_values)*(np.matmul(np.linalg.inv(self.monocular_camera.K),xyw).T)

                            rows, cols = indx# indx = (row_indices, col_indices)

                            #for i in range(len(rows)):
                            #    cv2.circle(depth_img_color, (cols[i], rows[i]), 3, (0, 255, 255), -1)

                            new = np.matmul((coord_3d-self.translation_vertical), self.rotation_vertical)  
                            new2 = np.column_stack((new[:, 0], new[:, 2], new[:, 1]))
                            xyz_cloud2 = np.vstack((xyz_cloud2, new2))
                    

                    xyz_cloud2 = xyz_cloud2[1:]
                    # Create a column of 60s
                    conf = np.full((xyz_cloud2.shape[0], 1), self.conf_e)

                    # Concatenate to form a (100, 4) array
                    xyz_cloud_with_conf2 = np.hstack((xyz_cloud2, conf))
                    
                    #merged_cloud = filtered_pointcloud_with_conf
                    merged_cloud = np.vstack((filtered_pointcloud_with_conf, filtered_pointcloud3_with_conf, filtered_pointcloud4_with_conf, xyz_cloud_with_conf, xyz_cloud_with_conf2))




                    ## Transfrom to Sonar frame
                    merged_cloud = merged_cloud[:, 0:4] + np.array([0.3, 0.0, 0.0, 0.0]) #np.matmul((merged_cloud -self.translation), self.rotation)
                    ## Get translation
                    #t = np.array([pose.position.x,pose.position.y,pose.position.z])# Centered on robot center
                    ## Get rotation
                    ## Convert quaternion to rotation matrix
                    #quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    #roll, pitch, yaw = euler_from_quaternion(quaternion)
                    #R = euler_matrix(roll, 0, yaw)[:3, :3]
                    #xyz_cloud = self.rotate_cloud(t, R, xyz_cloud)
                    #merged_cloud = np.hstack((xyz_cloud, merged_cloud[:, 3].reshape(-1, 1)))

                    return merged_cloud, depth_img_color, stamp, horizontal_feature_image, vertical_feature_image
                else:
                    print("No overlap found")
                    return np.zeros(0), image, stamp, horizontal_feature_image, vertical_feature_image
            else:
                print("No segmentation found, applying original sonar")
                #stereo_pointcloud, horizontal_feature_image, vertical_feature_image = self.stero_sonar.run_stereo_original(imgHorizontal, imgVertical, stamp, horizontal_sonar)
                stereo_pointcloud, horizontal_feature_image, vertical_feature_image, _, _, _, _ = self.stero_sonar.run_stereo(imgHorizontal, imgVertical, stamp, horizontal_sonar)
                if stereo_pointcloud.shape[0] > 0:
                    # Create a column of 70s
                    conf = np.full((stereo_pointcloud.shape[0], 1), self.conf_ss)

                    # Concatenate to form a (100, 4) array
                    stereo_pointcloud_with_conf = np.hstack((stereo_pointcloud, conf))
                    merged_cloud = stereo_pointcloud_with_conf[:, 0:4] + np.array([0.3, 0.0, 0.0, 0.0])
                    
                    
                    return merged_cloud, image, stamp, horizontal_feature_image, vertical_feature_image
                else:
                    print("No overlap found")
                    return np.zeros(0), image, stamp, horizontal_feature_image, vertical_feature_image
        else:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)