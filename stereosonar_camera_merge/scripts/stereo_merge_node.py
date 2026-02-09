#!/usr/bin/env python3

# Python libraries
import numpy as np
import threading
import time
import os
import math
#from line_profiler import LineProfiler
from scipy.spatial.transform import Rotation as R

# OpenCV
import cv2
import cv_bridge

# ROS libraries
import rospy
import rospkg

import concurrent.futures

# ROS Messages
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
import tf2_ros

#Custom ROS Messages
from sonar_oculus.msg import OculusPing

# Custom libraries
from stereo_merge import MergeFunctions

# Stats
import cProfile, pstats

class MergeNode:
    """
    This ROS node merges sonar, camera, and odometry data to create a fused point cloud 
    and segmented images for enhanced underwater perception.
    """
    def __init__(self, ns="~"):
        """
        Initializes the MergeNode class, setting up ROS publishers, subscribers, and processing parameters.

        Parameters:
        - ns (str): Namespace for retrieving ROS parameters. Default is "~" (private namespace).
        """
        rospy.init_node('stereo_merge_node', anonymous=True)
        self.publish_rate = rospy.get_param("~publish_rate", 5)

        # Subscribers
        horizontal_sonar_sub = Subscriber(rospy.get_param(ns + "horizontal_sonar_sub"), OculusPing)
        vertical_sonar_sub = Subscriber(rospy.get_param(ns + "vertical_sonar_sub"), OculusPing)
        odom_sub  = Subscriber(rospy.get_param(ns + "odom_sub"), Odometry)
        image_sub = Subscriber(rospy.get_param(ns + "image_sub"), CompressedImage)
        self.br = tf2_ros.TransformBroadcaster()
        
        # define time sync object for both sonar images
        self.timeSync = ApproximateTimeSynchronizer(
            [horizontal_sonar_sub, vertical_sonar_sub, image_sub, odom_sub], 1, 0.35
        )
        self.timeSync.registerCallback(self.sonar_callback)

        # Publishers
        self.segmented_image_pub = rospy.Publisher(rospy.get_param(ns + "segmented_image_pub"), CompressedImage, queue_size=1)
        self.merge_cloud_pub = rospy.Publisher(rospy.get_param(ns + "merge_cloud_pub"), PointCloud2, queue_size=1)
        self.horizontal_feature_image_pub = rospy.Publisher(rospy.get_param(ns + "horizontal_feature_image_pub"), CompressedImage, queue_size=1)
        self.vertical_feature_image_pub = rospy.Publisher(rospy.get_param(ns + "vertical_feature_image_pub"), CompressedImage, queue_size=1)

        # Sonar to Camera Transfromation Matrix
        Ts_c_horizontal = np.array(rospy.get_param(ns + "Ts_c_horizontal"), copy=False).reshape((4, 4))
        Ts_c_vertical = np.array(rospy.get_param(ns + "Ts_c_vertical"), copy=False).reshape((4, 4))

        # Get other merge parameters
        self.fast_performance = rospy.get_param(ns + "fast_performance")
        # Merge confidence Values
        conf_ss = rospy.get_param(ns + "conf_ss")
        conf_s = rospy.get_param(ns + "conf_s")
        conf_e = rospy.get_param(ns + "conf_e")
        self.scale_factor = 0.5
        self.merge = MergeFunctions(Ts_c_horizontal, Ts_c_vertical, conf_ss, conf_s, conf_e)

        # Get monocular camera parameters
        model_name = rospy.get_param(ns + "model_name")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("stereosonar_camera_merge") 
        model_path = os.path.join(pkg_path, "models", model_name)  
        rgb_width = rospy.get_param(ns + "image_width")
        rgb_height = rospy.get_param(ns + "image_height")
        K = np.array(rospy.get_param(ns + "camera_matrix/data"), copy=False).reshape((3, 3))
        D = np.array(rospy.get_param(ns + "distortion_coefficients/data"), copy=False)
        if self.fast_performance:
            rgb_height = int(rgb_height*self.scale_factor)
            rgb_width = int(rgb_width*self.scale_factor)
            K = K*self.scale_factor
            K[2,2] = 1
        self.merge.set_camera_params(K, D, rgb_width, rgb_height, model_path)
       
        # Sonar Prameters
        sonar_range = rospy.get_param(ns + "sonarRange") # default value, reads in new value from msg
        vertical_FOV = rospy.get_param(ns + "verticalAperture")
        sonar_features = rospy.get_param(ns + "sonar_features")
        detector_threshold = rospy.get_param(ns + "horizontal/threshold")
        self.merge.set_horizontal_sonar_params(sonar_range, detector_threshold, vertical_FOV, sonar_features, self.fast_performance)
        detector_threshold = rospy.get_param(ns + "vertical/threshold")
        self.merge.set_vertical_sonar_params(sonar_range, detector_threshold, vertical_FOV, sonar_features, self.fast_performance)
        #read in CFAR parameters
        Ntc = rospy.get_param(ns  + "horizontal/CFAR/Ntc")
        Ngc = rospy.get_param(ns  + "horizontal/CFAR/Ngc")
        Pfa = rospy.get_param(ns  + "horizontal/CFAR/Pfa")
        rank = rospy.get_param(ns + "horizontal/CFAR/rank")
        # define the CFAR detector
        self.merge.init_horizontal_CFAR(Ntc, Ngc, Pfa, rank)
        #read in CFAR parameters
        Ntc = rospy.get_param(ns  + "vertical/CFAR/Ntc")
        Ngc = rospy.get_param(ns  + "vertical/CFAR/Ngc")
        Pfa = rospy.get_param(ns  + "vertical/CFAR/Pfa")
        rank = rospy.get_param(ns + "vertical/CFAR/rank")
        # define the CFAR detector
        self.merge.init_vertical_CFAR(Ntc, Ngc, Pfa, rank)

        # define laser fields for fused point cloud
        self.laserFields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # the threading lock
        self.lock = threading.Lock()
        self.num_workers = 4 
        self.latest_data = None  # Store the latest sensor data
        self.hz = 0.0
        self.calls = 0.0
        self.times = []

        
        # CV bridge
        self.bridge = cv_bridge.CvBridge()

        # Initialize sensor information
        self.last_position = None
        self.last_yaw = None

        # Reusable header object
        self.header = Header()
        self.image_msg = CompressedImage()
        self.image_msg.format = "jpeg"
        
    def __del__(self):
        # After collecting all times
        avg_time = np.mean(self.times)
        std_time = np.std(self.times)    
        
    def sonar_callback(self, msgHorizontal, msgVertical, image_msg, odom_msg)->None:
        """Called when all synced data is available. Submit job to thread pool."""
        pose = odom_msg.pose.pose
        # Extract position
        x = pose.position.x
        y = pose.position.y

        # Extract yaw from quaternion
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        yaw = r.as_euler("xyz", degrees=False)[2]  # roll, pitch, yaw → index 2

        if self.last_position is None:
            self.last_position = (x, y)
            self.last_yaw = yaw
            return

        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        distance = math.sqrt(dx*dx + dy*dy)

        dtheta = yaw - self.last_yaw
        # Normalize rotation to [-pi, pi]
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

        if distance > 0.05 or abs(dtheta) > math.radians(10):  # 5 cm or 10 deg
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
            # Bundle data and push to executor
            data = (image, msgHorizontal, msgVertical, odom_msg.header.stamp)
            self.process_data(data)
            
            self.last_position = (x, y)
            self.last_yaw = yaw


    def process_data(self, data):
        """
        Worker function that runs merge_data() in parallel.
        """
        image, horizontal_sonar, vertical_sonar, stamp = data
        if image is None or  horizontal_sonar is None or vertical_sonar is None:
            return  # No new data available, skip this iteration

        #self.lp.enable()
        # Run merging process
        if self.fast_performance:
            # Resize the image
            image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        # Start the timer to measure merge_data execution time
        start = time.perf_counter()  # high-res timer
        point_cloud, segmented_image, stamp2, horizontal_feature_image, vertical_feature_image = self.merge.merge_data(image, horizontal_sonar, vertical_sonar)
        elapsed = time.perf_counter() - start
        #self.lp.disable()
        #self.lp.print_stats()

        # Publish results immediately
        if point_cloud.size > 0:
            self.times.append(elapsed)
            self.calls += 1
            #rospy.loginfo(f"Average merge_data() time: {np.mean(self.times)} s")

            # Log the time it took to execute
            self.header.stamp = stamp2
            self.header.frame_id = "base_link"
            cloud_msg = pc2.create_cloud(self.header, self.laserFields, point_cloud)
            self.merge_cloud_pub.publish(cloud_msg)


        if segmented_image.size > 0:
            segmented_encoded = np.array(cv2.imencode('.jpg', segmented_image)[1]).tobytes()
            horizontal_feature_encoded = np.array(cv2.imencode('.jpg', horizontal_feature_image)[1]).tobytes()
            vertical_feature_encoded = np.array(cv2.imencode('.jpg', vertical_feature_image)[1]).tobytes()

            # Publish segmented image
            self.image_msg.header.stamp = stamp2
            self.image_msg.data = segmented_encoded
            self.segmented_image_pub.publish(self.image_msg)

            # Publish feature image
            self.image_msg.data = horizontal_feature_encoded
            self.horizontal_feature_image_pub.publish(self.image_msg)
            # Publish feature image
            self.image_msg.data = vertical_feature_encoded
            self.vertical_feature_image_pub.publish(self.image_msg)


if __name__ == "__main__":
    try:
        node = MergeNode()
        rospy.loginfo("Start stereo sonar camera merge node...")
        rospy.spin()  # no manual loop needed
    except rospy.ROSInterruptException:
        rospy.logwarn("ROS Interrupt Exception. Shutting down stereo_merge_node.")
