#!/usr/bin/env python3

# Python libraries
import numpy as np
import threading
import time
import os
import math
from scipy.spatial.transform import Rotation as R

# OpenCV
import cv2
import cv_bridge

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS Messages
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2 # Note: In ROS 2, this is often in sensor_msgs_py
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

# Message Filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Custom ROS Messages
from sonar_oculus.msg import OculusPing

# Custom libraries
# Assuming this library is compatible with Python 3 and available in your path
from stereosonar_camera_merge.stereo_merge import MergeFunctions

class MergeNode(Node):
    """
    This ROS node merges sonar, camera, and odometry data to create a fused point cloud 
    and segmented images for enhanced underwater perception.
    """
    def __init__(self):
        """
        Initializes the MergeNode class.
        """
        super().__init__('stereo_merge_node')
        
        # --- Parameter Declaration & Retrieval ---
        # Helper to declare and retrieve in one step
        def get_p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        self.publish_rate = get_p("publish_rate", 5)
        
        # Topic Names
        h_sonar_topic = get_p("horizontal_sonar_sub", "/sonar/horizontal")
        v_sonar_topic = get_p("vertical_sonar_sub", "/sonar/vertical")
        odom_topic = get_p("odom_sub", "/odom")
        image_topic = get_p("image_sub", "/camera/segmented_img/compressed")
        
        # Subscribers (using message_filters)
        # QoS setup for sensors (Best Effort is common for sensors, but Reliability is flexible here)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        horizontal_sonar_sub = Subscriber(self, OculusPing, h_sonar_topic)
        vertical_sonar_sub = Subscriber(self, OculusPing, v_sonar_topic)
        odom_sub  = Subscriber(self, Odometry, odom_topic)
        image_sub = Subscriber(self, CompressedImage, image_topic)

        self.br = tf2_ros.TransformBroadcaster(self)
        
        # Time Sync
        self.timeSync = ApproximateTimeSynchronizer(
            [horizontal_sonar_sub, vertical_sonar_sub, image_sub, odom_sub], 
            queue_size=10, 
            slop=0.35
        )
        self.timeSync.registerCallback(self.sonar_callback)

        # Publishers
        self.segmented_image_pub = self.create_publisher(CompressedImage, get_p("segmented_image_pub", "output/segmented"), 1)
        self.merge_cloud_pub = self.create_publisher(PointCloud2, get_p("merge_cloud_pub", "output/cloud"), 1)
        self.horizontal_feature_image_pub = self.create_publisher(CompressedImage, get_p("horizontal_feature_image_pub", "output/feat_horiz"), 1)
        self.vertical_feature_image_pub = self.create_publisher(CompressedImage, get_p("vertical_feature_image_pub", "output/feat_vert"), 1)

        # Matrices (ROS 2 params usually come as flat lists/arrays)
        # Note: Ensure your YAML file provides these as lists of floats
        ts_c_h_flat = get_p("Ts_c_horizontal", [0.0]*16)
        ts_c_v_flat = get_p("Ts_c_vertical", [0.0]*16)
        Ts_c_horizontal = np.array(ts_c_h_flat).reshape((4, 4))
        Ts_c_vertical = np.array(ts_c_v_flat).reshape((4, 4))

        # Other merge parameters
        self.fast_performance = get_p("fast_performance", False)
        conf_ss = get_p("conf_ss", 0.5)
        conf_s = get_p("conf_s", 0.5)
        conf_e = get_p("conf_e", 0.5)
        self.scale_factor = 0.5
        
        self.merge = MergeFunctions(Ts_c_horizontal, Ts_c_vertical, conf_ss, conf_s, conf_e)

        # Camera parameters
        model_name = get_p("model_name", "default_model")
        
        # In ROS 2, retrieving package paths is different
        from ament_index_python.packages import get_package_share_directory
        try:
            pkg_path = get_package_share_directory("stereosonar_camera_merge")
            model_path = os.path.join(pkg_path, "models", model_name)
        except Exception as e:
            self.get_logger().error(f"Could not find package path: {e}")
            model_path = ""

        rgb_width = get_p("image_width", 640)
        rgb_height = get_p("image_height", 480)
        
        K_flat = get_p("camera_matrix.data", [0.0]*9)
        D_flat = get_p("distortion_coefficients.data", [0.0]*5)
        K = np.array(K_flat).reshape((3, 3))
        D = np.array(D_flat)

        if self.fast_performance:
            rgb_height = int(rgb_height * self.scale_factor)
            rgb_width = int(rgb_width * self.scale_factor)
            K = K * self.scale_factor
            K[2, 2] = 1.0
            
        self.merge.set_camera_params(K, D, rgb_width, rgb_height, model_path)
       
        # Sonar Parameters
        sonar_range = get_p("sonarRange", 10.0) 
        vertical_FOV = get_p("verticalAperture", 10.0)
        sonar_features = get_p("sonar_features", True)
        
        det_thresh_h = get_p("horizontal.threshold", 0.5)
        self.merge.set_horizontal_sonar_params(sonar_range, det_thresh_h, vertical_FOV, sonar_features, self.fast_performance)
        
        det_thresh_v = get_p("vertical.threshold", 0.5)
        self.merge.set_vertical_sonar_params(sonar_range, det_thresh_v, vertical_FOV, sonar_features, self.fast_performance)
        
        # CFAR parameters (Horizontal)
        Ntc_h = get_p("horizontal.CFAR.Ntc", 10)
        Ngc_h = get_p("horizontal.CFAR.Ngc", 10)
        Pfa_h = get_p("horizontal.CFAR.Pfa", 0.1)
        rank_h = get_p("horizontal.CFAR.rank", 5)
        self.merge.init_horizontal_CFAR(Ntc_h, Ngc_h, Pfa_h, rank_h)

        # CFAR parameters (Vertical)
        Ntc_v = get_p("vertical.CFAR.Ntc", 10)
        Ngc_v = get_p("vertical.CFAR.Ngc", 10)
        Pfa_v = get_p("vertical.CFAR.Pfa", 0.1)
        rank_v = get_p("vertical.CFAR.rank", 5)
        self.merge.init_vertical_CFAR(Ntc_v, Ngc_v, Pfa_v, rank_v)

        # Laser fields
        self.laserFields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # Stats & State
        self.lock = threading.Lock()
        self.times = []
        self.bridge = cv_bridge.CvBridge()
        self.last_position = None
        self.last_yaw = None

        # Reusable messages
        self.header = Header()
        self.image_msg = CompressedImage()
        self.image_msg.format = "jpeg"

        self.get_logger().info("Stereo Merge Node Initialized")
        
    def sonar_callback(self, msgHorizontal, msgVertical, image_msg, odom_msg):
        """Called when all synced data is available."""
        pose = odom_msg.pose.pose
        x = pose.position.x
        y = pose.position.y

        # Extract yaw from quaternion
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        yaw = r.as_euler("xyz", degrees=False)[2]

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

        if distance > 0.05 or abs(dtheta) > math.radians(10):
            try:
                image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
                # In ROS 2, timestamps are objects, we pass the message header stamp directly
                data = (image, msgHorizontal, msgVertical, odom_msg.header.stamp)
                self.process_data(data)
                
                self.last_position = (x, y)
                self.last_yaw = yaw
            except Exception as e:
                self.get_logger().error(f"Error in callback: {e}")

    def process_data(self, data):
        """
        Worker function to process data.
        """
        image, horizontal_sonar, vertical_sonar, stamp = data
        if image is None or horizontal_sonar is None or vertical_sonar is None:
            return

        if self.fast_performance:
            image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        
        start = time.perf_counter()
        
        # Call the merge logic (Assuming merge.merge_data returns valid numpy arrays)
        point_cloud, segmented_image, stamp2, horizontal_feature_image, vertical_feature_image = self.merge.merge_data(
            image, horizontal_sonar, vertical_sonar
        )
        
        elapsed = time.perf_counter() - start

        # Publish results
        if point_cloud is not None and point_cloud.size > 0:
            self.times.append(elapsed)
            
            # Update header
            self.header.stamp = stamp2 # Ensure stamp2 is a valid ROS 2 Time or message stamp
            self.header.frame_id = "base_link"
            
            # create_cloud in ROS 2 (sensor_msgs_py)
            point_cloud = point_cloud.astype(np.float32)
            cloud_msg = pc2.create_cloud(self.header, self.laserFields, point_cloud)
            self.merge_cloud_pub.publish(cloud_msg)

        if segmented_image is not None and segmented_image.size > 0:
            # OpenCV encoding
            segmented_encoded = np.array(cv2.imencode('.jpg', segmented_image)[1]).tobytes()
            horizontal_feature_encoded = np.array(cv2.imencode('.jpg', horizontal_feature_image)[1]).tobytes()
            vertical_feature_encoded = np.array(cv2.imencode('.jpg', vertical_feature_image)[1]).tobytes()

            # Publish segmented image
            self.image_msg.header.stamp = stamp2
            self.image_msg.header.frame_id = "base_link" # Ensure frame_id is set
            
            self.image_msg.data = segmented_encoded
            self.segmented_image_pub.publish(self.image_msg)

            # Publish feature images
            self.image_msg.data = horizontal_feature_encoded
            self.horizontal_feature_image_pub.publish(self.image_msg)
            
            self.image_msg.data = vertical_feature_encoded
            self.vertical_feature_image_pub.publish(self.image_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MergeNode()
        node.get_logger().info("Start stereo sonar camera merge node...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # We can't use node.get_logger() if node creation failed, so use print
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()