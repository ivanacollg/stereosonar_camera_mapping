#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Assumed header names based on previous context
#include "markerarray_pub.h" 
#include "gpcoctomap.h"

using std::placeholders::_1;
using namespace std::chrono_literals; // Useful for time durations

class GpcoctomapHandler : public rclcpp::Node {
public:
    GpcoctomapHandler(const std::string& cloud_topic)
        : Node("gpcoctomap_server"),
          frame_id_("map"), 
          first_(true), 
          updated_(false),
          position_change_thresh_(0.1), 
          orientation_change_thresh_(0.2) 
    {
        // --- Parameter Declaration & Retrieval ---
        this->declare_parameter("topic", "/occupied_cells_vis_array");
        this->declare_parameter("topic_free", "/free_cells_vis_array");
        this->declare_parameter("max_range", -1.0);
        this->declare_parameter("resolution", 0.1);
        this->declare_parameter("block_depth", 4);
        this->declare_parameter("sf2", 1.0);
        this->declare_parameter("ell", 1.0);
        this->declare_parameter("free_resolution", 0.1);
        this->declare_parameter("ds_resolution", 0.1);
        this->declare_parameter("free_thresh", 0.3);
        this->declare_parameter("occupied_thresh", 0.7);
        this->declare_parameter("min_z", 0.0);
        this->declare_parameter("max_z", 0.0);
        this->declare_parameter("original_size", true);
        this->declare_parameter("noise", 0.01);
        this->declare_parameter("l", 100.0);
        this->declare_parameter("min_var", 0.001);
        this->declare_parameter("max_var", 1000.0);
        this->declare_parameter("max_known_var", 0.02);

        map_topic_occ_ = this->get_parameter("topic").as_string();
        map_topic_free_ = this->get_parameter("topic_free").as_string();
        max_range_ = this->get_parameter("max_range").as_double();
        resolution_ = this->get_parameter("resolution").as_double();
        block_depth_ = this->get_parameter("block_depth").as_int();
        sf2_ = this->get_parameter("sf2").as_double();
        ell_ = this->get_parameter("ell").as_double();
        free_resolution_ = this->get_parameter("free_resolution").as_double();
        ds_resolution_ = this->get_parameter("ds_resolution").as_double();
        free_thresh_ = this->get_parameter("free_thresh").as_double();
        occupied_thresh_ = this->get_parameter("occupied_thresh").as_double();
        min_z_ = this->get_parameter("min_z").as_double();
        max_z_ = this->get_parameter("max_z").as_double();
        original_size_ = this->get_parameter("original_size").as_bool();
        noise_ = this->get_parameter("noise").as_double();
        l_ = this->get_parameter("l").as_double();
        min_var_ = this->get_parameter("min_var").as_double();
        max_var_ = this->get_parameter("max_var").as_double();
        max_known_var_ = this->get_parameter("max_known_var").as_double();

        // --- Initialization of Core Components ---
        
        // TF Buffer & Listener (Safe to do in constructor)
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // GPC Octomap (Safe to do in constructor)
        map_ = std::make_unique<gpcoctomap::GPCOctoMap>(resolution_, block_depth_, sf2_, ell_,
                                     noise_, l_, min_var_, max_var_, max_known_var_,
                                     free_thresh_, occupied_thresh_);

        // Subscriber
        point_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, 100, std::bind(&GpcoctomapHandler::cloudHandler, this, _1));
            
        RCLCPP_INFO(this->get_logger(), "Node constructed, waiting for initialization...");
    }

    /**
     * @brief Call this method immediately after creating the shared_ptr of this node.
     * This handles components that require 'shared_from_this()'.
     */
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing publishers...");
        
        // Marker Publishers (Now safe to call shared_from_this())
        m_pub_occ_ = std::make_unique<gpcoctomap::MarkerArrayPub>(this->shared_from_this(), map_topic_occ_, resolution_);
        m_pub_free_ = std::make_unique<gpcoctomap::MarkerArrayPub>(this->shared_from_this(), map_topic_free_, resolution_);
        
        RCLCPP_INFO(this->get_logger(), "Initialization complete.");
    }

    ~GpcoctomapHandler() {
        // Save Occupied Cloud
        if (occupied_cloud_ && !occupied_cloud_->empty()) {
            if (pcl::io::savePCDFileBinary("/home/ivana-rfal/gpcoctomap.pcd", *occupied_cloud_) == -1) {
                RCLCPP_ERROR(this->get_logger(), "Failed to save PCD file!");
            } else {
                RCLCPP_INFO_STREAM(this->get_logger(), "Saved occupancy map with " << occupied_cloud_->size()
                                << " points to /home/ivana-rfal/gpcoctomap.pcd");
            }
        }

        // Statistics
        if (!times_.empty()) {
             double sum = std::accumulate(times_.begin(), times_.end(), 0.0);
             double avg = sum / times_.size();
             RCLCPP_INFO_STREAM(this->get_logger(), "Final Timing Stats: Average = " << avg << "s");
        }
    }

private:
    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
        // Ensure publishers are initialized before processing
        if (!m_pub_occ_ || !m_pub_free_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "Publishers not initialized yet. Call node->initialize() in main.");
            return;
        }

        geometry_msgs::msg::TransformStamped transform_stamped;
        
        try {
            // FIX: Use tf2::durationFromSec(1.0) instead of 1.0s
            transform_stamped = tf_buffer_->lookupTransform(
                frame_id_, 
                cloud->header.frame_id, 
                tf2::TimePointZero, 
                tf2::durationFromSec(1.0)); 
        } catch (tf2::TransformException &ex) {
            // Lowered to WARN so it doesn't spam errors on startup
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Transform error: %s", ex.what());
            return;
        }

        rclcpp::Time start = this->get_clock()->now();
        gpcoctomap::point3f origin;

        // ... rest of the function remains the same ... 
        
        // Extract translation and rotation
        tf2::Vector3 translation(
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z
        );
        
        tf2::Quaternion orientation(
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z,
            transform_stamped.transform.rotation.w
        );

        if (first_ || orientation.angleShortestPath(last_orientation_) > orientation_change_thresh_ ||
            translation.distance(last_position_) > position_change_thresh_) 
        {
            last_position_ = translation;
            last_orientation_ = orientation;
            first_ = false;

            origin.x() = (float) translation.x();
            origin.y() = (float) translation.y();
            origin.z() = (float) translation.z();

            // Transform Cloud
            sensor_msgs::msg::PointCloud2 cloud_map_msg;
            tf2::doTransform(*cloud, cloud_map_msg, transform_stamped);

            // Convert to PCL
            gpcoctomap::PCLCPointCloud::Ptr pcl_cloud(new gpcoctomap::PCLCPointCloud());
            pcl::fromROSMsg(cloud_map_msg, *pcl_cloud);

            if (pcl_cloud->size() > 5) {
                map_->insert_pointcloud(*pcl_cloud, origin, (float) resolution_, 
                                        (float) free_resolution_, (float) max_range_);
            }
            updated_ = true;
        }

        if (updated_) {
            m_pub_occ_->clear();
            m_pub_free_->clear();
            occupied_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

            for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
                gpcoctomap::point3f p = it.get_loc();
                if (it.get_node().get_state() == gpcoctomap::State::OCCUPIED) {
                    if (original_size_) {
                        m_pub_occ_->insert_point3d(p.x(), p.y(), p.z(), min_z_, max_z_, it.get_size());
                        occupied_cloud_->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
                    } else {
                        auto pruned = it.get_pruned_locs();
                        for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                            m_pub_occ_->insert_point3d(n->x(), n->y(), n->z(), min_z_, max_z_, map_->get_resolution());
                    }
                }
            }

            m_pub_occ_->publish();
            m_pub_free_->publish();
            updated_ = false;

            rclcpp::Time end2 = this->get_clock()->now();
            double duration = (end2 - start).seconds();
            times_.push_back(duration);
        }
    }

    // --- Members ---
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_sub_;
    
    std::string frame_id_;
    std::unique_ptr<gpcoctomap::GPCOctoMap> map_;
    
    std::unique_ptr<gpcoctomap::MarkerArrayPub> m_pub_occ_;
    std::unique_ptr<gpcoctomap::MarkerArrayPub> m_pub_free_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr occupied_cloud_;

    tf2::Vector3 last_position_;
    tf2::Quaternion last_orientation_;
    bool first_, updated_;
    double position_change_thresh_, orientation_change_thresh_;

    std::string map_topic_occ_, map_topic_free_;
    double max_range_, resolution_, sf2_, ell_;
    double free_resolution_, ds_resolution_;
    double free_thresh_, occupied_thresh_;
    double min_z_, max_z_;
    bool original_size_;
    double noise_, l_, min_var_, max_var_, max_known_var_;
    int block_depth_;

    std::vector<double> times_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    std::string cloud_topic = "/sonar_camera_merge/cloud";
    
    // 1. Create the shared pointer
    auto node = std::make_shared<GpcoctomapHandler>(cloud_topic);
    
    // 2. IMPORTANT: Call initialize() explicitly after creation
    // This allows shared_from_this() to work inside this method
    node->initialize();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}