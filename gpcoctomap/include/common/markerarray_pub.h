#include <rclcpp/rclcpp.hpp>

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ROS 2 Message includes (note the ::msg namespace)
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <cmath>
#include <string>
#include <memory>
#include <algorithm> // for std::min, std::max
#include "state.h"

namespace gpcoctomap {

    std_msgs::msg::ColorRGBA heightMapColor(double h) {

        std_msgs::msg::ColorRGBA color;
        color.a = 1.0;
        // blend over HSV-values (more colors)

        double s = 1.0;
        double v = 1.0;

        h -= floor(h);
        h *= 6;
        int i;
        double m, n, f;

        i = floor(h);
        f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        m = v * (1 - s);
        n = v * (1 - s * f);

        switch (i) {
            case 6:
            case 0:
                color.r = v;
                color.g = n;
                color.b = m;
                break;
            case 1:
                color.r = n;
                color.g = v;
                color.b = m;
                break;
            case 2:
                color.r = m;
                color.g = v;
                color.b = n;
                break;
            case 3:
                color.r = m;
                color.g = n;
                color.b = v;
                break;
            case 4:
                color.r = n;
                color.g = m;
                color.b = v;
                break;
            case 5:
                color.r = v;
                color.g = m;
                color.b = n;
                break;
            default:
                color.r = 1;
                color.g = 0.5;
                color.b = 0.5;
                break;
        }

        return color;
    }

    class MarkerArrayPub {
        typedef pcl::PointXYZ PointType;
        typedef pcl::PointCloud<PointType> PointCloud;
    public:
        // Changed NodeHandle to rclcpp::Node::SharedPtr
        MarkerArrayPub(rclcpp::Node::SharedPtr node, std::string topic, float resolution) 
            : node_(node),
              msg_(std::make_shared<visualization_msgs::msg::MarkerArray>()),
              topic_(topic),
              resolution_(resolution),
              markerarray_frame_id_("map") {
            
            // Handle Latching (Transient Local durability)
            rclcpp::QoS qos_profile(1);
            qos_profile.transient_local();
            qos_profile.reliable();

            pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic, qos_profile);

            msg_->markers.resize(10);
            for (int i = 0; i < 10; ++i) {
                msg_->markers[i].header.frame_id = markerarray_frame_id_;
                msg_->markers[i].ns = "map";
                msg_->markers[i].id = i;
                msg_->markers[i].type = visualization_msgs::msg::Marker::CUBE_LIST;
                msg_->markers[i].scale.x = resolution * pow(2, i);
                msg_->markers[i].scale.y = resolution * pow(2, i);
                msg_->markers[i].scale.z = resolution * pow(2, i);
                
                std_msgs::msg::ColorRGBA color;
                color.r = 0.0;
                color.g = 0.0;
                color.b = 1.0;
                color.a = 1.0;
                msg_->markers[i].color = color;
            }
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, float size) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / resolution_);

            if (depth < msg_->markers.size()) {
                 msg_->markers[depth].points.push_back(center);

                if (min_z < max_z) {
                    double h = (1.0 - std::min(std::max((z - min_z) / (max_z - min_z), 0.0f), 1.0f)) * 0.8;
                    msg_->markers[depth].colors.push_back(heightMapColor(h));
                }
            }
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, float size, float prob) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / resolution_);

             if (depth < msg_->markers.size()) {
                msg_->markers[depth].points.push_back(center);

                std_msgs::msg::ColorRGBA color;
                color.a = 1.0;

                if(prob < 0.5){
                    color.r = 0.8; 
                    color.g = 0.8; 
                    color.b = 0.8; 
                }
                else{
                    color = heightMapColor(std::min(2.0-2.0*prob, 0.6));
                }

                msg_->markers[depth].colors.push_back(color);
            }
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z) {
            insert_point3d(x, y, z, min_z, max_z, -1.0f);
        }

        void insert_point3d(float x, float y, float z) {
            insert_point3d(x, y, z, 1.0f, 0.0f, -1.0f);
        }

        void insert_color_point3d(float x, float y, float z, double min_v, double max_v, double v) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            msg_->markers[depth].points.push_back(center);

            double h = (1.0 - std::min(std::max((v - min_v) / (max_v - min_v), 0.0), 1.0)) * 0.8;
            msg_->markers[depth].colors.push_back(heightMapColor(h));
        }

        void insert_state_point3d(float x, float y, float z, State state) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            msg_->markers[depth].points.push_back(center);
            
            std_msgs::msg::ColorRGBA color;
            color.a = 1.0;  // fully opaque

            switch (state) {
                case State::OCCUPIED: // Red
                    color.r = 1.0;
                    color.g = 0.0;
                    color.b = 0.0;
                    break;
                case State::FREE: // Green
                    color.r = 0.0;
                    color.g = 1.0;
                    color.b = 0.0;
                    break;
                case State::UNKNOWN:  // Magenta
                    color.a = 0.01;                   
                    color.r = 1.0;
                    color.g = 0.0;
                    color.b = 1.0;
                    break;
                case State::PRUNED: // Yellow                     
                    color.r = 1.0;
                    color.g = 1.0;
                    color.b = 0.0;
                    break;
                default: // State::UNCERTAIN: // Blue
                    color.r = 0.0;
                    color.g = 0.0;
                    color.b = 1.0;
                    break;
            }
            msg_->markers[depth].colors.push_back(color);
        }

        void clear() {
            for (size_t i = 0; i < msg_->markers.size(); ++i) {
                msg_->markers[i].points.clear();
                msg_->markers[i].colors.clear();
            }
        }

        void publish() {
            // Need to set the timestamp for the first marker (or all)
            rclcpp::Time now = node_->get_clock()->now();
            if (!msg_->markers.empty()) {
                msg_->markers[0].header.stamp = now;
            }
            pub_->publish(*msg_);
        }

    private:
        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
        std::shared_ptr<visualization_msgs::msg::MarkerArray> msg_;
        std::string markerarray_frame_id_;
        std::string topic_;
        float resolution_;
    };

}