#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h> 

#include <cmath>
#include <memory> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

// RTAB-Map
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/utilite/ULogger.h>

#include "SuperPoint.h"

class VinsLoopDetector {
public:
    VinsLoopDetector() : frame_count_(0), has_received_odom_(false), first_frame_(true) {
        ros::NodeHandle nh("~");

        ULogger::setType(ULogger::kTypeConsole);
        ULogger::setLevel(ULogger::kWarning); 

        std::string model_path = "/home/ark/weights/superpoint.engine"; 
        FILE* f = fopen(model_path.c_str(), "r");
        if (f == NULL) {
            ROS_ERROR("Engine file not found at %s", model_path.c_str());
            ros::shutdown();
            return;
        }
        fclose(f);

        sp_detector_ = std::make_shared<SuperPoint>(model_path);
        ROS_INFO(">>> SuperPoint Detector Initialized (Using CUDA) <<<");

        // --- 参数配置 ---
        rtabmap::ParametersMap parameters;
        
        // 匹配策略
        parameters.insert(rtabmap::ParametersPair("Kp/NNStrategy", "3")); // BruteForce
        parameters.insert(rtabmap::ParametersPair("Kp/NndrRatio", "0.8")); 
        parameters.insert(rtabmap::ParametersPair("Kp/DictionaryPath", "")); 

        // 关键：保存原始描述子
        parameters.insert(rtabmap::ParametersPair("Mem/RawDescriptorsKept", "true"));
        parameters.insert(rtabmap::ParametersPair("Mem/BinDataKept", "true"));
        
        // 回环检测
        parameters.insert(rtabmap::ParametersPair("Rtabmap/LoopThr", "0.10")); 
        parameters.insert(rtabmap::ParametersPair("Vis/MinInliers", "15"));
        
        // 内存管理
        parameters.insert(rtabmap::ParametersPair("Mem/STMSize", "30")); 
        parameters.insert(rtabmap::ParametersPair("Mem/RehearsalSimilarity", "0.3")); // 恢复正常防止内存爆炸
        parameters.insert(rtabmap::ParametersPair("Mem/NotLinkedNodesKept", "true")); 

        // 禁用内部过滤
        parameters.insert(rtabmap::ParametersPair("RGBD/AngularUpdate", "0"));
        parameters.insert(rtabmap::ParametersPair("RGBD/LinearUpdate", "0"));
        parameters.insert(rtabmap::ParametersPair("Kp/DetectorStrategy", "0")); 
        parameters.insert(rtabmap::ParametersPair("Kp/MaxFeatures", "1000"));

        parameters.insert(rtabmap::ParametersPair("RGBD/OptimizeFromGraphEnd", "false")); 
        parameters.insert(rtabmap::ParametersPair("Db/Sqlite3Output", "")); 

        rtabmap_.init(parameters);

        sub_image_ = nh.subscribe("/camera/infra1/image_rect_raw", 10, &VinsLoopDetector::imageCallback, this);
        sub_odom_  = nh.subscribe("/vins_estimator/odometry", 100, &VinsLoopDetector::odomCallback, this);
        sub_info_  = nh.subscribe("/camera/infra1/camera_info", 10, &VinsLoopDetector::infoCallback, this);

        pub_path_ = nh.advertise<nav_msgs::Path>("/loop_fusion/pose_graph_path", 10);
        pub_match_img_ = nh.advertise<sensor_msgs::Image>("/loop_fusion/match_image", 1, true); 
        pub_marker_array_ = nh.advertise<visualization_msgs::MarkerArray>("/loop_fusion/pose_graph", 10);

        ROS_INFO(">>> VINS LOOP DETECTOR (Fixed Compilation) STARTED <<<");
    }

    void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
        last_odom_ = msg;
        if (!has_received_odom_) {
            has_received_odom_ = true;
            ROS_INFO(">>> VINS ODOM RECEIVED! System Running. <<<");
        }
    }

    void infoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
        if (camera_model_.isValidForProjection()) return;
        cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
        K.at<double>(0,0) = msg->K[0]; K.at<double>(1,1) = msg->K[4];
        K.at<double>(0,2) = msg->K[2]; K.at<double>(1,2) = msg->K[5];
        cv::Mat D = cv::Mat(1, msg->D.size(), CV_64FC1);
        for(size_t i=0; i<msg->D.size(); ++i) D.at<double>(i) = msg->D[i];
        camera_model_ = rtabmap::CameraModel("pinhole", cv::Size(msg->width, msg->height), K, D, cv::Mat(), cv::Mat(), rtabmap::CameraModel::opticalRotation());
        ROS_INFO(">>> Camera Intrinsics Loaded: %dx%d <<<", msg->width, msg->height);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
        if (!last_odom_ || !camera_model_.isValidForProjection()) return;

        rtabmap::Transform current_pose(
            last_odom_->pose.pose.position.x, last_odom_->pose.pose.position.y, last_odom_->pose.pose.position.z,
            last_odom_->pose.pose.orientation.x, last_odom_->pose.pose.orientation.y, 
            last_odom_->pose.pose.orientation.z, last_odom_->pose.pose.orientation.w
        );

        bool is_keyframe = false;
        if (first_frame_) {
            is_keyframe = true;
            first_frame_ = false;
        } else {
            // 计算位移
            float dist = current_pose.getDistance(last_keyframe_pose_);
            
            // 【关键修复】计算旋转角度的正确方法
            // 先计算两个位姿的相对变换(diff)，再获取这个变换的角度
            rtabmap::Transform diff = last_keyframe_pose_.inverse() * current_pose;
            float angle = diff.getAngle(); 

            // 阈值：移动 8cm 或 旋转 > 0.1 rad (约6度)
            if (dist > 0.08f || angle > 0.1f) { 
                is_keyframe = true;
            }
        }

        if (is_keyframe) {
            last_keyframe_pose_ = current_pose;
            process(img_msg, last_odom_);
        }
    }

    void process(const sensor_msgs::ImageConstPtr& img_msg, const nav_msgs::OdometryConstPtr& odom_msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(img_msg, "mono8"); } 
        catch (cv_bridge::Exception& e) { return; }

        if (cv_ptr->image.empty()) return;

        std::vector<cv::KeyPoint> kps;
        cv::Mat descriptors;
        
        sp_detector_->extractFeatures(cv_ptr->image, kps, descriptors);

        if (kps.size() < 10) return; 

        rtabmap::Transform odom_pose(
            odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z,
            odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, 
            odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w
        );

        int id = ++frame_count_; 
        // ID设为0，自动管理
        rtabmap::SensorData data(cv_ptr->image, 0, odom_msg->header.stamp.toSec());
        data.setCameraModel(camera_model_); 
        
        if (descriptors.type() != CV_32F) {
            descriptors.convertTo(descriptors, CV_32F);
        }
        
        std::vector<cv::Point3f> kps3D; 
        data.setFeatures(kps, kps3D, descriptors);

        cv::Mat covariance = cv::Mat::eye(6, 6, CV_64FC1) * 0.01;
        bool success = rtabmap_.process(data, odom_pose, covariance);
        int loop_id = rtabmap_.getLoopClosureId();
        
        if (++frame_count_ % 20 == 0) {
            const std::map<std::string, float>& stats = rtabmap_.getStatistics().data();
            int wm_size = 0;
            if (stats.find("Mem/Working_memory_size") != stats.end())
                wm_size = (int)stats.at("Mem/Working_memory_size");
            
            ROS_INFO("Processed %d Keyframes | WM Size: %d | Last Feat: %d", frame_count_, wm_size, (int)kps.size());
        }

        if (success) {
            std::map<int, rtabmap::Transform> poses;
            std::multimap<int, rtabmap::Link> links;
            rtabmap_.getGraph(poses, links, true, true);
            publishOptimizedPath(poses, odom_msg->header);

            if (loop_id > 0) {
                ROS_WARN(">>> SUPERPOINT LOOP DETECTED! Current -> Old ID: %d <<<", loop_id);
                publishLoopConstraints(poses, links, odom_msg->header);
                
                sensor_msgs::Image img_copy = *img_msg;
                img_copy.header.frame_id = "world"; 
                pub_match_img_.publish(img_copy); 
            }
        }
    }

    void publishOptimizedPath(const std::map<int, rtabmap::Transform>& poses, const std_msgs::Header& header) {
        if (poses.empty()) return;
        nav_msgs::Path path;
        path.header.frame_id = "world"; path.header.stamp = header.stamp;
        for (auto it = poses.begin(); it != poses.end(); ++it) {
            geometry_msgs::PoseStamped pose;
            pose.header = header; pose.header.frame_id = "world";
            rtabmap::Transform t = it->second;
            pose.pose.position.x = t.x(); pose.pose.position.y = t.y(); pose.pose.position.z = t.z();
            Eigen::Quaternionf q = t.getQuaternionf();
            pose.pose.orientation.x = q.x(); pose.pose.orientation.y = q.y(); pose.pose.orientation.z = q.z(); pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
        }
        pub_path_.publish(path);
    }

    void publishLoopConstraints(const std::map<int, rtabmap::Transform>& poses, 
                                const std::multimap<int, rtabmap::Link>& links, 
                                const std_msgs::Header& header) {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker_edge;
        marker_edge.header.frame_id = "world"; marker_edge.header.stamp = header.stamp;
        marker_edge.action = visualization_msgs::Marker::ADD; marker_edge.type = visualization_msgs::Marker::LINE_LIST;
        marker_edge.ns = "loop_edges"; marker_edge.id = 0; marker_edge.pose.orientation.w = 1;
        marker_edge.scale.x = 0.05; 
        marker_edge.color.r = 0.0; marker_edge.color.g = 1.0; marker_edge.color.b = 0.0; marker_edge.color.a = 1.0;

        for (auto it = links.begin(); it != links.end(); ++it) {
            if (it->second.type() != rtabmap::Link::kNeighbor) {
                if (poses.find(it->first) != poses.end() && poses.find(it->second.to()) != poses.end()) {
                    rtabmap::Transform p1 = poses.at(it->first);
                    rtabmap::Transform p2 = poses.at(it->second.to());
                    geometry_msgs::Point gp1, gp2;
                    gp1.x = p1.x(); gp1.y = p1.y(); gp1.z = p1.z();
                    gp2.x = p2.x(); gp2.y = p2.y(); gp2.z = p2.z();
                    marker_edge.points.push_back(gp1); marker_edge.points.push_back(gp2);
                }
            }
        }
        if(!marker_edge.points.empty()) {
            marker_array.markers.push_back(marker_edge);
            pub_marker_array_.publish(marker_array);
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_image_, sub_odom_, sub_info_;
    ros::Publisher pub_path_, pub_match_img_, pub_marker_array_;
    nav_msgs::OdometryConstPtr last_odom_;
    rtabmap::Rtabmap rtabmap_;
    rtabmap::CameraModel camera_model_;
    std::shared_ptr<SuperPoint> sp_detector_;
    int frame_count_;
    bool has_received_odom_;
    bool first_frame_;
    rtabmap::Transform last_keyframe_pose_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_vins_loop"); 
    VinsLoopDetector detector;
    ros::spin();
    return 0;
}