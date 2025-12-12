#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h> 

// Standard C++
#include <cmath>
#include <vector>
#include <map>

// OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

// RTAB-Map Core Headers
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Transform.h>

class VinsLoopDetector {
public:
    VinsLoopDetector() : frame_count_(0), has_received_odom_(false), first_frame_(true) {
        ros::NodeHandle nh("~");

        // 使用 OpenCV ORB 特征 (CPU)
        // 参数: nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31...
        orb_detector_ = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

        // RTAB-Map 参数配置
        rtabmap::ParametersMap parameters;
        
        // 核心回环阈值
        parameters.insert(rtabmap::ParametersPair("Rtabmap/LoopThr", "0.15")); 
        
        // ORB 特征匹配内点阈值
        parameters.insert(rtabmap::ParametersPair("Vis/MinInliers", "25"));
        
        // 内存管理：启用 Rehearsal 防止静止时数据堆积
        parameters.insert(rtabmap::ParametersPair("Mem/RehearsalSimilarity", "0.3")); 
        
        // 禁用自动全局优化，改为按需手动触发
        parameters.insert(rtabmap::ParametersPair("RGBD/OptimizeFromGraphEnd", "false")); 
        parameters.insert(rtabmap::ParametersPair("RGBD/LinearUpdate", "0")); 
        parameters.insert(rtabmap::ParametersPair("RGBD/AngularUpdate", "0"));

        // 纯内存运行，不保存数据库
        parameters.insert(rtabmap::ParametersPair("Db/Sqlite3Output", "")); 

        rtabmap_.init(parameters);

        // 订阅与发布
        // 注意：根据实际情况 Remap 话题
        sub_image_ = nh.subscribe("/camera/infra1/image_rect_raw", 10, &VinsLoopDetector::imageCallback, this);
        sub_odom_  = nh.subscribe("/vins_estimator/odometry", 100, &VinsLoopDetector::odomCallback, this);
        sub_info_  = nh.subscribe("/camera/infra1/camera_info", 10, &VinsLoopDetector::infoCallback, this);

        pub_path_ = nh.advertise<nav_msgs::Path>("/loop_fusion/pose_graph_path", 10);
        pub_match_img_ = nh.advertise<sensor_msgs::Image>("/loop_fusion/match_image", 1, true); 
        pub_marker_array_ = nh.advertise<visualization_msgs::MarkerArray>("/loop_fusion/pose_graph", 10);

        ROS_INFO(">>> VINS LOOP DETECTOR (ORB Version) STARTED <<<");
        ROS_INFO("Waiting for VINS to initialize...");
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
        if (!last_odom_) {
            ROS_WARN_THROTTLE(5.0, "Waiting for VINS Odometry...");
            return;
        }
        if (!camera_model_.isValidForProjection()) {
            ROS_WARN_THROTTLE(5.0, "Waiting for Camera Info...");
            return;
        }

        // 转换 VINS 位姿
        rtabmap::Transform current_pose(
            last_odom_->pose.pose.position.x, last_odom_->pose.pose.position.y, last_odom_->pose.pose.position.z,
            last_odom_->pose.pose.orientation.x, last_odom_->pose.pose.orientation.y, 
            last_odom_->pose.pose.orientation.z, last_odom_->pose.pose.orientation.w
        );

        // 运动关键帧策略
        bool is_keyframe = false;
        if (first_frame_) {
            is_keyframe = true;
            first_frame_ = false;
        } else {
            // 手动计算距离 (兼容旧版本 API)
            float dx = current_pose.x() - last_keyframe_pose_.x();
            float dy = current_pose.y() - last_keyframe_pose_.y();
            float dz = current_pose.z() - last_keyframe_pose_.z();
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            // 手动计算角度
            rtabmap::Transform diff = last_keyframe_pose_.inverse() * current_pose;
            float angle = diff.getAngle(); 

            // 阈值：移动 0.1m 或 旋转 0.2rad
            if (dist > 0.1f || angle > 0.2f) {
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

        // 提取 ORB 特征
        std::vector<cv::KeyPoint> kps;
        cv::Mat descriptors;
        orb_detector_->detectAndCompute(cv_ptr->image, cv::noArray(), kps, descriptors);

        // 特征太少跳过
        if (kps.size() < 50) return;

        rtabmap::Transform odom_pose(
            odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z,
            odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, 
            odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w
        );

        int id = ++frame_count_; 
        rtabmap::SensorData data(cv_ptr->image, id, odom_msg->header.stamp.toSec());
        data.setCameraModel(camera_model_); 
        
        std::vector<cv::Point3f> kps3D; 
        data.setFeatures(kps, kps3D, descriptors);

        bool success = rtabmap_.process(data, odom_pose);
        int loop_id = rtabmap_.getLoopClosureId();
        
        if (id % 10 == 0 || loop_id > 0) {
            ROS_INFO("Frame %d | Feat: %d | Loop: %d", id, (int)kps.size(), loop_id);
        }

        // 只有回环或者每50帧才优化一次，节省CPU
        if (success && (loop_id > 0 || id % 50 == 0)) {
            std::map<int, rtabmap::Transform> poses;
            std::multimap<int, rtabmap::Link> links;
            
            rtabmap_.getGraph(poses, links, true, true);
            
            publishOptimizedPath(poses, odom_msg->header);

            if (loop_id > 0) {
                ROS_WARN(">>> LOOP DETECTED: %d -> %d", id, loop_id);
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
            pose.header = header; 
            pose.header.frame_id = "world";
            
            rtabmap::Transform t = it->second;
            pose.pose.position.x = t.x(); 
            pose.pose.position.y = t.y(); 
            pose.pose.position.z = t.z();
            
            Eigen::Quaternionf q = t.getQuaternionf();
            pose.pose.orientation.x = q.x(); 
            pose.pose.orientation.y = q.y(); 
            pose.pose.orientation.z = q.z(); 
            pose.pose.orientation.w = q.w();
            
            path.poses.push_back(pose);
        }
        pub_path_.publish(path);
    }

    void publishLoopConstraints(const std::map<int, rtabmap::Transform>& poses, const std::multimap<int, rtabmap::Link>& links, const std_msgs::Header& header) {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker_edge;
        marker_edge.header.frame_id = "world"; 
        marker_edge.header.stamp = header.stamp;
        marker_edge.action = visualization_msgs::Marker::ADD; 
        marker_edge.type = visualization_msgs::Marker::LINE_LIST;
        marker_edge.ns = "loop_edges"; 
        marker_edge.id = 0; 
        marker_edge.pose.orientation.w = 1;
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
                    
                    marker_edge.points.push_back(gp1); 
                    marker_edge.points.push_back(gp2);
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
    cv::Ptr<cv::ORB> orb_detector_;
    
    int frame_count_;
    bool has_received_odom_;
    bool first_frame_; 
    rtabmap::Transform last_keyframe_pose_; 
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "vins_loop"); 
    VinsLoopDetector detector;
    ros::spin();
    return 0;
}