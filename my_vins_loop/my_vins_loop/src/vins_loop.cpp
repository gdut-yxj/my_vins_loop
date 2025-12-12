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
    #include <vector>
    #include <map>

    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/features2d.hpp>

    // RTAB-Map
    #include <rtabmap/core/Rtabmap.h>
    #include <rtabmap/core/SensorData.h>
    #include <rtabmap/core/Parameters.h>
    #include <rtabmap/core/CameraModel.h>
    #include <rtabmap/core/StereoCameraModel.h>
    #include <rtabmap/core/Optimizer.h>
    #include <rtabmap/core/Transform.h>
    #include <rtabmap/utilite/ULogger.h>

    class VinsLoopDetector {
    public:
        VinsLoopDetector() : frame_count_(0), has_received_odom_(false), has_camera_info_(false), first_frame_(true) {
            ros::NodeHandle nh("~");

            ULogger::setType(ULogger::kTypeConsole);
            ULogger::setLevel(ULogger::kWarning);

            // --- RTAB-Map 参数配置 ---
            rtabmap::ParametersMap parameters;
            
            // 1. 【核心修复】告诉 RTAB-Map 输入图像已经矫正过了
            parameters.insert(rtabmap::ParametersPair("Rtabmap/ImagesAlreadyRectified", "true"));

            // 2. 启用双目模式
            parameters.insert(rtabmap::ParametersPair("Stereo/Enabled", "true")); 
            parameters.insert(rtabmap::ParametersPair("Stereo/MaxDisparity", "128.0")); 

            // 3. 特征提取 (Auto ORB)
            parameters.insert(rtabmap::ParametersPair("Kp/DetectorStrategy", "2")); 
            parameters.insert(rtabmap::ParametersPair("Kp/MaxFeatures", "1000"));

            // 4. 匹配策略
            parameters.insert(rtabmap::ParametersPair("Kp/NNStrategy", "3")); 
            parameters.insert(rtabmap::ParametersPair("Kp/NndrRatio", "0.8")); 
            parameters.insert(rtabmap::ParametersPair("Kp/DictionaryPath", "")); 

            // 5. 数据保存
            parameters.insert(rtabmap::ParametersPair("Mem/BinDataKept", "true"));
            parameters.insert(rtabmap::ParametersPair("Mem/RawDescriptorsKept", "true"));
            parameters.insert(rtabmap::ParametersPair("Mem/UseOdomFeatures", "false"));

            // 6. 回环判定
            parameters.insert(rtabmap::ParametersPair("Rtabmap/LoopThr", "0.15")); 
            parameters.insert(rtabmap::ParametersPair("Vis/MinInliers", "30"));
            parameters.insert(rtabmap::ParametersPair("Vis/EstimationType", "0")); // 3D-3D

            // 7. 内存管理
            parameters.insert(rtabmap::ParametersPair("Mem/STMSize", "30")); 
            parameters.insert(rtabmap::ParametersPair("Mem/NotLinkedNodesKept", "true")); 
            
            // 8. 禁用内部过滤
            parameters.insert(rtabmap::ParametersPair("RGBD/AngularUpdate", "0"));
            parameters.insert(rtabmap::ParametersPair("RGBD/LinearUpdate", "0"));
            parameters.insert(rtabmap::ParametersPair("RGBD/OptimizeFromGraphEnd", "false")); 

            // 9. 纯内存运行
            parameters.insert(rtabmap::ParametersPair("Db/Sqlite3Output", "")); 

            rtabmap_.init(parameters);

            // 订阅话题
            sub_left_  = nh.subscribe("/camera/infra1/image_rect_raw", 10, &VinsLoopDetector::leftCallback, this);
            sub_right_ = nh.subscribe("/camera/infra2/image_rect_raw", 10, &VinsLoopDetector::rightCallback, this);
            sub_odom_  = nh.subscribe("/vins_fusion/odometry", 100, &VinsLoopDetector::odomCallback, this);
            sub_info_  = nh.subscribe("/camera/infra1/camera_info", 10, &VinsLoopDetector::infoCallback, this);

            // 发布话题
            pub_path_ = nh.advertise<nav_msgs::Path>("/loop_fusion/pose_graph_path", 10);
            pub_match_img_ = nh.advertise<sensor_msgs::Image>("/loop_fusion/match_image", 1, true); 
            pub_marker_array_ = nh.advertise<visualization_msgs::MarkerArray>("/loop_fusion/pose_graph", 10);

            ROS_INFO(">>> VINS LOOP DETECTOR (Stereo Fixed) STARTED <<<");
        }

        void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
            last_odom_ = msg;
            if (!has_received_odom_) {
                has_received_odom_ = true;
                ROS_INFO(">>> VINS ODOM RECEIVED! System Running. <<<");
            }
        }

        void infoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
            if (has_camera_info_) return;

            if (msg->width == 0 || msg->height == 0) return;

            double fx = msg->P[0];
            double fy = msg->P[5];
            double cx = msg->P[2];
            double cy = msg->P[6];
            double baseline = 0.05; 

            cv::Size size(msg->width, msg->height);
            
            // 【关键修正】参数顺序：name, fx, fy, cx, cy, transform, tx, size
            rtabmap::CameraModel leftModel(
                "pinhole", 
                fx, fy, cx, cy, 
                rtabmap::Transform::getIdentity(), 
                0.0, 
                size
            );

            stereo_model_ = rtabmap::StereoCameraModel(
                leftModel.fx(), leftModel.fy(), leftModel.cx(), leftModel.cy(), 
                baseline, rtabmap::CameraModel::opticalRotation(), size
            );

            has_camera_info_ = true;
            ROS_INFO(">>> Stereo Model Created: %dx%d, Baseline: %.3f <<<", msg->width, msg->height, baseline);
        }

        void leftCallback(const sensor_msgs::ImageConstPtr& msg) {
            img_left_buf_ = msg;
        }

        void rightCallback(const sensor_msgs::ImageConstPtr& msg) {
            if (!img_left_buf_ || !last_odom_ || !has_camera_info_) return;

            double dt = std::abs((img_left_buf_->header.stamp - msg->header.stamp).toSec());
            if (dt > 0.01) return; 

            if ((ros::Time::now() - msg->header.stamp).toSec() > 0.1) return;

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
                // 手写距离计算
                float dx = current_pose.x() - last_keyframe_pose_.x();
                float dy = current_pose.y() - last_keyframe_pose_.y();
                float dz = current_pose.z() - last_keyframe_pose_.z();
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                
                if (dist > 0.08f) is_keyframe = true;
            }

            if (is_keyframe) {
                last_keyframe_pose_ = current_pose;
                process(img_left_buf_, msg, last_odom_);
            }
            
            img_left_buf_ = nullptr;
        }

        void process(const sensor_msgs::ImageConstPtr& left_msg, 
                    const sensor_msgs::ImageConstPtr& right_msg,
                    const nav_msgs::OdometryConstPtr& odom_msg) {
            
            cv_bridge::CvImageConstPtr ptr_l, ptr_r;
            try {
                ptr_l = cv_bridge::toCvCopy(left_msg, "mono8");
                ptr_r = cv_bridge::toCvCopy(right_msg, "mono8");
            } catch (cv_bridge::Exception& e) { return; }

            if (ptr_l->image.empty() || ptr_r->image.empty()) return;

            int model_w = stereo_model_.left().imageWidth();
            int model_h = stereo_model_.left().imageHeight();
            
            if (ptr_l->image.cols != model_w || ptr_l->image.rows != model_h) {
                return;
            }

            int id = ++frame_count_;

            rtabmap::Transform odom_pose(
                odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z,
                odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, 
                odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w
            );

            rtabmap::SensorData data(ptr_l->image, ptr_r->image, stereo_model_, 0, odom_msg->header.stamp.toSec());
            
            cv::Mat covariance = cv::Mat::eye(6, 6, CV_64FC1) * 0.01;
            
            bool success = rtabmap_.process(data, odom_pose, covariance);
            int loop_id = rtabmap_.getLoopClosureId();
            
            if (id % 10 == 0 || loop_id > 0) {
                const std::map<std::string, float>& stats = rtabmap_.getStatistics().data();
                int wm_size = 0;
                if (stats.find("Mem/Working_memory_size") != stats.end())
                    wm_size = (int)stats.at("Mem/Working_memory_size");
                
                float highest_hyp = 0.0f;
                if (stats.find("Loop/Highest_hypothesis_value") != stats.end())
                    highest_hyp = stats.at("Loop/Highest_hypothesis_value");

                ROS_INFO("Frame %d | WM: %d | Hyp: %.4f | Loop: %d | Accept: %s", 
                        id, wm_size, highest_hyp, loop_id, success?"YES":"NO");
            }

            if (success && loop_id > 0) {
                ROS_WARN(">>> STEREO LOOP DETECTED: %d -> %d <<<", id, loop_id);
                
                std::map<int, rtabmap::Transform> poses;
                std::multimap<int, rtabmap::Link> links;
                rtabmap_.getGraph(poses, links, true, true);
                
                publishOptimizedPath(poses, odom_msg->header);
                publishLoopConstraints(poses, links, odom_msg->header);
                
                sensor_msgs::Image img_copy = *left_msg;
                img_copy.header.frame_id = "world"; 
                pub_match_img_.publish(img_copy); 
            } else if (id % 50 == 0) {
                std::map<int, rtabmap::Transform> poses;
                std::multimap<int, rtabmap::Link> links;
                rtabmap_.getGraph(poses, links, true, true);
                publishOptimizedPath(poses, odom_msg->header);
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

        void publishLoopConstraints(const std::map<int, rtabmap::Transform>& poses, const std::multimap<int, rtabmap::Link>& links, const std_msgs::Header& header) {
            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker marker_edge;
            marker_edge.header.frame_id = "world"; marker_edge.header.stamp = header.stamp;
            marker_edge.action = visualization_msgs::Marker::ADD; marker_edge.type = visualization_msgs::Marker::LINE_LIST;
            marker_edge.ns = "loop_edges"; marker_edge.id = 0; marker_edge.pose.orientation.w = 1;
            marker_edge.scale.x = 0.05; 
            marker_edge.color.r = 1.0; marker_edge.color.g = 0.0; marker_edge.color.b = 0.0; marker_edge.color.a = 1.0;

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
        ros::Subscriber sub_left_, sub_right_, sub_odom_, sub_info_;
        ros::Publisher pub_path_, pub_match_img_, pub_marker_array_;
        
        sensor_msgs::ImageConstPtr img_left_buf_;
        nav_msgs::OdometryConstPtr last_odom_;
        
        rtabmap::Rtabmap rtabmap_;
        rtabmap::StereoCameraModel stereo_model_; 
        
        int frame_count_;
        bool has_camera_info_;
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