#include "SuperPoint.h"
#include <fstream>
#include <cmath>
#include <algorithm> 

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

SuperPoint::SuperPoint(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error reading engine file: " << engine_path << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    context_ = engine_->createExecutionContext();
    delete[] trtModelStream;

    // 自动获取绑定索引
    int idx_input = engine_->getBindingIndex("input");
    int idx_semi = engine_->getBindingIndex("semi");
    int idx_desc = engine_->getBindingIndex("desc");

    std::cout << ">>> TensorRT Ready. Ports: In=" << idx_input 
              << " Semi=" << idx_semi << " Desc=" << idx_desc << " <<<" << std::endl;

    input_size_ = 1 * input_h_ * input_w_ * sizeof(float);
    semi_size_  = 1 * 65 * (input_h_/8) * (input_w_/8) * sizeof(float);
    desc_size_  = 1 * 256 * (input_h_/8) * (input_w_/8) * sizeof(float);

    void* gpu_input_ptr = nullptr;
    void* gpu_semi_ptr = nullptr;
    void* gpu_desc_ptr = nullptr;

    CHECK(cudaMalloc(&gpu_input_ptr, input_size_));
    CHECK(cudaMalloc(&gpu_semi_ptr, semi_size_));
    CHECK(cudaMalloc(&gpu_desc_ptr, desc_size_));

    buffers_[idx_input] = gpu_input_ptr;
    buffers_[idx_semi]  = gpu_semi_ptr;
    buffers_[idx_desc]  = gpu_desc_ptr;
    
    semi_host_ = new float[semi_size_ / sizeof(float)];
    desc_host_ = new float[desc_size_ / sizeof(float)];

    CHECK(cudaStreamCreate(&stream_));
}

SuperPoint::~SuperPoint() {
    int idx_input = engine_->getBindingIndex("input");
    int idx_semi = engine_->getBindingIndex("semi");
    int idx_desc = engine_->getBindingIndex("desc");

    cudaStreamDestroy(stream_);
    CHECK(cudaFree(buffers_[idx_input]));
    CHECK(cudaFree(buffers_[idx_semi]));
    CHECK(cudaFree(buffers_[idx_desc]));
    
    delete[] semi_host_;
    delete[] desc_host_;
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
}

void SuperPoint::extractFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (img.empty()) return;

    // 1. 预处理
    cv::Mat gray, img_float;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img.clone();
    
    if (gray.rows != input_h_ || gray.cols != input_w_) {
        cv::resize(gray, gray, cv::Size(input_w_, input_h_));
    }
    gray.convertTo(img_float, CV_32F, 1.0/255.0);

    int idx_input = engine_->getBindingIndex("input");
    int idx_semi = engine_->getBindingIndex("semi");
    int idx_desc = engine_->getBindingIndex("desc");

    CHECK(cudaMemcpyAsync(buffers_[idx_input], img_float.data, input_size_, cudaMemcpyHostToDevice, stream_));
    context_->enqueueV2(buffers_, stream_, nullptr);
    CHECK(cudaMemcpyAsync(semi_host_, buffers_[idx_semi], semi_size_, cudaMemcpyDeviceToHost, stream_));
    CHECK(cudaMemcpyAsync(desc_host_, buffers_[idx_desc], desc_size_, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // 2. 后处理
    int Hc = input_h_ / 8; 
    int Wc = input_w_ / 8; 
    
    // 最佳实践：0.04 能平衡数量和质量
    float conf_thresh = 0.04; 

    std::vector<cv::KeyPoint> candidates;
    candidates.reserve(2000);

    for (int i = 0; i < Hc; ++i) {
        for (int j = 0; j < Wc; ++j) {
            int offset = i * Wc + j; 
            float dust_prob = semi_host_[64 * (Hc * Wc) + offset];
            
            if (dust_prob > 0.5) continue;

            float max_val = 0;
            int max_idx = -1;
            for (int c = 0; c < 64; ++c) {
                float val = semi_host_[c * (Hc * Wc) + offset];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }

            if (max_val > conf_thresh) {
                int r_blk = max_idx / 8;
                int c_blk = max_idx % 8;
                float pt_y = i * 8 + r_blk;
                float pt_x = j * 8 + c_blk;
                
                cv::KeyPoint kp;
                kp.pt = cv::Point2f(pt_x, pt_y);
                kp.response = max_val;
                candidates.push_back(kp);
            }
        }
    }

    // 3. Top-K 筛选 (1000个最佳点)
    if (candidates.size() > 1000) {
        std::sort(candidates.begin(), candidates.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });
        candidates.resize(1000);
    }
    keypoints = candidates;

    // 4. 描述子采样
    int num_kps = keypoints.size();
    if (num_kps == 0) return;

    descriptors = cv::Mat(num_kps, 256, CV_32F);
    
    for (int k = 0; k < num_kps; ++k) {
        int x_map = std::round(keypoints[k].pt.x / 8.0);
        int y_map = std::round(keypoints[k].pt.y / 8.0);
        x_map = std::max(0, std::min(x_map, Wc - 1));
        y_map = std::max(0, std::min(y_map, Hc - 1));
        int offset = y_map * Wc + x_map;

        for (int d = 0; d < 256; ++d) {
            float val = desc_host_[d * (Hc * Wc) + offset];
            descriptors.at<float>(k, d) = val;
        }
        
        // 必须进行 L2 归一化
        cv::Mat row = descriptors.row(k);
        cv::normalize(row, row, 1.0, 0.0, cv::NORM_L2);
    }
}