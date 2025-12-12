#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class SuperPoint {
public:
    SuperPoint(const std::string& engine_path);
    ~SuperPoint();

    void extractFeatures(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
        }
    } logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;

    void* buffers_[3]; 
    int input_h_ = 480;
    int input_w_ = 640;
    
    size_t input_size_;
    size_t semi_size_;
    size_t desc_size_;

    float* semi_host_;
    float* desc_host_;
};