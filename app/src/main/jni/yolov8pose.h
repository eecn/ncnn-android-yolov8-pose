//
// Created by wangke on 2024/4/21.
//

#ifndef NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H
#define NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H

#include <vector>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>

#include <net.h>

struct Pose
{
    float confidence{0.0};
    cv::Rect box{};
    std::vector<float> kps;
};

class Inference
{
public:
    Inference();
    int loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu = false);
    std::vector<Pose> runInference(const cv::Mat &input);
    int draw(cv::Mat& rgb, const std::vector<Pose>& objects);

private:
    //void loadNcnnNetwork();

    std::string modelPath{};
    bool gpuEnabled{};

    int modelShape;

    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    ncnn::Net net;

    float meanVals[3];
    float normVals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H
