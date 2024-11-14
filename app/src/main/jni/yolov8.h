//
// Created by wangke on 2024/11/13.
//

#ifndef NCNN_ANDROID_YOLOV8_H
#define NCNN_ANDROID_YOLOV8_H

#include <vector>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <net.h>

struct Detection
{
    int class_id{0};
    float confidence{0.0};
    cv::Rect box{};
};

class Inference_det
{
public:
    Inference_det();
    int loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu = false);
    std::vector<Detection> runInference(const cv::Mat &input);
    int draw(cv::Mat& rgb, const std::vector<Detection>& objects);

private:

    std::string modelPath{};
    bool gpuEnabled{};

    std::vector<std::string> class_names{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                                         "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                                         "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                                         "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                         "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                                         "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                                         "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};



    int modelShape;

    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.5};

    ncnn::Net net;

    float meanVals[3];
    float normVals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //NCNN_ANDROID_YOLOV8_H
