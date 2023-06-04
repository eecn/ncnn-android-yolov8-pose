//
// Created by wangke on 2023/5/15.
//

#ifndef YOLO_POSE_H
#define YOLO_POSE_H

#include <opencv2/core/core.hpp>
#include <net.h>

#include <vector>

struct Object_pose
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> kps;
};


class Yolo_pose
{
public:
    Yolo_pose();
    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object_pose>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.65f);

    int draw(cv::Mat& rgb, const std::vector<Object_pose>& objects);

private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //YOLO_POSE_H
