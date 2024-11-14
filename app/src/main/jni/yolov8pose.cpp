//
// Created by wangke on 2024/4/21.
//

#include "yolov8pose.h"
#include <cpu.h>
#include <iostream>
#include <vector>

const int MAX_STRIDE = 32;
const int COCO_POSE_POINT_NUM = 17;

const std::vector<std::vector<unsigned int>> KPS_COLORS =
        { {0,   255, 0}, {0,   255, 0},  {0,   255, 0}, {0,   255, 0},
          {0,   255, 0},  {255, 128, 0},  {255, 128, 0}, {255, 128, 0},
          {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51,  153, 255},
          {51,  153, 255},{51,  153, 255},{51,  153, 255},{51,  153, 255},
          {51,  153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON =
        { {16, 14},  {14, 12},  {17, 15},  {15, 13},   {12, 13}, {6,  12},
          {7,  13},  {6,  7},   {6,  8},   {7,  9},   {8,  10},  {9,  11},
          {2,  3}, {1,  2},  {1,  3},  {2,  4},  {3,  5},   {4,  6},  {5,  7} };

const std::vector<std::vector<unsigned int>> LIMB_COLORS =
        { {51,  153, 255}, {51,  153, 255},   {51,  153, 255},
          {51,  153, 255}, {255, 51,  255},   {255, 51,  255},
          {255, 51,  255}, {255, 128, 0},     {255, 128, 0},
          {255, 128, 0},   {255, 128, 0},     {255, 128, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0} };

typedef struct {
    cv::Rect box;
    float confidence;
    int index;
}BBOX;

bool cmp_score(BBOX box1, BBOX box2) {
    return box1.confidence > box2.confidence;
}


static float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}

void my_nms_boxes(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float confThreshold, float nmsThreshold, std::vector<int>& indices)
{
    BBOX bbox;
    std::vector<BBOX> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), cmp_score);

    int updated_size = bboxes.size();
    for (i = 0; i < updated_size; i++)
    {
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (j = i + 1; j < updated_size; j++)
        {
            float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold)
            {
                bboxes.erase(bboxes.begin() + j);
                j=j-1;
                updated_size = bboxes.size();
            }
        }
    }
}

Inference::Inference(){
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Inference::loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu)
{
    modelShape = modelInputShape;
    gpuEnabled = useGpu;

    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = useGpu;
#endif

    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov8%s-pose_ncnn_model/model.ncnn.param", modeltype);
    sprintf(modelpath, "yolov8%s-pose_ncnn_model/model.ncnn.bin", modeltype);

    net.load_param(mgr, parampath);
    net.load_model(mgr, modelpath);

    this->meanVals[0] = meanVals[0];
    this->meanVals[1] = meanVals[1];
    this->meanVals[2] = meanVals[2];
    this->normVals[0] = normVals[0];
    this->normVals[1] = normVals[1];
    this->normVals[2] = normVals[2];
    return 0;
}

std::vector<Pose> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    int imgWidth = modelInput.cols;
    int imgHeight = modelInput.rows;

    int w = imgWidth;
    int h = imgHeight;
    float scale = 1.f;
    if (w > h) {
        scale = (float)modelShape / w;
        w = modelShape;
        h = (int)(h * scale);
    }
    else {
        scale = (float)modelShape / h;
        h = modelShape;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(modelInput.data, ncnn::Mat::PIXEL_BGR2RGB, imgWidth, imgHeight, w, h);

    int wpad = (modelShape + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (modelShape + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(meanVals, normVals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // yolov8 has an output of shape (batchSize, 56,  8400) (COCO_POSE_POINT_NUM x point[x,y,prop] + prop + box[x,y,w,h])
    cv::Mat output(out.h, out.w, CV_32FC1, out.data);
    cv::transpose(output, output);
    std::cout<<output.rows << output.cols << output.channels()<<std::endl;
    float* data = (float*)output.data;


    std::vector<float>  confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> keyPoints;

    int rows = output.rows;
    int dimensions = output.cols;
    for (int row = 0; row < rows; row++) {
        float score = *(data + 4);
        if (score > modelScoreThreshold) {
            confidences.push_back(score);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            int width = int(w);
            int height = int(h);

            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<float> kps((data + 5), data + 5+COCO_POSE_POINT_NUM * 3);
            keyPoints.push_back(kps);
        }
        data += dimensions;
    }
    std::vector<int> nms_result;
    my_nms_boxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);


    std::vector<Pose> poses;
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        float confidence = confidences[idx];

        cv::Rect box = { int(((boxes[idx].x - int(wpad / 2)) / scale)),
                         int(((boxes[idx].y - int(hpad / 2))) / scale),
                         int(boxes[idx].width / scale),
                         int(boxes[idx].height / scale) };
        std::vector<float> kps;
        for (int j = 0; j < keyPoints[idx].size()/3; j++) {
            kps.push_back((keyPoints[idx][3 * j + 0] - int(wpad / 2)) / scale);
            kps.push_back((keyPoints[idx][3 * j + 1] - int(hpad / 2)) / scale);
            kps.push_back(keyPoints[idx][3 * j + 2]);
        };
        Pose pose;
        pose.box = box;
        pose.confidence = confidence;
        pose.kps = kps; //{ confidence, box, kps };
        poses.push_back(pose);
    }
    return poses;
}

int Inference::draw(cv::Mat& rgb, const std::vector<Pose>& objects) {

    cv::Mat res = rgb;
    for (auto& obj : objects) {
        cv::rectangle(res, obj.box, { 0, 0, 255 }, 2);

        char text[256];
        sprintf(text, "person %.1f%%", obj.confidence * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x1 = obj.box.x;
        int y1 = obj.box.y - label_size.height - baseLine;
        if (y1 < 0)
            y1 = 0;
        if (x1 + label_size.width > rgb.cols)
            x1 = rgb.cols - label_size.width;
        cv::putText(rgb, text, cv::Point(x1, y1 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        int x = (int)obj.box.x;
        int y = (int)obj.box.y + 1;

        if (y > res.rows)
            y = res.rows;

        auto& kps = obj.kps;
        for (int k = 0; k < COCO_POSE_POINT_NUM + 2; k++) {
            if (k < COCO_POSE_POINT_NUM) {
                int kps_x = (int)std::round(kps[k * 3]);
                int kps_y = (int)std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.4f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, { kps_x, kps_y }, 5, kps_color, -1);
                }
            }
            auto& ske = SKELETON[k];
            int pos1_x = (int)std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = (int)std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = (int)std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = (int)std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color, 2);
            }
        }
    }
    return 0;
}
