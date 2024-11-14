//
// Created by wangke on 2024/11/13.
//

#include "yolov8.h"
#include <cpu.h>
#include <iostream>
#include <vector>

const int MAX_STRIDE = 32;

typedef struct {
    cv::Rect box;
    float confidence;
    int index;
}BBOX_det;

bool cmp_score_det(BBOX_det box1, BBOX_det box2) {
    return box1.confidence > box2.confidence;
}


static float get_iou_value_det(cv::Rect rect1, cv::Rect rect2)
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

void my_nms_boxes_det(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float confThreshold, float nmsThreshold, std::vector<int>& indices)
{
    BBOX_det bbox;
    std::vector<BBOX_det> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), cmp_score_det);

    int updated_size = bboxes.size();
    for (i = 0; i < updated_size; i++)
    {
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (j = i + 1; j < updated_size; j++)
        {
            float iou = get_iou_value_det(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold)
            {
                bboxes.erase(bboxes.begin() + j);
                j=j-1;
                updated_size = bboxes.size();
            }
        }
    }
}

Inference_det::Inference_det(){
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Inference_det::loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu)
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
    sprintf(parampath, "yolov8%s_ncnn_model/model.ncnn.param", modeltype);
    sprintf(modelpath, "yolov8%s_ncnn_model/model.ncnn.bin", modeltype);

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

std::vector<Detection> Inference_det::runInference(const cv::Mat &input)
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

    // yolov8 has an output of shape (batchSize, 84,  8400) (class num + box[x,y,w,h])
    cv::Mat output(out.h, out.w, CV_32FC1, out.data);
    cv::transpose(output, output);
    //std::cout<<output.rows << output.cols << output.channels()<<std::endl;
    float* data = (float*)output.data;

    std::vector<int> class_ids;
    std::vector<float>  confidences;
    std::vector<cv::Rect> boxes;

    int rows = output.rows;
    int dimensions = output.cols;
    for (int row = 0; row < rows; row++) {
        float* score = (data + 4);
        cv::Mat scores(1, class_names.size(), CV_32FC1, score);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold) {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            int width = int(w);
            int height = int(h);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }
    std::vector<int> nms_result;
    my_nms_boxes_det(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);


    std::vector<Detection> detections;
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        float confidence = confidences[idx];
        int class_id = class_ids[idx];

        cv::Rect box = { int(((boxes[idx].x - int(wpad / 2)) / scale)),
                         int(((boxes[idx].y - int(hpad / 2))) / scale),
                         int(boxes[idx].width / scale),
                         int(boxes[idx].height / scale) };

        Detection det;
        det.box = box;
        det.confidence = confidence;
        det.class_id = class_id;
        detections.push_back(det);
    }
    return detections;
}

int Inference_det::draw(cv::Mat& rgb, const std::vector<Detection>& objects) {

    static const unsigned char colors[19][3] = {
            { 54,  67, 244},            { 99,  30, 233},            {176,  39, 156},            {183,  58, 103},
            {181,  81,  63},            {243, 150,  33},            {244, 169,   3},            {212, 188,   0},
            {136, 150,   0},            { 80, 175,  76},            { 74, 195, 139},            { 57, 220, 205},
            { 59, 235, 255},            {  7, 193, 255},            {  0, 152, 255},            { 34,  87, 255},
            { 72,  85, 121},            {158, 158, 158},            {139, 125,  96}
    };
    int color_index = 0;
    cv::Mat res = rgb;
    for (auto& obj : objects) {
        const unsigned char* color = colors[color_index % 19];
        color_index++;
        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(res, obj.box, { 0, 0, 255 }, 2);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.class_id].c_str(), obj.confidence * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        int x = obj.box.x;
        int y = obj.box.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }
    return 0;
}
