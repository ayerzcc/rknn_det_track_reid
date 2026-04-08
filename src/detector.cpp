/**
 * @file detector.cpp
 * @brief PersonDetector类的实现文件 - 基于RKNN API的人员检测器
 * @details 支持RK3588多NPU核心优化，实现高效人员检测
 */

#include "detector.hpp"
#include "config.hpp"  // 添加Config头文件
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <cfloat>  // 添加FLT_MAX支持

#include "config.hpp"

std::shared_ptr<Config> params_config = Config::getDefaultInstance();
int OBJ_CLASS_NUM = params_config->getKeyValue<int>("detect_class_num");
float DETECT_SCORE_THRESHOLD = params_config->getKeyValue<float>("detect_score_threshold");
float NMS_IOU_THRESHOLD = params_config->getKeyValue<float>("nms_iou_threshold");

static inline bool is_nhwc_tensor(const rknn_tensor_attr& attr)
{
    return attr.fmt == RKNN_TENSOR_NHWC;
}

static inline int tensor_height(const rknn_tensor_attr& attr)
{
    return is_nhwc_tensor(attr) ? attr.dims[1] : attr.dims[2];
}

static inline int tensor_width(const rknn_tensor_attr& attr)
{
    return is_nhwc_tensor(attr) ? attr.dims[2] : attr.dims[3];
}

static inline int tensor_channel(const rknn_tensor_attr& attr)
{
    return is_nhwc_tensor(attr) ? attr.dims[3] : attr.dims[1];
}

inline static int clamp(float val, int min, int max) 
{ 
    return val > min ? (val < max ? val : max) : min; 
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) 
{ 
    return ((float)qnt - (float)zp) * scale; 
}

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// YOLOv2.6 direct regression format: box=(cx,cy,cw,ch), single-class sigmoid score
static int process_yolo26_direct(int8_t *box_tensor, int32_t box_zp, float box_scale,
                                  int8_t *score_tensor, int32_t score_zp, float score_scale,
                                  int grid_h, int grid_w, int stride,
                                  std::vector<float> &boxes,
                                  std::vector<float> &objProbs,
                                  std::vector<int> &classId,
                                  float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            // score: sigmoid(dequantized int8)
            float cls_score = sigmoid(deqnt_affine_to_f32(score_tensor[i * grid_w + j], score_zp, score_scale));
            if (cls_score < threshold) {
                continue;
            }

            // box: (cx, cy, cw, ch) in CHW layout
            float cx = deqnt_affine_to_f32(box_tensor[0 * grid_len + i * grid_w + j], box_zp, box_scale);
            float cy = deqnt_affine_to_f32(box_tensor[1 * grid_len + i * grid_w + j], box_zp, box_scale);
            float cw = deqnt_affine_to_f32(box_tensor[2 * grid_len + i * grid_w + j], box_zp, box_scale);
            float ch = deqnt_affine_to_f32(box_tensor[3 * grid_len + i * grid_w + j], box_zp, box_scale);

            float xmin = (j + 0.5f - cx) * stride;
            float ymin = (i + 0.5f - cy) * stride;
            float xmax = (j + 0.5f + cw) * stride;
            float ymax = (i + 0.5f + ch) * stride;

            if (xmin < 0) xmin = 0;
            if (ymin < 0) ymin = 0;
            if (xmax < 0 || ymax < 0 || xmax <= xmin || ymax <= ymin) {
                continue;
            }

            float w = xmax - xmin;
            float h = ymax - ymin;
            boxes.push_back(xmin);
            boxes.push_back(ymin);
            boxes.push_back(w);
            boxes.push_back(h);

            objProbs.push_back(cls_score);
            classId.push_back(0);
            validCount++;
        }
    }
    return validCount;
}

// Original DFL-based format: multi-class int8 score, DFL box decoding, optional score_sum
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_i8 && max_class_id == 0){
                offset = i* grid_w + j;
                float box[4];
                if (dfl_len <= 1) {
                    // Direct regression: box=(cx,cy,cw,ch)
                    float cx = deqnt_affine_to_f32(box_tensor[offset + 0 * grid_len], box_zp, box_scale);
                    float cy = deqnt_affine_to_f32(box_tensor[offset + 1 * grid_len], box_zp, box_scale);
                    float cw = deqnt_affine_to_f32(box_tensor[offset + 2 * grid_len], box_zp, box_scale);
                    float ch = deqnt_affine_to_f32(box_tensor[offset + 3 * grid_len], box_zp, box_scale);
                    box[0] = cx;
                    box[1] = cy;
                    box[2] = cw;
                    box[3] = ch;
                } else {
                    // DFL format: box_tensor is [dfl_len*4, H, W]
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k * grid_len], box_zp, box_scale);
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                }

                float x1,y1,x2,y2,w,h;
                if (dfl_len <= 1) {
                    // Direct regression: (cx,cy,cw,ch) format
                    x1 = (j + 0.5f - box[0]) * stride;
                    y1 = (i + 0.5f - box[1]) * stride;
                    x2 = (j + 0.5f + box[2]) * stride;
                    y2 = (i + 0.5f + box[3]) * stride;
                } else {
                    // DFL format
                    x1 = (-box[0] + j + 0.5)*stride;
                    y1 = (-box[1] + i + 0.5)*stride;
                    x2 = (box[2] + j + 0.5)*stride;
                    y2 = (box[3] + i + 0.5)*stride;
                }
                w = x2 - x1;
                h = y2 - y1;
                if (w <= 0 || h <= 0) continue;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                if (dfl_len <= 1) {
                    // Direct regression: single-class, apply sigmoid
                    objProbs.push_back(sigmoid(deqnt_affine_to_f32(max_score, score_zp, score_scale)));
                } else {
                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                }
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}


PersonDetector::PersonDetector() 
    : is_initialized_(false),
      rknn_context_(std::make_unique<RKNNContext>()) {
    // 初始化RKNN上下文
    rknn_context_->ctx = 0;
    rknn_context_->input_attrs = nullptr;
    rknn_context_->output_attrs = nullptr;
    rknn_context_->model_width = 0;
    rknn_context_->model_height = 0;
    rknn_context_->is_quant = false;
}

/**
 * @brief 析构函数，自动释放资源
 */
PersonDetector::~PersonDetector() {
    cleanup();
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

/**
 * @brief 初始化检测模型
 * @param model_path RKNN模型路径，默认为配置中的路径
 * @return 成功初始化返回true，失败返回false
 */
bool PersonDetector::initialize(const std::string& model_path) {
    try 
    {
        std::string actual_model_path = model_path.empty()
            ? params_config->getKeyValue<std::string>("detect_rknn_model")
            : model_path;
        
        // 加载RKNN模型
        // 需要将const char*转换为void*以匹配rknn_init签名
        int ret = rknn_init(&rknn_context_->ctx, 
                       const_cast<void*>(static_cast<const void*>(actual_model_path.c_str())), 
                       0, 
                       0,
                       nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("RKNN模型初始化失败, 错误码: " + std::to_string(ret));
        }
        
        // 获取模型输入输出信息
        ret = rknn_query(rknn_context_->ctx, RKNN_QUERY_IN_OUT_NUM, 
                        &rknn_context_->io_num, sizeof(rknn_context_->io_num));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("获取模型输入输出数量失败");
        }
        
        // 获取输入属性
        rknn_context_->input_attrs = new rknn_tensor_attr[rknn_context_->io_num.n_input];
        for (uint32_t i = 0; i < rknn_context_->io_num.n_input; i++) {
            rknn_context_->input_attrs[i].index = i;
            ret = rknn_query(rknn_context_->ctx, RKNN_QUERY_INPUT_ATTR, 
                            &(rknn_context_->input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC) {
                throw std::runtime_error("获取输入属性失败");
            }
            // dump_tensor_attr(&(rknn_context_->input_attrs[i]));
        }
        
        // 获取输出属性
        rknn_context_->output_attrs = new rknn_tensor_attr[rknn_context_->io_num.n_output];
        for (uint32_t i = 0; i < rknn_context_->io_num.n_output; i++) {
            rknn_context_->output_attrs[i].index = i;
            ret = rknn_query(rknn_context_->ctx, RKNN_QUERY_OUTPUT_ATTR, 
                            &(rknn_context_->output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC) {
                throw std::runtime_error("获取输出属性失败");
            }
            // dump_tensor_attr(&(rknn_context_->output_attrs[i]));
        }
        
        // 获取模型输入尺寸
        rknn_context_->model_width = tensor_width(rknn_context_->input_attrs[0]);
        rknn_context_->model_height = tensor_height(rknn_context_->input_attrs[0]);
        rknn_context_->model_channel = tensor_channel(rknn_context_->input_attrs[0]);
        
        // 检查是否为量化模型
        rknn_context_->is_quant = (rknn_context_->input_attrs[0].type == RKNN_TENSOR_INT8);

        int core_num = params_config->getKeyValue<int>("rknn_model_inference");
        ret = rknn_set_core_mask(rknn_context_->ctx, (rknn_core_mask)core_num);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("模型运行NPU核心设置失败");
        }

        std::cout << "检测模型初始化完成" << std::endl;
        // std::cout << "模型输入尺寸: " << rknn_context_->model_width << "x" << rknn_context_->model_height << std::endl;
        // std::cout << "量化模型: " << (rknn_context_->is_quant ? "是" : "否") << std::endl;
        
        is_initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "检测模型初始化失败: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

/**
 * @brief 执行人员检测
 * @param image 输入图像（BGR格式）
 * @return 检测结果结构体
 */
bool PersonDetector::detect(const cv::Mat& image, DetectionResult &result) {
    if (!is_initialized_) {
        throw std::runtime_error("检测模型未初始化");
    }

    result.boxes.clear();
    result.class_ids.clear();
    result.scores.clear();

    // 准备输入数据
    rknn_input inputs[rknn_context_->io_num.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = rknn_context_->model_width * rknn_context_->model_height * rknn_context_->model_channel;
    inputs[0].buf = image.data;
    
    // 设置输入
    int ret = rknn_inputs_set(rknn_context_->ctx, rknn_context_->io_num.n_input, inputs);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("设置模型输入失败");
    }
    
    // 执行推理
    ret = rknn_run(rknn_context_->ctx, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("模型推理失败");
    }
    
    // 获取输出
    rknn_output outputs[rknn_context_->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < rknn_context_->io_num.n_output; i++) {
        outputs[i].want_float = false; // 总是获取浮点输出
    }
    
    ret = rknn_outputs_get(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("获取模型输出失败");
    }

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    std::vector<int> indexArray;

    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = rknn_context_->model_width;
    int model_in_h = rknn_context_->model_height;

    const int num_outputs = static_cast<int>(rknn_context_->io_num.n_output);
    int output_per_branch = 0;
    if (num_outputs >= 3 && num_outputs % 3 == 0)
    {
        const rknn_tensor_attr& maybe_score_sum_attr = rknn_context_->output_attrs[2];
        const rknn_tensor_attr& first_box_attr = rknn_context_->output_attrs[0];
        if (tensor_height(maybe_score_sum_attr) == tensor_height(first_box_attr) &&
            tensor_width(maybe_score_sum_attr) == tensor_width(first_box_attr) &&
            tensor_channel(maybe_score_sum_attr) == 1)
        {
            output_per_branch = 3;
        }
    }
    if (output_per_branch == 0 && num_outputs % 2 == 0)
    {
        output_per_branch = 2;
    }
    if (output_per_branch == 0)
    {
        rknn_outputs_release(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs);
        throw std::runtime_error("检测模型输出数量不符合后处理预期");
    }

    const int branch_count = num_outputs / output_per_branch;
    for (int i = 0; i < branch_count; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0f;
        const int box_idx = i * output_per_branch;
        const int score_idx = box_idx + 1;
        const rknn_tensor_attr& box_attr = rknn_context_->output_attrs[box_idx];
        const int reg_channel = tensor_channel(box_attr);
        const int dfl_len = reg_channel / 4;

        if (output_per_branch == 3)
        {
            score_sum = outputs[box_idx + 2].buf;
            score_sum_zp = rknn_context_->output_attrs[box_idx + 2].zp;
            score_sum_scale = rknn_context_->output_attrs[box_idx + 2].scale;
        }

        grid_h = tensor_height(box_attr);
        grid_w = tensor_width(box_attr);
        if (grid_h <= 0 || grid_w <= 0)
        {
            continue;
        }
        stride = model_in_h / grid_h;

        if (rknn_context_->is_quant)
        {
            if (reg_channel == 4)
            {
                validCount += process_yolo26_direct(
                    (int8_t *)outputs[box_idx].buf, box_attr.zp, box_attr.scale,
                    (int8_t *)outputs[score_idx].buf, rknn_context_->output_attrs[score_idx].zp, rknn_context_->output_attrs[score_idx].scale,
                    grid_h, grid_w, stride,
                    filterBoxes, objProbs, classId, DETECT_SCORE_THRESHOLD);
            }
            else if (dfl_len > 1)
            {
                validCount += process_i8((int8_t *)outputs[box_idx].buf, box_attr.zp, box_attr.scale,
                                         (int8_t *)outputs[score_idx].buf, rknn_context_->output_attrs[score_idx].zp, rknn_context_->output_attrs[score_idx].scale,
                                         (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                         grid_h, grid_w, stride, dfl_len,
                                         filterBoxes, objProbs, classId, DETECT_SCORE_THRESHOLD);
            }
        }
    }

    if (validCount <= 0)
    {
        filterBoxes.clear();
        objProbs.clear();
        classId.clear();
        indexArray.clear();

        rknn_outputs_release(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs);
        return false;
    }
    
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set){
        nms(validCount, filterBoxes, classId, indexArray, c, NMS_IOU_THRESHOLD);
    }

    int last_count = 0;
    // /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1){
            continue;
        }

        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] ;
        float y1 = filterBoxes[n * 4 + 1] ;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[n];

        std::array<float, 4> box;
        box[0] = (int)(clamp(x1, 0, model_in_w) );
        box[1] = (int)(clamp(y1, 0, model_in_h) );
        box[2] = (int)(clamp(x2, 0, model_in_w) );
        box[3] = (int)(clamp(y2, 0, model_in_h) );

        result.boxes.push_back(box);
        result.scores.push_back(obj_conf);
        result.class_ids.push_back(id);

        last_count++;
    }

    filterBoxes.clear();
    objProbs.clear();
    classId.clear();
    indexArray.erase(indexArray.begin(), indexArray.end());

    // 释放输出
    rknn_outputs_release(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs);
    
    return true;
}   

/**
 * @brief 清理资源，释放RKNN上下文
 */
void PersonDetector::cleanup() {
    if (rknn_context_->ctx) {
        rknn_destroy(rknn_context_->ctx);
        rknn_context_->ctx = 0;
    }
    
    if (rknn_context_->input_attrs) {
        delete[] rknn_context_->input_attrs;
        rknn_context_->input_attrs = nullptr;
    }
    
    if (rknn_context_->output_attrs) {
        delete[] rknn_context_->output_attrs;
        rknn_context_->output_attrs = nullptr;
    }
    
    is_initialized_ = false;
}

/**
 * @brief 检查检测器是否已初始化
 * @return 已初始化返回true，否则返回false
 */
bool PersonDetector::is_initialized() const {
    return is_initialized_;
}
