#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "personfeature_extractor.hpp"

#include "config.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

PersonFeatureExtractor::PersonFeatureExtractor(): is_initialized_(false),
      rknn_context_(std::make_unique<RKNNContext>()) 
{
    // 初始化RKNN上下文
    rknn_context_->ctx = 0;
    rknn_context_->input_attrs = nullptr;
    rknn_context_->output_attrs = nullptr;
    rknn_context_->model_width = 0;
    rknn_context_->model_height = 0;
    rknn_context_->is_quant = false;
}

/**
 * @brief 初始化Reid模型
 * @param model_path RKNN模型路径，默认为配置中的路径
 * @return 成功初始化返回true，失败返回false
 */
bool PersonFeatureExtractor::initialize(const std::string &model_path) {
    try {
        std::shared_ptr<Config> params_config = Config::getDefaultInstance();
        std::string actual_model_path = model_path.empty()
            ? params_config->getKeyValue<std::string>("person_reid_rknn_model")
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
            //  dump_tensor_attr(&(rknn_context_->input_attrs[i]));
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
        rknn_context_->model_width = rknn_context_->input_attrs[0].dims[2];
        rknn_context_->model_height = rknn_context_->input_attrs[0].dims[1];
        rknn_context_->model_channel = rknn_context_->input_attrs[0].dims[3];
        
        // 检查是否为量化模型
        rknn_context_->is_quant = (rknn_context_->input_attrs[0].type == RKNN_TENSOR_INT8);

        int core_num = params_config->getKeyValue<int>("rknn_model_inference");
        ret = rknn_set_core_mask(rknn_context_->ctx, (rknn_core_mask)core_num);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("模型运行NPU核心设置失败");
        }
        
        std::cout << "Reid模型初始化完成" << std::endl;
        // std::cout << "模型输入尺寸: " << rknn_context_->model_width << "x" << rknn_context_->model_height << std::endl;
        // std::cout << "量化模型: " << (rknn_context_->is_quant ? "是" : "否") << std::endl;
        
        is_initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Reid模型初始化失败: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

/**
 * @brief 检查是否已初始化
 * @return 已初始化返回true，否则返回false
 */
bool PersonFeatureExtractor::is_initialized() const {
    return is_initialized_;
}

PersonFeatureExtractor::~PersonFeatureExtractor()
{
    cleanup();
}

/**
 * @brief 清理资源，释放RKNN上下文
 */
void PersonFeatureExtractor::cleanup() {
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

std::vector<float> PersonFeatureExtractor::extractFeature(const cv::Mat &bbox_image, cv::Mat &thumbnail)
{
    // Match reid_test: crop -> resize with aspect ratio -> center pad
    thumbnail = preprocessPersonRegion(bbox_image);
    
    // Extract all features in a single optimized pass
    std::vector<float> feature_vector(FEATURE_DIMENSION);
    feature_vector.reserve(FEATURE_DIMENSION);
    
    // Extract features efficiently by reusing intermediate results
    extractAllFeatures(thumbnail, feature_vector);
    
    // Ensure the feature vector has exactly 1024 dimensions
    if (feature_vector.size() != static_cast<size_t>(FEATURE_DIMENSION)) {
        throw std::runtime_error("Feature dimension mismatch: expected " + 
                                std::to_string(FEATURE_DIMENSION) + ", got " + 
                                std::to_string(feature_vector.size()));
    }

    const float norm = std::sqrt(std::inner_product(feature_vector.begin(), feature_vector.end(),
                                                    feature_vector.begin(), 0.0f)) + 1e-6f;
    for (float& value : feature_vector) {
        value /= norm;
    }
    
    return feature_vector;
}

std::vector<std::vector<float>> PersonFeatureExtractor::extract(
    const cv::Mat& image,
    const std::vector<std::array<float, 4>>& boxes)
{
    std::vector<std::vector<float>> features;
    features.reserve(boxes.size());

    for (const auto& box : boxes)
    {
        const int x1 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(box[0]))));
        const int y1 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(box[1]))));
        const int x2 = std::max(0, std::min(image.cols, static_cast<int>(std::round(box[2]))));
        const int y2 = std::max(0, std::min(image.rows, static_cast<int>(std::round(box[3]))));

        if (x2 <= x1 || y2 <= y1)
        {
            features.emplace_back();
            continue;
        }

        try
        {
            cv::Mat thumbnail;
            cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
            features.push_back(extractFeature(crop, thumbnail));
        }
        catch (const std::exception&)
        {
            features.emplace_back();
        }
    }

    return features;
}

cv::Mat PersonFeatureExtractor::preprocessPersonRegion(const cv::Mat &image) {
    if(image.empty()){
        return cv::Mat();
    }
    return resizeWithAspectRatio(image, TARGET_HEIGHT, TARGET_WIDTH);
}

cv::Mat PersonFeatureExtractor::enhanceColor(const cv::Mat& image) {
    cv::Mat enhanced;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    
    // Apply CLAHE to each channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));
    
    for (auto& channel : channels) {
        clahe->apply(channel, channel);
    }
    
    cv::merge(channels, enhanced);
    return enhanced;
}

cv::Mat PersonFeatureExtractor::enhanceContours(const cv::Mat& image) {
    // Convert to grayscale for contour enhancement
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply Gaussian blur
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(0, 0), 3.0);
    
    // Unsharp mask: original - blurred * amount
    double amount = 1.5;
    cv::Mat unsharp;
    cv::addWeighted(gray, 1.0 + amount, blurred, -amount, 0, unsharp);
    
    // Convert back to color if original was color
    if (image.channels() == 3) {
        cv::Mat enhanced_color;
        cv::cvtColor(unsharp, enhanced_color, cv::COLOR_GRAY2BGR);
        return enhanced_color;
    }
    
    return unsharp;
}

cv::Mat PersonFeatureExtractor::resizeWithAspectRatio(const cv::Mat& image, int target_height, int target_width) {
    int orig_height = image.rows;
    int orig_width = image.cols;
    
    // Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
    double scale = std::min(static_cast<double>(target_height) / orig_height, 
                           static_cast<double>(target_width) / orig_width);
    
    int new_height = static_cast<int>(orig_height * scale);
    int new_width = static_cast<int>(orig_width * scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    
    // Match reid_test: center pad with 128 instead of zeros
    cv::Mat padded(target_height, target_width, image.type(), cv::Scalar(128, 128, 128));
    
    // Center the resized image on the canvas
    int y_offset = (target_height - new_height) / 2;
    int x_offset = (target_width - new_width) / 2;
    resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));
    
    return padded;
}

cv::Mat PersonFeatureExtractor::computeLBP(const cv::Mat& gray) {
    cv::Mat lbp_image = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    // Use optimized LBP computation with precomputed patterns
    for (int y = 1; y < gray.rows - 1; y++) {
        const uchar* center_ptr = gray.ptr<uchar>(y);
        uchar* lbp_ptr = lbp_image.ptr<uchar>(y);
        
        for (int x = 1; x < gray.cols - 1; x++) {
            unsigned char center = center_ptr[x];
            unsigned char code = 0;
            
            // Use pointer arithmetic for faster neighborhood access
            code |= (gray.at<uchar>(y-1, x-1) > center) << 7;
            code |= (gray.at<uchar>(y-1, x) > center) << 6;
            code |= (gray.at<uchar>(y-1, x+1) > center) << 5;
            code |= (gray.at<uchar>(y, x+1) > center) << 4;
            code |= (gray.at<uchar>(y+1, x+1) > center) << 3;
            code |= (gray.at<uchar>(y+1, x) > center) << 2;
            code |= (gray.at<uchar>(y+1, x-1) > center) << 1;
            code |= (gray.at<uchar>(y, x-1) > center) << 0;
            
            lbp_ptr[x] = LBP_TABLE[code];
        }
    }
    
    return lbp_image;
}

void PersonFeatureExtractor::extractAllFeatures(const cv::Mat& image, std::vector<float>& features) 
{
    if(image.empty()){
        std::cerr << "extractAllFeatures error: image is invalid" << std::endl;
        return ;
    }

    // Align with reid_test RKNN path:
    // crop/pad first, then send RGB image values to RKNN backend.
    cv::Mat rgb_u8;
    cv::cvtColor(image, rgb_u8, cv::COLOR_BGR2RGB);

    rknn_input inputs[rknn_context_->io_num.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    cv::Mat model_input;
    if (rknn_context_->is_quant) {
        model_input = rgb_u8;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = rknn_context_->model_width * rknn_context_->model_height * rknn_context_->model_channel;
        inputs[0].buf = model_input.data;
    } else {
        rgb_u8.convertTo(model_input, CV_32FC3);
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].size = rknn_context_->model_width * rknn_context_->model_height * rknn_context_->model_channel * sizeof(float);
        inputs[0].buf = model_input.data;
    }
    
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

    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 1;
    outputs[0].index = 0;
    outputs[0].size = FEATURE_DIMENSION * sizeof(float);
    outputs[0].buf = features.data();

    ret = rknn_outputs_get(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("获取模型输出失败");
    }

    // 释放输出
    rknn_outputs_release(rknn_context_->ctx, rknn_context_->io_num.n_output, outputs);

    return ;

    // cv::Mat hsv_image;
    // if (image.channels() == 1) {
    //     // Convert grayscale to BGR (which will be used for both RGB and HSV)
    //     cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(rgb_image, hsv_image, cv::COLOR_BGR2HSV);
    // } else {
        // Assume image is in BGR format (OpenCV standard), convert to RGB for histogram
        // cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        // cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    // }

    // Precompute all channel splits for statistical features
    // std::vector<cv::Mat> rgb_channels;
    // cv::split(rgb_image, rgb_channels);
    
    // std::vector<cv::Mat> hsv_channels;
    // cv::split(hsv_image, hsv_channels);

    // // Extract color histogram features with optimized spatial pyramid
    // std::vector<int> grid_levels = {1, 2, 4};
    // for (int level : grid_levels) {
    //     int cell_height = image.rows / level;
    //     int cell_width = image.cols / level;
        
    //     for (int i = 0; i < level; i++) {
    //         for (int j = 0; j < level; j++) {
    //             int y_start = i * cell_height;
    //             int y_end = (i == level - 1) ? image.rows : (i + 1) * cell_height;
    //             int x_start = j * cell_width;
    //             int x_end = (j == level - 1) ? image.cols : (j + 1) * cell_width;
                
    //             cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
                
    //             // Process RGB and HSV channels in the same region
    //             for (int c = 0; c < 3; c++) {
    //                 cv::Mat hist;
    //                 int channels[] = {c};
    //                 int histSize[] = {HISTOGRAM_BINS};
    //                 float range[] = {0, 256};
    //                 const float* ranges[] = {range};
                    
    //                 // RGB histogram - use temporary Mat for ROI
    //                 cv::Mat rgb_roi = rgb_image(roi);
    //                 cv::calcHist(&rgb_roi, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    //                 cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    //                 for (int b = 0; b < HISTOGRAM_BINS; b++) {
    //                     features.push_back(hist.at<float>(b));
    //                 }
                    
    //                 // HSV histogram - use temporary Mat for ROI
    //                 cv::Mat hsv_roi = hsv_image(roi);
    //                 cv::calcHist(&hsv_roi, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    //                 cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    //                 for (int b = 0; b < HISTOGRAM_BINS; b++) {
    //                     features.push_back(hist.at<float>(b));
    //                 }
    //             }
    //         }
    //     }
    // }

    // // Extract statistical features (reuse already split channels)
    // for (const auto& channel : rgb_channels) {
    //     cv::Scalar mean, stddev;
    //     cv::meanStdDev(channel, mean, stddev);
    //     features.push_back(static_cast<float>(mean[0]));
    //     features.push_back(static_cast<float>(stddev[0]));
    // }
    
    // for (const auto& channel : hsv_channels) {
    //     cv::Scalar mean, stddev;
    //     cv::meanStdDev(channel, mean, stddev);
    //     features.push_back(static_cast<float>(mean[0]));
    //     features.push_back(static_cast<float>(stddev[0]));
    // }

    // // Extract texture features using optimized LBP
    // cv::Mat gray;
    // if (image.channels() == 3) {
    //     cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // } else {
    //     gray = image.clone();
    // }
    
    // cv::Mat lbp_image = computeLBP(gray);
    
    // // Divide image into 4 horizontal strips and compute variance
    // int strip_height = lbp_image.rows / 4;
    // for (int i = 0; i < 4; i++) {
    //     int y_start = i * strip_height;
    //     int y_end = (i == 3) ? lbp_image.rows : (i + 1) * strip_height;
    //     cv::Mat strip = lbp_image(cv::Rect(0, y_start, lbp_image.cols, y_end - y_start));
        
    //     cv::Scalar mean, stddev;
    //     cv::meanStdDev(strip, mean, stddev);
    //     features.push_back(static_cast<float>(stddev[0]));
    // }
}
