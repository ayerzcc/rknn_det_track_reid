/**
 * @file detector.hpp
 * @brief PersonDetector类头文件 - 定义人员检测器的接口
 * @details 基于RKNN API实现，支持RK3588多NPU核心优化
 */

#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

// RKNN API头文件
#include "rknn_api.h"

/**
 * @class PersonDetector
 * @brief 人员检测器类，负责使用RKNN模型进行人员检测
 * @note 支持RK3588多NPU核心并行推理，优化性能
 */
class PersonDetector {
public:
    /**
     * @brief 检测结果结构体
     */
    struct DetectionResult {
        std::vector<std::array<float, 4>> boxes;      ///< 边界框坐标 [x1, y1, x2, y2]
        std::vector<int> class_ids;                   ///< 类别ID
        std::vector<float> scores;                    ///< 置信度分数
    };

    /**
     * @brief 构造函数
     */
    PersonDetector();

    /**
     * @brief 析构函数，自动释放资源
     */
    ~PersonDetector();


    /**
     * @brief 初始化检测模型
     * @param model_path RKNN模型路径，默认为配置中的路径
     * @return 成功初始化返回true，失败返回false
     */
    bool initialize(const std::string& model_path = "");

    /**
     * @brief 执行人员检测
     * @param image 输入图像（BGR格式）
     * @return 检测结果结构体
     */
    bool detect(const cv::Mat& image, DetectionResult &result);

    /**
     * @brief 清理资源，释放RKNN上下文
     */
    void cleanup();

    /**
     * @brief 检查检测器是否已初始化
     * @return 已初始化返回true，否则返回false
     */
    bool is_initialized() const;

private:
    /**
     * @brief RKNN应用上下文结构体
     */
    struct RKNNContext {
        rknn_context ctx;                    ///< RKNN上下文
        rknn_input_output_num io_num;        ///< 输入输出数量
        rknn_tensor_attr* input_attrs;       ///< 输入属性
        rknn_tensor_attr* output_attrs;      ///< 输出属性
        int model_width;                     ///< 模型输入宽度
        int model_height;                    ///< 模型输入高度
        int model_channel;
        bool is_quant;                       ///< 是否量化模型
    };

    std::unique_ptr<RKNNContext> rknn_context_; ///< RKNN上下文
    bool is_initialized_;                       ///< 是否已初始化
};

#endif // DETECTOR_HPP
