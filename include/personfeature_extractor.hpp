#ifndef __PERSON_FEATURE_EXTRACTOR_HPP__
#define __PERSON_FEATURE_EXTRACTOR_HPP__

#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

#include "rknn_api.h"

class PersonFeatureExtractor
{
    public:
        PersonFeatureExtractor();
        ~PersonFeatureExtractor();

    public:
        /**
         * @brief Extract a 512-dimensional feature vector from a person bounding box
         * @param image Input image containing the person
         * @param bbox Bounding box of the person in the image
         * @return 512-dimensional feature vector with high discriminative power
         */
        std::vector<float> extractFeature(const cv::Mat &bbox_image, cv::Mat &thumbnail);

        /**
         * @brief Batch extract features from frame + xyxy boxes, aligned with reid_test interface
         * @param image Full input frame
         * @param boxes Detection boxes in [x1, y1, x2, y2]
         * @return N x 512 feature vectors
         */
        std::vector<std::vector<float>> extract(const cv::Mat& image,
                                                const std::vector<std::array<float, 4>>& boxes);

        /**
         * @brief 检查姿态估计器是否已初始化
         * @return 已初始化返回true，否则返回false
         */
       bool initialize(const std::string &model_path = "");

        /**
        * @brief 清理资源
        */
       void cleanup();

    private:
        /**
         * @brief Crop and preprocess the person region from the image
         * @param image Input image
         * @param bbox Bounding box
         * @return Preprocessed person region
         */
        cv::Mat preprocessPersonRegion(const cv::Mat &image);

        /**
         * @brief Enhance color contrast using CLAHE
         * @param image Input image
         * @return Color-enhanced image
         */
        cv::Mat enhanceColor(const cv::Mat &image);

        /**
         * @brief Enhance contours using unsharp masking
         * @param image Input image
         * @return Contour-enhanced image
         */
        cv::Mat enhanceContours(const cv::Mat &image);

        /**
         * @brief Resize image to target size while maintaining aspect ratio
         * @param image Input image
         * @param target_height Target height
         * @param target_width Target width
         * @return Resized image with padding if necessary
         */
        cv::Mat resizeWithAspectRatio(const cv::Mat &image, int target_height, int target_width);

        /**
         * @brief Extract all features in a single optimized pass
         * @param image Input image
         * @param features Output feature vector to populate
         */
        void extractAllFeatures(const cv::Mat &image, std::vector<float> &features);

        /**
         * @brief Compute LBP image efficiently
         * @param gray Input grayscale image
         * @return LBP image
         */
        cv::Mat computeLBP(const cv::Mat &gray);

        /**
         * @brief 检查是否已初始化
         * @return 已初始化返回true，否则返回false
         */
        bool is_initialized() const ;
    
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
        bool is_initialized_;       

        // Parameters
        const int TARGET_HEIGHT = 256;
        const int TARGET_WIDTH = 128;
        const int HISTOGRAM_BINS = 8;
        const int FEATURE_DIMENSION = 512;
        
        // Precomputed LBP patterns for optimization
        static constexpr unsigned char LBP_TABLE[256] = {
            // Precomputed LBP pattern table for faster computation
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
            32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
            64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
            96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
            128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
            160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
            192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
            224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255
        };
};

#endif
