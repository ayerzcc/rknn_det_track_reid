#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace bot_sort
{

class OcclusionAwareModule
{
public:
    OcclusionAwareModule(float k_x = 4.24264f, float k_y = 3.0f,
                         float thre_occ = 5.0f);

    /**
     * 计算精细化遮挡系数。
     * @param bboxes 边界框列表 [x1,y1,x2,y2]
     * @param img_h  图像高度（0=自动推断）
     * @param img_w  图像宽度（0=自动推断）
     * @return 每个框的遮挡系数 [0,1]
     */
    std::vector<float> computeOcclusionCoefficients(
        const std::vector<cv::Rect2f>& bboxes,
        int img_h = 0, int img_w = 0);

private:
    float k_x_, k_y_, thre_occ_;

    static float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b);
    cv::Mat computeGaussianMap(const std::vector<cv::Rect2f>& bboxes,
                               int img_h, int img_w);
};

class OcclusionAwareOffset
{
public:
    explicit OcclusionAwareOffset(float tau = 0.15f);

    /**
     * 精细化空间一致性分数。
     * S = tau * (1 - Oc) + (1 - tau) * IoU
     * @param estimations (M,4) 轨迹估计框
     * @param iou_matrix  (M,N) IoU 矩阵
     * @return (M,N) 精细化后的一致性分数
     */
    cv::Mat refineSpatialConsistency(
        const std::vector<cv::Rect2f>& estimations,
        const cv::Mat& iou_matrix);

private:
    float tau_;
    OcclusionAwareModule oam_;
};

class BiasAwareMomentum
{
public:
    BiasAwareMomentum();

    float computeMomentum(const cv::Rect2f& estimation,
                          const cv::Rect2f& detection,
                          float oc_last_obs);

    cv::Rect2f refineObservation(const cv::Rect2f& estimation,
                                 const cv::Rect2f& detection,
                                 float oc_last_obs);

private:
    OcclusionAwareModule oam_;
    static float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b);
};

} // namespace bot_sort
