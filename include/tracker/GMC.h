#pragma once

#include <opencv2/opencv.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace bot_sort
{

enum class GMCMethod
{
    ORB = 0,
    SparseOptFlow = 1,
    None = 2
};

class GMC
{
public:
    GMC(GMCMethod method = GMCMethod::ORB, int downscale = 2);

    /**
     * 估计帧间全局运动变换矩阵。
     * @param frame 当前帧 (BGR)
     * @param detections 检测框 [x1,y1,x2,y2] (可选，用于掩码前景)
     * @return 2x3 仿射变换矩阵。失败时返回单位矩阵。
     */
    cv::Matx23f apply(const cv::Mat& frame,
                      const std::vector<cv::Rect2f>& detections = {});
    const std::string& getLastDebugJson() const { return last_debug_json_; }

private:
    GMCMethod method_;
    int downscale_;

    // ORB
    cv::Ptr<cv::Feature2D> orb_detector_;
    cv::Ptr<cv::ORB> orb_extractor_;
    cv::Ptr<cv::BFMatcher> orb_matcher_;

    // SparseOptFlow
    cv::Ptr<cv::SparseOpticalFlow> optflow_;

    // 状态
    cv::Mat prev_gray_;
    std::vector<cv::KeyPoint> prev_kpts_;
    cv::Mat prev_desc_;
    bool initialized_;
    std::string last_debug_json_;
    size_t apply_count_ = 0;

    cv::Matx23f applyORB(const cv::Mat& gray, int w, int h,
                         const std::vector<cv::Rect2f>& dets);
    cv::Matx23f applySparseOptFlow(const cv::Mat& gray, int w, int h,
                                    const std::vector<cv::Rect2f>& dets);

    static cv::Mat buildMask(const cv::Size& size,
                             const std::vector<cv::Rect2f>& dets,
                             int downscale);
};

} // namespace bot_sort
