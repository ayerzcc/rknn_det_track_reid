#include "tracker/GMC.h"

#include <algorithm>
#include <cmath>

namespace bot_sort
{

// ------------------------------------------------------------------ 构造
GMC::GMC(GMCMethod method, int downscale)
    : method_(method),
      downscale_(std::max(1, downscale)),
      initialized_(false)
{
    if (method_ == GMCMethod::ORB)
    {
        orb_detector_ = cv::FastFeatureDetector::create(20);
        orb_extractor_ = cv::ORB::create();
        orb_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    }
    else if (method_ == GMCMethod::SparseOptFlow)
    {
        optflow_ = cv::SparsePyrLKOpticalFlow::create();
    }
}

// ------------------------------------------------------------------ apply
cv::Matx23f GMC::apply(const cv::Mat& frame,
                       const std::vector<cv::Rect2f>& detections)
{
    int h = frame.rows, w = frame.cols;
    if (h == 0 || w == 0) return cv::Matx23f::eye();

    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    if (method_ == GMCMethod::ORB)
        return applyORB(gray, w, h, detections);
    if (method_ == GMCMethod::SparseOptFlow)
        return applySparseOptFlow(gray, w, h, detections);

    return cv::Matx23f::eye();
}

// ------------------------------------------------------------------ mask
cv::Mat GMC::buildMask(const cv::Size& size,
                       const std::vector<cv::Rect2f>& dets,
                       int downscale)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    int bh = size.height, bw = size.width;
    mask(cv::Rect(bw / 50, bh / 50, bw * 48 / 50, bh * 48 / 50)).setTo(255);

    float s = 1.0f / downscale;
    for (const auto& d : dets)
    {
        cv::Rect r(static_cast<int>(d.x * s), static_cast<int>(d.y * s),
                   static_cast<int>(d.width * s), static_cast<int>(d.height * s));
        r = r & cv::Rect(0, 0, bw, bh);
        if (r.area() > 0) mask(r).setTo(0);
    }
    return mask;
}

// ------------------------------------------------------------------ ORB
cv::Matx23f GMC::applyORB(const cv::Mat& gray, int w, int h,
                           const std::vector<cv::Rect2f>& dets)
{
    int ds = downscale_;
    cv::Mat small;
    cv::resize(gray, small, cv::Size(w / ds, h / ds));
    int sh = small.rows, sw = small.cols;

    cv::Mat mask = buildMask(small.size(), dets, ds);

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    orb_detector_->detect(small, kpts, mask);
    if (!kpts.empty()) orb_extractor_->compute(small, kpts, desc);

    cv::Matx23f H = cv::Matx23f::eye();

    if (!initialized_)
    {
        prev_gray_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        initialized_ = true;
        return H;
    }

    if (desc.empty() || prev_desc_.empty() || kpts.empty())
    {
        prev_gray_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        return H;
    }

    std::vector<std::vector<cv::DMatch>> knn;
    orb_matcher_->knnMatch(prev_desc_, desc, knn, 2);

    const float max_spatial_x = 0.25f * static_cast<float>(w) / static_cast<float>(ds);
    const float max_spatial_y = 0.25f * static_cast<float>(h) / static_cast<float>(ds);
    std::vector<cv::Point2f> prev_pts, curr_pts;
    for (auto& pair : knn)
    {
        if (pair.size() < 2) continue;
        if (pair[0].distance >= 0.9f * pair[1].distance) continue;
        auto& pk = prev_kpts_[pair[0].queryIdx];
        auto& ck = kpts[pair[0].trainIdx];
        float dx = std::abs(pk.pt.x - ck.pt.x);
        float dy = std::abs(pk.pt.y - ck.pt.y);
        if (dx < max_spatial_x && dy < max_spatial_y)
        {
            prev_pts.push_back(pk.pt);
            curr_pts.push_back(ck.pt);
        }
    }

    if (prev_pts.size() >= 4)
    {
        cv::Mat inliers;
        cv::Mat Hm = cv::estimateAffinePartial2D(prev_pts, curr_pts, inliers, cv::RANSAC);
        if (!Hm.empty()) {
            H = cv::Matx23f(
                (float)Hm.at<double>(0,0), (float)Hm.at<double>(0,1), (float)Hm.at<double>(0,2),
                (float)Hm.at<double>(1,0), (float)Hm.at<double>(1,1), (float)Hm.at<double>(1,2));
        }
    }

    if (ds > 1) { H(0, 2) *= ds; H(1, 2) *= ds; }

    prev_gray_ = small.clone();
    prev_kpts_ = kpts;
    prev_desc_ = desc.clone();
    return H;
}

// ---------------------------------------------------------- SparseOptFlow
cv::Matx23f GMC::applySparseOptFlow(const cv::Mat& gray, int w, int h,
                                     const std::vector<cv::Rect2f>& dets)
{
    int ds = downscale_;
    cv::Mat small;
    cv::resize(gray, small, cv::Size(w / ds, h / ds));
    int sh = small.rows, sw = small.cols;

    cv::Mat mask = buildMask(small.size(), dets, ds);

    cv::Matx23f H = cv::Matx23f::eye();

    if (!initialized_)
    {
        prev_gray_ = small.clone();
        initialized_ = true;
        return H;
    }

    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(prev_gray_, prev_pts, 1000, 0.01, 1, mask);
    if (prev_pts.size() < 4)
    {
        prev_gray_ = small.clone();
        return H;
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(prev_gray_, small, prev_pts, curr_pts, status, cv::noArray(), cv::Size(21,21), 3);

    std::vector<cv::Point2f> pp, cp;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i]) { pp.push_back(prev_pts[i]); cp.push_back(curr_pts[i]); }
    }
    if (pp.size() >= 4)
    {
        cv::Mat inliers;
        cv::Mat Hm = cv::estimateAffinePartial2D(pp, cp, inliers, cv::RANSAC);
        if (!Hm.empty()) {
            H = cv::Matx23f(
                (float)Hm.at<double>(0,0), (float)Hm.at<double>(0,1), (float)Hm.at<double>(0,2),
                (float)Hm.at<double>(1,0), (float)Hm.at<double>(1,1), (float)Hm.at<double>(1,2));
        }
    }

    if (ds > 1) { H(0, 2) *= ds; H(1, 2) *= ds; }

    prev_gray_ = small.clone();
    return H;
}

} // namespace bot_sort
