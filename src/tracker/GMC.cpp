#include "tracker/GMC.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

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
    ++apply_count_;
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
    const std::vector<cv::KeyPoint> detected_kpts = kpts;
    if (!kpts.empty()) orb_extractor_->compute(small, kpts, desc);

    cv::Matx23f H = cv::Matx23f::eye();
    const size_t prev_kpt_count = prev_kpts_.size();
    int knn_count = 0;
    int kept_matches = 0;
    int inlier_count = 0;
    std::vector<cv::Point2f> kept_prev_debug;
    std::vector<cv::Point2f> kept_curr_debug;
    struct ProbeInfo {
        const char* name;
        cv::Point2f prev_pt;
        cv::Point2f curr_pt;
        int prev_idx = -1;
        int curr_idx = -1;
        int knn_len = 0;
        std::vector<std::pair<cv::Point2f, float>> topk;
        std::vector<std::pair<cv::Point2f, float>> nearest_prev;
        std::vector<std::pair<cv::Point2f, float>> nearest_curr_detected;
        std::vector<std::pair<cv::Point2f, float>> nearest_curr;
    };
    std::vector<ProbeInfo> probes = {
        {"cpp_only", cv::Point2f(238.0f, 218.0f), cv::Point2f(226.0f, 219.0f)},
        {"py_only",  cv::Point2f(395.0f, 114.0f), cv::Point2f(383.0f, 117.0f)},
    };

    if (!initialized_)
    {
        prev_gray_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        initialized_ = true;
        last_debug_json_ = "{\"stage\":\"init\"}";
        return H;
    }

    if (desc.empty() || prev_desc_.empty() || kpts.empty())
    {
        prev_gray_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        last_debug_json_ = "{\"stage\":\"empty_desc\"}";
        return H;
    }

    std::vector<std::vector<cv::DMatch>> knn;
    orb_matcher_->knnMatch(prev_desc_, desc, knn, 2);
    knn_count = static_cast<int>(knn.size());

    for (auto& probe : probes)
    {
        for (size_t i = 0; i < prev_kpts_.size(); ++i)
        {
            if (std::round(prev_kpts_[i].pt.x) == probe.prev_pt.x &&
                std::round(prev_kpts_[i].pt.y) == probe.prev_pt.y)
            {
                probe.prev_idx = static_cast<int>(i);
                break;
            }
        }
        for (size_t i = 0; i < kpts.size(); ++i)
        {
            if (std::round(kpts[i].pt.x) == probe.curr_pt.x &&
                std::round(kpts[i].pt.y) == probe.curr_pt.y)
            {
                probe.curr_idx = static_cast<int>(i);
                break;
            }
        }
        if (probe.prev_idx >= 0 && static_cast<size_t>(probe.prev_idx) < knn.size())
        {
            probe.knn_len = static_cast<int>(knn[probe.prev_idx].size());
            for (const auto& m : knn[probe.prev_idx])
            {
                if (m.trainIdx >= 0 && static_cast<size_t>(m.trainIdx) < kpts.size())
                    probe.topk.emplace_back(kpts[m.trainIdx].pt, m.distance);
            }
        }

        std::vector<std::pair<cv::Point2f, float>> prev_dists;
        prev_dists.reserve(prev_kpts_.size());
        for (const auto& kp : prev_kpts_)
        {
            const float dx = kp.pt.x - probe.prev_pt.x;
            const float dy = kp.pt.y - probe.prev_pt.y;
            prev_dists.emplace_back(kp.pt, std::sqrt(dx * dx + dy * dy));
        }
        std::sort(prev_dists.begin(), prev_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (size_t i = 0; i < std::min<size_t>(5, prev_dists.size()); ++i)
            probe.nearest_prev.push_back(prev_dists[i]);

        std::vector<std::pair<cv::Point2f, float>> curr_dists;
        curr_dists.reserve(kpts.size());
        for (const auto& kp : kpts)
        {
            const float dx = kp.pt.x - probe.curr_pt.x;
            const float dy = kp.pt.y - probe.curr_pt.y;
            curr_dists.emplace_back(kp.pt, std::sqrt(dx * dx + dy * dy));
        }
        std::sort(curr_dists.begin(), curr_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (size_t i = 0; i < std::min<size_t>(5, curr_dists.size()); ++i)
            probe.nearest_curr.push_back(curr_dists[i]);

        std::vector<std::pair<cv::Point2f, float>> curr_detect_dists;
        curr_detect_dists.reserve(detected_kpts.size());
        for (const auto& kp : detected_kpts)
        {
            const float dx = kp.pt.x - probe.curr_pt.x;
            const float dy = kp.pt.y - probe.curr_pt.y;
            curr_detect_dists.emplace_back(kp.pt, std::sqrt(dx * dx + dy * dy));
        }
        std::sort(curr_detect_dists.begin(), curr_detect_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        for (size_t i = 0; i < std::min<size_t>(5, curr_detect_dists.size()); ++i)
            probe.nearest_curr_detected.push_back(curr_detect_dists[i]);
    }

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
            kept_prev_debug.push_back(pk.pt);
            kept_curr_debug.push_back(ck.pt);
        }
    }
    kept_matches = static_cast<int>(prev_pts.size());

    if (prev_pts.size() >= 4)
    {
        cv::Mat inliers;
        cv::Mat Hm = cv::estimateAffinePartial2D(prev_pts, curr_pts, inliers, cv::RANSAC);
        if (!inliers.empty())
            inlier_count = cv::countNonZero(inliers);
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

    std::ostringstream dbg;
    dbg << "{\"prev_kpts\":" << prev_kpt_count
        << ",\"curr_kpts\":" << kpts.size()
        << ",\"knn_count\":" << knn_count
        << ",\"kept_matches\":" << kept_matches
        << ",\"inliers\":" << inlier_count
        << ",\"kept_prev_size\":" << kept_prev_debug.size()
        << ",\"kept_curr_size\":" << kept_curr_debug.size()
        << ",\"kept_prev\":[";
    for (size_t i = 0; i < kept_prev_debug.size(); ++i)
    {
        dbg << "[" << kept_prev_debug[i].x << "," << kept_prev_debug[i].y << "]";
        if (i + 1 < kept_prev_debug.size()) dbg << ",";
    }
    dbg << "],\"kept_curr\":[";
    for (size_t i = 0; i < kept_curr_debug.size(); ++i)
    {
        dbg << "[" << kept_curr_debug[i].x << "," << kept_curr_debug[i].y << "]";
        if (i + 1 < kept_curr_debug.size()) dbg << ",";
    }
    dbg << "]"
        << ",\"probes\":[";
    for (size_t i = 0; i < probes.size(); ++i)
    {
        const auto& probe = probes[i];
        dbg << "{\"name\":\"" << probe.name << "\""
            << ",\"prev_idx\":" << probe.prev_idx
            << ",\"curr_idx\":" << probe.curr_idx
            << ",\"knn_len\":" << probe.knn_len
            << ",\"topk\":[";
        for (size_t j = 0; j < probe.topk.size(); ++j)
        {
            dbg << "[[" << probe.topk[j].first.x << "," << probe.topk[j].first.y << "],"
                << probe.topk[j].second << "]";
            if (j + 1 < probe.topk.size()) dbg << ",";
        }
        dbg << "],\"nearest_prev\":[";
        for (size_t j = 0; j < probe.nearest_prev.size(); ++j)
        {
            dbg << "[[" << probe.nearest_prev[j].first.x << "," << probe.nearest_prev[j].first.y << "],"
                << probe.nearest_prev[j].second << "]";
            if (j + 1 < probe.nearest_prev.size()) dbg << ",";
        }
        dbg << "],\"nearest_curr_detected\":[";
        for (size_t j = 0; j < probe.nearest_curr_detected.size(); ++j)
        {
            dbg << "[[" << probe.nearest_curr_detected[j].first.x << "," << probe.nearest_curr_detected[j].first.y << "],"
                << probe.nearest_curr_detected[j].second << "]";
            if (j + 1 < probe.nearest_curr_detected.size()) dbg << ",";
        }
        dbg << "],\"nearest_curr\":[";
        for (size_t j = 0; j < probe.nearest_curr.size(); ++j)
        {
            dbg << "[[" << probe.nearest_curr[j].first.x << "," << probe.nearest_curr[j].first.y << "],"
                << probe.nearest_curr[j].second << "]";
            if (j + 1 < probe.nearest_curr.size()) dbg << ",";
        }
        dbg << "]}";
        if (i + 1 < probes.size()) dbg << ",";
    }
    dbg << "]"
        << ",\"H\":[" << H(0,0) << "," << H(0,1) << "," << H(0,2) << ","
        << H(1,0) << "," << H(1,1) << "," << H(1,2) << "]}";
    last_debug_json_ = dbg.str();

    if (apply_count_ >= 46)
    {
        std::ofstream dump("/tmp/cpp_gmc_pairs_" + std::to_string(apply_count_) + ".json");
        dump << dbg.str();
    }
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
