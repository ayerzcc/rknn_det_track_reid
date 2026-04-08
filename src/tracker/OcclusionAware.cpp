#include "tracker/OcclusionAware.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bot_sort
{

// ============================================================
//  OcclusionAwareModule
// ============================================================
OcclusionAwareModule::OcclusionAwareModule(float k_x, float k_y, float thre_occ)
    : k_x_(k_x), k_y_(k_y), thre_occ_(thre_occ) {}

float OcclusionAwareModule::computeIoU(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float ix1 = std::max(a.x, b.x);
    float iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.width, b.x + b.width);
    float iy2 = std::min(a.y + a.height, b.y + b.height);
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
    float inter = (ix2 - ix1) * (iy2 - iy1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float uni = area_a + area_b - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

cv::Mat OcclusionAwareModule::computeGaussianMap(
    const std::vector<cv::Rect2f>& bboxes, int img_h, int img_w)
{
    img_h = std::max(img_h, 1);
    img_w = std::max(img_w, 1);
    cv::Mat gm = cv::Mat::zeros(img_h, img_w, CV_32FC1);

    for (const auto& bb : bboxes)
    {
        float cx = bb.x + bb.width / 2.0f;
        float cy = bb.y + bb.height / 2.0f;
        float sigma_x = bb.width / k_x_;
        float sigma_y = bb.height / k_y_;
        if (sigma_x <= 0 || sigma_y <= 0) continue;

        int x1 = std::max(0, static_cast<int>(std::floor(bb.x)));
        int y1 = std::max(0, static_cast<int>(std::floor(bb.y)));
        int x2 = std::min(img_w, static_cast<int>(std::ceil(bb.x + bb.width)));
        int y2 = std::min(img_h, static_cast<int>(std::ceil(bb.y + bb.height)));
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Mat local_gm = cv::Mat::zeros(y2 - y1, x2 - x1, CV_32FC1);
        for (int yy = y1; yy < y2; ++yy)
        {
            float dy = yy - cy;
            float gy = std::exp(-dy * dy / (2.0f * sigma_y * sigma_y));
            for (int xx = x1; xx < x2; ++xx)
            {
                float dx = xx - cx;
                float gx = std::exp(-dx * dx / (2.0f * sigma_x * sigma_x));
                local_gm.at<float>(yy - y1, xx - x1) = gx * gy;
            }
        }
        cv::Mat roi = gm(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::max(roi, local_gm, roi);
    }
    return gm;
}

std::vector<float> OcclusionAwareModule::computeOcclusionCoefficients(
    const std::vector<cv::Rect2f>& bboxes, int img_h, int img_w)
{
    size_t N = bboxes.size();
    if (N == 0) return {};

    // 规范化 bbox
    std::vector<cv::Rect2f> normed(N);
    for (size_t i = 0; i < N; ++i)
    {
        float x1 = std::min(bboxes[i].x, bboxes[i].x + bboxes[i].width);
        float y1 = std::min(bboxes[i].y, bboxes[i].y + bboxes[i].height);
        float x2 = std::max(bboxes[i].x, bboxes[i].x + bboxes[i].width);
        float y2 = std::max(bboxes[i].y, bboxes[i].y + bboxes[i].height);
        normed[i] = cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
    }

    std::vector<float> oc_hat(N, 0.0f);

    // IoU 矩阵
    std::vector<std::vector<float>> iou_mat(N, std::vector<float>(N, 0.0f));
    float max_iou = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        if (normed[i].area() <= 0) continue;
        for (size_t j = i + 1; j < N; ++j)
        {
            if (normed[j].area() <= 0) continue;
            float iou = computeIoU(normed[i], normed[j]);
            iou_mat[i][j] = iou;
            iou_mat[j][i] = iou;
            max_iou = std::max(max_iou, iou);
        }
    }
    if (max_iou <= 0) return oc_hat;

    // 确定图像尺寸
    if (img_h <= 0 || img_w <= 0)
    {
        for (const auto& b : normed)
        {
            img_w = std::max(img_w, static_cast<int>(std::ceil(b.x + b.width)) + 1);
            img_h = std::max(img_h, static_cast<int>(std::ceil(b.y + b.height)) + 1);
        }
    }

    // 只处理有重叠的框
    std::vector<int> overlap_idx;
    for (size_t i = 0; i < N; ++i)
    {
        bool has_overlap = false;
        for (size_t j = 0; j < N; ++j)
        {
            if (i != j && iou_mat[i][j] > 0) { has_overlap = true; break; }
        }
        if (has_overlap) overlap_idx.push_back(static_cast<int>(i));
    }
    if (overlap_idx.empty()) return oc_hat;

    // 高斯地图（仅用有重叠的框）
    std::vector<cv::Rect2f> overlap_boxes;
    for (int idx : overlap_idx) overlap_boxes.push_back(normed[idx]);
    cv::Mat gm = computeGaussianMap(overlap_boxes, img_h, img_w);

    // 底边 + 面积
    std::vector<float> bottoms(N), areas(N);
    for (size_t i = 0; i < N; ++i)
    {
        bottoms[i] = normed[i].y + normed[i].height;
        areas[i] = normed[i].area();
    }

    for (int i : overlap_idx)
    {
        // 找遮挡者: j 遮挡 i 的条件: bottom[j] > bottom[i] + thre_occ 且 IoU > 0
        std::vector<int> occluders;
        for (size_t j = 0; j < N; ++j)
        {
            if (static_cast<int>(j) == i) continue;
            if (bottoms[i] > bottoms[j] + thre_occ_ && iou_mat[i][j] > 0)
                occluders.push_back(static_cast<int>(j));
        }
        if (occluders.empty()) continue;
        if (areas[i] <= 0) continue;

        // 裁剪局部高斯地图
        int bx1 = std::max(0, static_cast<int>(normed[i].x));
        int by1 = std::max(0, static_cast<int>(normed[i].y));
        int bx2 = std::min(img_w, static_cast<int>(std::ceil(normed[i].x + normed[i].width)));
        int by2 = std::min(img_h, static_cast<int>(std::ceil(normed[i].y + normed[i].height)));
        if (bx2 <= bx1 || by2 <= by1) continue;

        cv::Mat local_gm = gm(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1));
        int lh = by2 - by1, lw = bx2 - bx1;

        cv::Mat occ_map = cv::Mat::zeros(lh, lw, CV_32FC1);
        for (int j : occluders)
        {
            int t = std::max(0, static_cast<int>(std::max(normed[i].y, normed[j].y)) - by1);
            int b = std::min(lh, static_cast<int>(std::min(normed[i].y + normed[i].height,
                                                           normed[j].y + normed[j].height)) - by1);
            int l = std::max(0, static_cast<int>(std::max(normed[i].x, normed[j].x)) - bx1);
            int r = std::min(lw, static_cast<int>(std::min(normed[i].x + normed[i].width,
                                                           normed[j].x + normed[j].width)) - bx1);
            if (t < b && l < r)
                occ_map(cv::Rect(l, t, r - l, b - t)).setTo(1.0f);
        }

        double sum = cv::sum(local_gm.mul(occ_map))[0];
        oc_hat[i] = static_cast<float>(sum / areas[i]);
    }

    return oc_hat;
}

// ============================================================
//  OcclusionAwareOffset
// ============================================================
OcclusionAwareOffset::OcclusionAwareOffset(float tau) : tau_(tau) {}

cv::Mat OcclusionAwareOffset::refineSpatialConsistency(
    const std::vector<cv::Rect2f>& estimations,
    const cv::Mat& iou_matrix)
{
    if (estimations.empty()) return iou_matrix.clone();

    auto oc = oam_.computeOcclusionCoefficients(estimations);
    int M = static_cast<int>(estimations.size());
    int N = iou_matrix.cols;

    cv::Mat S = iou_matrix.clone();
    for (int i = 0; i < M && i < static_cast<int>(oc.size()); ++i)
    {
        float oc_val = oc[i];
        for (int j = 0; j < N; ++j)
        {
            S.at<float>(i, j) = tau_ * (1.0f - oc_val) + (1.0f - tau_) * iou_matrix.at<float>(i, j);
        }
    }
    return S;
}

// ============================================================
//  BiasAwareMomentum
// ============================================================
BiasAwareMomentum::BiasAwareMomentum() {}

float BiasAwareMomentum::computeIoU(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float ix1 = std::max(a.x, b.x);
    float iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.width, b.x + b.width);
    float iy2 = std::min(a.y + a.height, b.y + b.height);
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
    float inter = (ix2 - ix1) * (iy2 - iy1);
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

float BiasAwareMomentum::computeMomentum(const cv::Rect2f& estimation,
                                         const cv::Rect2f& detection,
                                         float oc_last_obs)
{
    float iou = computeIoU(estimation, detection);
    return iou * (1.0f - oc_last_obs);
}

cv::Rect2f BiasAwareMomentum::refineObservation(const cv::Rect2f& estimation,
                                                const cv::Rect2f& detection,
                                                float oc_last_obs)
{
    float bam = computeMomentum(estimation, detection, oc_last_obs);
    float ex = bam * detection.x + (1.0f - bam) * estimation.x;
    float ey = bam * detection.y + (1.0f - bam) * estimation.y;
    float ew = bam * detection.width + (1.0f - bam) * estimation.width;
    float eh = bam * detection.height + (1.0f - bam) * estimation.height;
    return cv::Rect2f(ex, ey, ew, eh);
}

} // namespace bot_sort
