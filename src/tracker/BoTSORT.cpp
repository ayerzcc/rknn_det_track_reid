#include "tracker/BoTSORT.h"
#include "tracker/lapjv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace bot_sort
{

// ============================================================
//  Local helpers
// ============================================================

static byte_track::Rect<float> warpRect(const byte_track::Rect<float>& r,
                                         const cv::Matx23f& H)
{
    float x1 = r.tl_x(), y1 = r.tl_y();
    float x2 = r.br_x(), y2 = r.br_y();
    std::vector<cv::Point2f> pts = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
    std::vector<cv::Point2f> out(4);
    cv::transform(pts, out, H);
    float nx1 = std::min({out[0].x, out[1].x, out[2].x, out[3].x});
    float ny1 = std::min({out[0].y, out[1].y, out[2].y, out[3].y});
    float nx2 = std::max({out[0].x, out[1].x, out[2].x, out[3].x});
    float ny2 = std::max({out[0].y, out[1].y, out[2].y, out[3].y});
    return byte_track::Rect<float>(nx1, ny1,
                                   std::max(1.0f, nx2 - nx1),
                                   std::max(1.0f, ny2 - ny1));
}

static byte_track::Rect<float> toByteRect(const cv::Rect2f& r)
{
    return byte_track::Rect<float>(r.x, r.y, r.width, r.height);
}

static cv::Rect2f toCvRect(const byte_track::Rect<float>& r)
{
    return cv::Rect2f(r.tl_x(), r.tl_y(), r.width(), r.height());
}

// ============================================================
//  BoTSORT
// ============================================================

BoTSORT::BoTSORT(float track_high_thresh, float track_low_thresh,
                 float new_track_thresh, float match_thresh,
                 int track_buffer, GMCMethod gmc_method, int gmc_downscale)
    : track_high_thresh_(track_high_thresh),
      track_low_thresh_(track_low_thresh),
      new_track_thresh_(new_track_thresh),
      match_thresh_(match_thresh),
      max_time_lost_(static_cast<size_t>(track_buffer)),
      frame_id_(0),
      track_id_count_(0),
      gmc_(std::make_unique<GMC>(gmc_method, gmc_downscale))
{}

const std::vector<BoTSORT::STrackPtr>& BoTSORT::getTrackedStracks() const
{
    return tracked_stracks_;
}

const std::vector<BoTSORT::STrackPtr>& BoTSORT::getLostStracks() const
{
    return lost_stracks_;
}

size_t BoTSORT::getFrameId() const
{
    return frame_id_;
}

size_t BoTSORT::getMaxTimeLost() const
{
    return max_time_lost_;
}

const std::string& BoTSORT::getLastDebugJson() const
{
    return last_debug_json_;
}

bool BoTSORT::recoverTrack(const STrackPtr& track, const byte_track::Object& object)
{
    if (!track)
    {
        return false;
    }

    auto det = std::make_shared<byte_track::STrack>(object.rect, object.prob);
    track->reActivate(*det, frame_id_);

    lost_stracks_ = subStracks(lost_stracks_, std::vector<STrackPtr>{track});
    tracked_stracks_ = jointStracks(tracked_stracks_, std::vector<STrackPtr>{track});
    removeDuplicateStracks(tracked_stracks_, lost_stracks_);
    return true;
}


// ---- static helpers ----

std::vector<BoTSORT::STrackPtr> BoTSORT::jointStracks(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b)
{
    std::map<size_t, bool> exists;
    std::vector<STrackPtr> res;
    res.reserve(a.size() + b.size());
    for (const auto& t : a)
    {
        exists[t->getTrackId()] = true;
        res.push_back(t);
    }
    for (const auto& t : b)
    {
        if (exists.find(t->getTrackId()) == exists.end())
        {
            exists[t->getTrackId()] = true;
            res.push_back(t);
        }
    }
    return res;
}

std::vector<BoTSORT::STrackPtr> BoTSORT::subStracks(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b)
{
    std::map<size_t, STrackPtr> set;
    for (const auto& t : b) set[t->getTrackId()] = t;

    std::vector<STrackPtr> res;
    for (const auto& t : a)
    {
        if (set.find(t->getTrackId()) == set.end())
            res.push_back(t);
    }
    return res;
}

void BoTSORT::removeDuplicateStracks(std::vector<STrackPtr>& a,
                                      std::vector<STrackPtr>& b)
{
    if (a.empty() || b.empty()) return;

    std::vector<std::tuple<float, int, int>> pairs;
    for (int i = 0; i < static_cast<int>(a.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(b.size()); ++j)
        {
            float dist = 1.0f - a[i]->getRect().calcIoU(b[j]->getRect());
            if (dist < 0.15f)
                pairs.emplace_back(dist, i, j);
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& p1, const auto& p2) { return std::get<0>(p1) < std::get<0>(p2); });

    std::vector<int> dup_a, dup_b;
    for (const auto& [v, i, j] : pairs)
    {
        if (std::find(dup_a.begin(), dup_a.end(), i) != dup_a.end()) continue;
        if (std::find(dup_b.begin(), dup_b.end(), j) != dup_b.end()) continue;

        int ta = static_cast<int>(a[i]->getFrameId() - a[i]->getStartFrameId());
        int tb = static_cast<int>(b[j]->getFrameId() - b[j]->getStartFrameId());
        if (ta > tb) dup_b.push_back(j);
        else        dup_a.push_back(i);
    }

    std::vector<STrackPtr> new_a, new_b;
    for (int i = 0; i < static_cast<int>(a.size()); ++i)
        if (std::find(dup_a.begin(), dup_a.end(), i) == dup_a.end())
            new_a.push_back(a[i]);
    for (int j = 0; j < static_cast<int>(b.size()); ++j)
        if (std::find(dup_b.begin(), dup_b.end(), j) == dup_b.end())
            new_b.push_back(b[j]);

    a = std::move(new_a);
    b = std::move(new_b);
}

std::vector<std::vector<float>> BoTSORT::calcIouDistance(
    const std::vector<STrackPtr>& atracks,
    const std::vector<STrackPtr>& btracks)
{
    const size_t M = atracks.size(), N = btracks.size();
    std::vector<std::vector<float>> dist(M, std::vector<float>(N, 1.0f));
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            dist[i][j] = 1.0f - atracks[i]->getRect().calcIoU(btracks[j]->getRect());
    return dist;
}

void BoTSORT::linearAssignment(
    const std::vector<std::vector<float>>& cost_matrix,
    float thresh,
    std::vector<std::vector<int>>& matches,
    std::vector<int>& unmatched_rows,
    std::vector<int>& unmatched_cols)
{
    const size_t M = cost_matrix.size();
    if (M == 0) return;
    const size_t N = cost_matrix[0].size();
    if (N == 0)
    {
        for (size_t i = 0; i < M; ++i) unmatched_rows.push_back(static_cast<int>(i));
        return;
    }

    const size_t n = std::max(M, N);
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));

    double cost_max = 0.0;
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            cost[i][j] = static_cast<double>(cost_matrix[i][j]);
            cost_max = std::max(cost_max, cost[i][j]);
        }
    cost_max += 1.0;

    // pad rows
    for (size_t i = M; i < n; ++i)
        for (size_t j = 0; j < N; ++j)
            cost[i][j] = 0.0;
    // pad cols
    for (size_t i = 0; i < M; ++i)
        for (size_t j = N; j < n; ++j)
            cost[i][j] = cost_max;
    // bottom-right
    for (size_t i = M; i < n; ++i)
        for (size_t j = N; j < n; ++j)
            cost[i][j] = 0.0;

    double** cost_ptr = new double*[n];
    for (size_t i = 0; i < n; ++i) cost_ptr[i] = cost[i].data();

    int* x_c = new int[n];
    int* y_c = new int[n];
    int ret = byte_track::lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        delete[] x_c;
        delete[] y_c;
        throw std::runtime_error("lapjv_internal failed in BoTSORT::linearAssignment");
    }

    for (size_t i = 0; i < M; ++i)
    {
        if (x_c[i] >= 0 && static_cast<size_t>(x_c[i]) < N &&
            cost_matrix[i][x_c[i]] < thresh)
        {
            matches.push_back({static_cast<int>(i), x_c[i]});
        }
        else
        {
            unmatched_rows.push_back(static_cast<int>(i));
        }
    }

    std::vector<bool> col_matched(N, false);
    for (const auto& m : matches) col_matched[static_cast<size_t>(m[1])] = true;
    for (size_t j = 0; j < N; ++j)
        if (!col_matched[j]) unmatched_cols.push_back(static_cast<int>(j));

    delete[] x_c;
    delete[] y_c;
}

// ---- IoU distance with GMC compensation ----

// ---- main update ----

std::vector<BoTSORT::STrackPtr> BoTSORT::update(
    const std::vector<byte_track::Object>& objects,
    const cv::Mat& img)
{
    frame_id_++;

    std::vector<cv::Rect2f> det_rects_cv;
    for (const auto& obj : objects)
        det_rects_cv.emplace_back(obj.rect.tl_x(), obj.rect.tl_y(),
                                   obj.rect.width(), obj.rect.height());

    cv::Matx23f H = gmc_->apply(img, det_rects_cv);

    std::vector<STrackPtr> detections_high;
    std::vector<STrackPtr> detections_low;

    for (const auto& obj : objects)
    {
        auto st = std::make_shared<byte_track::STrack>(obj.rect, obj.prob, obj.feature);
        if (obj.prob >= track_high_thresh_)
            detections_high.push_back(st);
        else if (obj.prob >= track_low_thresh_)
            detections_low.push_back(st);
    }

    for (auto& t : tracked_stracks_) t->applyGmc(H);
    for (auto& t : lost_stracks_)    t->applyGmc(H);
    for (auto& t : tracked_stracks_) t->predict();
    for (auto& t : lost_stracks_)    t->predict();

    std::vector<STrackPtr> activated_stracks;
    std::vector<STrackPtr> refind_stracks;
    std::vector<STrackPtr> current_lost_stracks;
    std::vector<STrackPtr> current_removed_stracks;

    std::vector<STrackPtr> tracked_stracks;
    std::vector<STrackPtr> unconfirmed;
    for (const auto& t : tracked_stracks_)
    {
        if (t->isActivated()) tracked_stracks.push_back(t);
        else                  unconfirmed.push_back(t);
    }

    std::vector<STrackPtr> pool = jointStracks(tracked_stracks, lost_stracks_);

    std::vector<std::vector<int>> matches1;
    std::vector<int> um_tracks1, um_dets1;
    std::vector<std::vector<float>> stage1_iou_dist;
    if (!pool.empty() && !detections_high.empty())
    {
        stage1_iou_dist = calcIouDistance(pool, detections_high);
        linearAssignment(stage1_iou_dist, match_thresh_, matches1, um_tracks1, um_dets1);
    }
    else if (pool.empty())
    {
        for (size_t j = 0; j < detections_high.size(); ++j)
            um_dets1.push_back(static_cast<int>(j));
    }
    else
    {
        for (size_t i = 0; i < pool.size(); ++i)
            um_tracks1.push_back(static_cast<int>(i));
    }

    for (const auto& match : matches1)
    {
        auto& track = pool[static_cast<size_t>(match[0])];
        auto& det = detections_high[static_cast<size_t>(match[1])];
        if (track->getSTrackState() == byte_track::STrackState::Tracked)
        {
            track->update(*det, frame_id_);
            activated_stracks.push_back(track);
        }
        else
        {
            track->reActivate(*det, frame_id_);
            refind_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> r_tracked_stracks;
    for (int idx : um_tracks1)
    {
        const auto& track = pool[static_cast<size_t>(idx)];
        if (track->getSTrackState() == byte_track::STrackState::Tracked)
            r_tracked_stracks.push_back(track);
    }

    std::vector<std::vector<int>> matches2;
    std::vector<int> um_tracks2, um_dets2;
    if (!r_tracked_stracks.empty() && !detections_low.empty())
    {
        auto dist2 = calcIouDistance(r_tracked_stracks, detections_low);
        linearAssignment(dist2, 0.5f, matches2, um_tracks2, um_dets2);
    }
    else if (!r_tracked_stracks.empty())
    {
        for (size_t i = 0; i < r_tracked_stracks.size(); ++i)
            um_tracks2.push_back(static_cast<int>(i));
    }

    for (const auto& match : matches2)
    {
        auto& track = r_tracked_stracks[static_cast<size_t>(match[0])];
        auto& det = detections_low[static_cast<size_t>(match[1])];
        if (track->getSTrackState() == byte_track::STrackState::Tracked)
        {
            track->update(*det, frame_id_);
            activated_stracks.push_back(track);
        }
        else
        {
            track->reActivate(*det, frame_id_);
            refind_stracks.push_back(track);
        }
    }

    for (int idx : um_tracks2)
    {
        auto& track = r_tracked_stracks[static_cast<size_t>(idx)];
        if (track->getSTrackState() != byte_track::STrackState::Lost)
        {
            track->markAsLost();
            current_lost_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> detections_remain;
    for (int idx : um_dets1)
        detections_remain.push_back(detections_high[static_cast<size_t>(idx)]);

    std::vector<std::vector<int>> matches3;
    std::vector<int> um_unconfirmed, um_dets3;
    if (!unconfirmed.empty() && !detections_remain.empty())
    {
        auto dist3 = calcIouDistance(unconfirmed, detections_remain);
        linearAssignment(dist3, 0.7f, matches3, um_unconfirmed, um_dets3);
    }
    else if (!unconfirmed.empty())
    {
        for (size_t i = 0; i < unconfirmed.size(); ++i)
            um_unconfirmed.push_back(static_cast<int>(i));
    }
    else
    {
        for (size_t i = 0; i < detections_remain.size(); ++i)
            um_dets3.push_back(static_cast<int>(i));
    }

    for (const auto& match : matches3)
    {
        auto& track = unconfirmed[static_cast<size_t>(match[0])];
        auto& det = detections_remain[static_cast<size_t>(match[1])];
        track->update(*det, frame_id_);
        activated_stracks.push_back(track);
    }

    for (int idx : um_unconfirmed)
    {
        auto& track = unconfirmed[static_cast<size_t>(idx)];
        track->markAsRemoved();
        current_removed_stracks.push_back(track);
    }

    for (int idx : um_dets3)
    {
        auto& det = detections_remain[static_cast<size_t>(idx)];
        if (det->getScore() < new_track_thresh_)
            continue;
        det->activate(frame_id_, ++track_id_count_);
        activated_stracks.push_back(det);
    }

    for (const auto& track : lost_stracks_)
    {
        if ((frame_id_ - track->getFrameId()) > max_time_lost_)
        {
            track->markAsRemoved();
            current_removed_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> tracked_only;
    for (const auto& t : tracked_stracks_)
    {
        if (t->getSTrackState() == byte_track::STrackState::Tracked)
            tracked_only.push_back(t);
    }
    tracked_stracks_ = jointStracks(tracked_only, activated_stracks);
    tracked_stracks_ = jointStracks(tracked_stracks_, refind_stracks);
    lost_stracks_ = subStracks(lost_stracks_, tracked_stracks_);
    lost_stracks_ = jointStracks(lost_stracks_, current_lost_stracks);
    lost_stracks_ = subStracks(lost_stracks_, current_removed_stracks);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    removeDuplicateStracks(tracked_stracks_, lost_stracks_);

    std::vector<STrackPtr> output;
    for (const auto& t : tracked_stracks_)
    {
        if (t->isActivated() && t->getSTrackState() == byte_track::STrackState::Tracked)
            output.push_back(t);
    }

    std::ostringstream dbg;
    dbg << "{\"frame\":" << frame_id_
        << ",\"gmc\":[" << H(0,0) << "," << H(0,1) << "," << H(0,2) << "," << H(1,0) << "," << H(1,1) << "," << H(1,2) << "]"
        << ",\"gmc_debug\":" << gmc_->getLastDebugJson()
        << ",\"pool_ids\":[";
    for (size_t i = 0; i < pool.size(); ++i) {
        dbg << pool[i]->getTrackId();
        if (i + 1 < pool.size()) dbg << ",";
    }
    dbg << "],\"warped_boxes\":[";
    for (size_t i = 0; i < pool.size(); ++i) {
        auto wr = warpRect(pool[i]->getRect(), H);
        dbg << "[" << wr.tl_x() << "," << wr.tl_y() << "," << wr.br_x() << "," << wr.br_y() << "]";
        if (i + 1 < pool.size()) dbg << ",";
    }
    dbg << "],\"det_boxes\":[";
    for (size_t i = 0; i < detections_high.size(); ++i) {
        const auto& r = detections_high[i]->getRect();
        dbg << "[" << r.tl_x() << "," << r.tl_y() << "," << r.br_x() << "," << r.br_y() << "]";
        if (i + 1 < detections_high.size()) dbg << ",";
    }
    dbg << "],\"high_det_scores\":[";
    for (size_t i = 0; i < detections_high.size(); ++i) {
        dbg << detections_high[i]->getScore();
        if (i + 1 < detections_high.size()) dbg << ",";
    }
    dbg << "],\"matches1\":[";
    for (size_t i = 0; i < matches1.size(); ++i) {
        dbg << "[" << matches1[i][0] << "," << matches1[i][1] << "]";
        if (i + 1 < matches1.size()) dbg << ",";
    }
    dbg << "],\"um_tracks1\":[";
    for (size_t i = 0; i < um_tracks1.size(); ++i) {
        dbg << um_tracks1[i];
        if (i + 1 < um_tracks1.size()) dbg << ",";
    }
    dbg << "],\"um_dets1\":[";
    for (size_t i = 0; i < um_dets1.size(); ++i) {
        dbg << um_dets1[i];
        if (i + 1 < um_dets1.size()) dbg << ",";
    }
    dbg << "],\"iou_dist1\":[";
    for (size_t i = 0; i < stage1_iou_dist.size(); ++i) {
        dbg << "[";
        for (size_t j = 0; j < stage1_iou_dist[i].size(); ++j) {
            dbg << stage1_iou_dist[i][j];
            if (j + 1 < stage1_iou_dist[i].size()) dbg << ",";
        }
        dbg << "]";
        if (i + 1 < stage1_iou_dist.size()) dbg << ",";
    }
    dbg << "],\"matches2\":[";
    for (size_t i = 0; i < matches2.size(); ++i) {
        dbg << "[" << matches2[i][0] << "," << matches2[i][1] << "]";
        if (i + 1 < matches2.size()) dbg << ",";
    }
    dbg << "],\"matches3\":[";
    for (size_t i = 0; i < matches3.size(); ++i) {
        dbg << "[" << matches3[i][0] << "," << matches3[i][1] << "]";
        if (i + 1 < matches3.size()) dbg << ",";
    }
    dbg << "],\"tracked_ids\":[";
    for (size_t i = 0; i < output.size(); ++i) {
        dbg << output[i]->getTrackId();
        if (i + 1 < output.size()) dbg << ",";
    }
    dbg << "],\"lost_ids\":[";
    for (size_t i = 0; i < lost_stracks_.size(); ++i) {
        dbg << lost_stracks_[i]->getTrackId();
        if (i + 1 < lost_stracks_.size()) dbg << ",";
    }
    dbg << "]}";
    last_debug_json_ = dbg.str();

    return output;
}

// ============================================================
//  OABoTSORT
// ============================================================

OABoTSORT::OABoTSORT(float track_high_thresh, float track_low_thresh,
                     float new_track_thresh, float match_thresh,
                     int track_buffer, GMCMethod gmc_method, int gmc_downscale,
                     bool use_oao, bool use_bam,
                     float oa_tau, float oa_k_x, float oa_k_y)
    : BoTSORT(track_high_thresh, track_low_thresh, new_track_thresh, match_thresh,
              track_buffer, gmc_method, gmc_downscale),
      use_oao_(use_oao),
      use_bam_(use_bam),
      oam_(std::make_unique<OcclusionAwareModule>(oa_k_x, oa_k_y)),
      oao_(std::make_unique<OcclusionAwareOffset>(oa_tau)),
      bam_(std::make_unique<BiasAwareMomentum>())
{}

std::vector<OABoTSORT::STrackPtr> OABoTSORT::update(
    const std::vector<byte_track::Object>& objects,
    const cv::Mat& img)
{
    frame_id_++;

    if (!img.empty())
    {
        img_h_ = img.rows;
        img_w_ = img.cols;
    }

    // Build detection rects for GMC
    std::vector<cv::Rect2f> det_rects_cv;
    for (const auto& obj : objects)
        det_rects_cv.emplace_back(obj.rect.tl_x(), obj.rect.tl_y(),
                                   obj.rect.width(), obj.rect.height());

    cv::Matx23f H = gmc_->apply(img, det_rects_cv);

    std::vector<STrackPtr> detections_high;
    std::vector<STrackPtr> detections_low;
    for (const auto& obj : objects)
    {
        auto st = std::make_shared<byte_track::STrack>(obj.rect, obj.prob, obj.feature);
        if (obj.prob >= track_high_thresh_)
            detections_high.push_back(st);
        else if (obj.prob >= track_low_thresh_)
            detections_low.push_back(st);
    }

    for (auto& t : tracked_stracks_) t->applyGmc(H);
    for (auto& t : lost_stracks_)    t->applyGmc(H);
    for (auto& t : tracked_stracks_) t->predict();
    for (auto& t : lost_stracks_)    t->predict();

    std::vector<STrackPtr> activated_stracks;
    std::vector<STrackPtr> refind_stracks;
    std::vector<STrackPtr> current_lost_stracks;
    std::vector<STrackPtr> current_removed_stracks;

    std::vector<STrackPtr> tracked_stracks;
    std::vector<STrackPtr> unconfirmed;
    for (const auto& t : tracked_stracks_)
    {
        if (t->isActivated()) tracked_stracks.push_back(t);
        else                  unconfirmed.push_back(t);
    }

    std::vector<STrackPtr> pool = jointStracks(tracked_stracks, lost_stracks_);
    std::vector<std::vector<int>> matches1;
    std::vector<int> um_tracks1, um_dets1;
    std::vector<std::vector<float>> stage1_raw_iou_dist;
    std::vector<std::vector<float>> stage1_refined_dist;
    std::vector<float> stage2_oc_coeffs;

    if (!pool.empty() && !detections_high.empty())
    {
        auto iou_dist = calcIouDistance(pool, detections_high);
        stage1_raw_iou_dist = iou_dist;

        if (use_oao_)
        {
            std::vector<cv::Rect2f> estimations;
            for (const auto& t : pool)
                estimations.push_back(toCvRect(t->getRect()));

            const size_t M = pool.size(), N = detections_high.size();
            cv::Mat iou_mat(static_cast<int>(M), static_cast<int>(N), CV_32FC1);
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    iou_mat.at<float>(static_cast<int>(i), static_cast<int>(j)) = 1.0f - iou_dist[i][j];

            cv::Mat refined = oao_->refineSpatialConsistency(estimations, iou_mat);

            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    iou_dist[i][j] = 1.0f - refined.at<float>(static_cast<int>(i), static_cast<int>(j));
        }
        stage1_refined_dist = iou_dist;

        linearAssignment(iou_dist, match_thresh_, matches1, um_tracks1, um_dets1);
    }
    else if (pool.empty())
    {
        for (size_t j = 0; j < detections_high.size(); ++j)
            um_dets1.push_back(static_cast<int>(j));
    }
    else
    {
        for (size_t i = 0; i < pool.size(); ++i)
            um_tracks1.push_back(static_cast<int>(i));
    }

    for (const auto& m_matches1 : matches1)
    {
        auto& track = pool[static_cast<size_t>(m_matches1[0])];
        auto& det = detections_high[static_cast<size_t>(m_matches1[1])];
        if (track->getSTrackState() == byte_track::STrackState::Tracked)
        {
            track->update(*det, frame_id_);
            activated_stracks.push_back(track);
        }
        else
        {
            track->reActivate(*det, frame_id_);
            refind_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> r_tracked_stracks;
    for (int idx : um_tracks1)
    {
        const auto& track = pool[static_cast<size_t>(idx)];
        if (track->getSTrackState() == byte_track::STrackState::Tracked)
            r_tracked_stracks.push_back(track);
    }

    std::vector<std::vector<int>> matches2;
    std::vector<int> um_tracks2, um_dets2;
    if (!r_tracked_stracks.empty() && !detections_low.empty())
    {
        auto dist2 = calcIouDistance(r_tracked_stracks, detections_low);
        linearAssignment(dist2, 0.5f, matches2, um_tracks2, um_dets2);

        for (const auto& match : matches2)
        {
            auto& track = r_tracked_stracks[static_cast<size_t>(match[0])];
            auto& det = detections_low[static_cast<size_t>(match[1])];
            if (track->getSTrackState() == byte_track::STrackState::Tracked)
            {
                if (use_bam_)
                {
                    cv::Rect2f est = toCvRect(track->getRect());
                    cv::Rect2f det_cv = toCvRect(det->getRect());
                    cv::Rect2f refined = bam_->refineObservation(est, det_cv, track->getOcclusionCoeff());
                    auto refined_st = std::make_shared<byte_track::STrack>(toByteRect(refined), det->getScore());
                    track->update(*refined_st, frame_id_);
                }
                else
                {
                    track->update(*det, frame_id_);
                }
                activated_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_);
                refind_stracks.push_back(track);
            }
        }
    }
    else if (!r_tracked_stracks.empty())
    {
        for (size_t i = 0; i < r_tracked_stracks.size(); ++i)
            um_tracks2.push_back(static_cast<int>(i));
    }

    for (int idx : um_tracks2)
    {
        auto& track = r_tracked_stracks[static_cast<size_t>(idx)];
        if (track->getSTrackState() != byte_track::STrackState::Lost)
        {
            track->markAsLost();
            current_lost_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> detections_remain;
    for (int idx : um_dets1)
        detections_remain.push_back(detections_high[static_cast<size_t>(idx)]);

    std::vector<std::vector<int>> matches3;
    std::vector<int> um_unconfirmed, um_dets3;
    if (!unconfirmed.empty() && !detections_remain.empty())
    {
        auto dist3 = calcIouDistance(unconfirmed, detections_remain);
        linearAssignment(dist3, 0.7f, matches3, um_unconfirmed, um_dets3);
    }
    else if (!unconfirmed.empty())
    {
        for (size_t i = 0; i < unconfirmed.size(); ++i)
            um_unconfirmed.push_back(static_cast<int>(i));
    }
    else
    {
        for (size_t i = 0; i < detections_remain.size(); ++i)
            um_dets3.push_back(static_cast<int>(i));
    }

    for (const auto& match : matches3)
    {
        auto& track = unconfirmed[static_cast<size_t>(match[0])];
        auto& det = detections_remain[static_cast<size_t>(match[1])];
        track->update(*det, frame_id_);
        activated_stracks.push_back(track);
    }

    for (int idx : um_unconfirmed)
    {
        auto& track = unconfirmed[static_cast<size_t>(idx)];
        track->markAsRemoved();
        current_removed_stracks.push_back(track);
    }

    for (int idx : um_dets3)
    {
        auto& det = detections_remain[static_cast<size_t>(idx)];
        if (det->getScore() < new_track_thresh_)
            continue;
        det->activate(frame_id_, ++track_id_count_);
        activated_stracks.push_back(det);
    }

    for (const auto& track : lost_stracks_)
    {
        if ((frame_id_ - track->getFrameId()) > max_time_lost_)
        {
            track->markAsRemoved();
            current_removed_stracks.push_back(track);
        }
    }

    std::vector<STrackPtr> tracked_only;
    for (const auto& t : tracked_stracks_)
    {
        if (t->getSTrackState() == byte_track::STrackState::Tracked)
            tracked_only.push_back(t);
    }
    tracked_stracks_ = jointStracks(tracked_only, activated_stracks);
    tracked_stracks_ = jointStracks(tracked_stracks_, refind_stracks);
    lost_stracks_ = subStracks(lost_stracks_, tracked_stracks_);
    lost_stracks_ = jointStracks(lost_stracks_, current_lost_stracks);
    lost_stracks_ = subStracks(lost_stracks_, current_removed_stracks);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    if (use_bam_ && !activated_stracks.empty() && img_h_ > 0 && img_w_ > 0)
    {
        std::vector<cv::Rect2f> active_bboxes;
        active_bboxes.reserve(activated_stracks.size());
        for (const auto& t : activated_stracks)
            active_bboxes.push_back(toCvRect(t->getRect()));
        const auto oc_coeffs = oam_->computeOcclusionCoefficients(active_bboxes, img_h_, img_w_);
        stage2_oc_coeffs = oc_coeffs;
        for (size_t i = 0; i < activated_stracks.size() && i < oc_coeffs.size(); ++i)
            activated_stracks[i]->setOcclusionCoeff(oc_coeffs[i]);
    }

    removeDuplicateStracks(tracked_stracks_, lost_stracks_);

    std::vector<STrackPtr> output;
    for (const auto& t : tracked_stracks_)
        if (t->isActivated() && t->getSTrackState() == byte_track::STrackState::Tracked)
            output.push_back(t);

    std::ostringstream dbg;
    dbg << "{\"frame\":" << frame_id_
        << ",\"gmc\":[" << H(0,0) << "," << H(0,1) << "," << H(0,2) << "," << H(1,0) << "," << H(1,1) << "," << H(1,2) << "]"
        << ",\"gmc_debug\":" << gmc_->getLastDebugJson()
        << ",\"pool_ids\":[";
    for (size_t i = 0; i < pool.size(); ++i) {
        dbg << pool[i]->getTrackId();
        if (i + 1 < pool.size()) dbg << ",";
    }
    dbg << "],\"matches1\":[";
    for (size_t i = 0; i < matches1.size(); ++i) {
        dbg << "[" << matches1[i][0] << "," << matches1[i][1] << "]";
        if (i + 1 < matches1.size()) dbg << ",";
    }
    dbg << "],\"um_tracks1\":[";
    for (size_t i = 0; i < um_tracks1.size(); ++i) {
        dbg << um_tracks1[i];
        if (i + 1 < um_tracks1.size()) dbg << ",";
    }
    dbg << "],\"um_dets1\":[";
    for (size_t i = 0; i < um_dets1.size(); ++i) {
        dbg << um_dets1[i];
        if (i + 1 < um_dets1.size()) dbg << ",";
    }
    dbg << "],\"iou_dist1_raw\":[";
    for (size_t i = 0; i < stage1_raw_iou_dist.size(); ++i) {
        dbg << "[";
        for (size_t j = 0; j < stage1_raw_iou_dist[i].size(); ++j) {
            dbg << stage1_raw_iou_dist[i][j];
            if (j + 1 < stage1_raw_iou_dist[i].size()) dbg << ",";
        }
        dbg << "]";
        if (i + 1 < stage1_raw_iou_dist.size()) dbg << ",";
    }
    dbg << "],\"iou_dist1_refined\":[";
    for (size_t i = 0; i < stage1_refined_dist.size(); ++i) {
        dbg << "[";
        for (size_t j = 0; j < stage1_refined_dist[i].size(); ++j) {
            dbg << stage1_refined_dist[i][j];
            if (j + 1 < stage1_refined_dist[i].size()) dbg << ",";
        }
        dbg << "]";
        if (i + 1 < stage1_refined_dist.size()) dbg << ",";
    }
    dbg << "],\"oc_coeffs\":[";
    for (size_t i = 0; i < stage2_oc_coeffs.size(); ++i) {
        dbg << stage2_oc_coeffs[i];
        if (i + 1 < stage2_oc_coeffs.size()) dbg << ",";
    }
    dbg << "],\"matches2\":[";
    for (size_t i = 0; i < matches2.size(); ++i) {
        dbg << "[" << matches2[i][0] << "," << matches2[i][1] << "]";
        if (i + 1 < matches2.size()) dbg << ",";
    }
    dbg << "],\"tracked_ids\":[";
    for (size_t i = 0; i < output.size(); ++i) {
        dbg << output[i]->getTrackId();
        if (i + 1 < output.size()) dbg << ",";
    }
    dbg << "],\"lost_ids\":[";
    for (size_t i = 0; i < lost_stracks_.size(); ++i) {
        dbg << lost_stracks_[i]->getTrackId();
        if (i + 1 < lost_stracks_.size()) dbg << ",";
    }
    dbg << "]}";
    last_debug_json_ = dbg.str();

    return output;
}

} // namespace bot_sort
