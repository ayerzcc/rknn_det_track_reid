#include "tracker/OAByteTracker.h"

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

byte_track::OAByteTracker::OAByteTracker(const int& frame_rate,
                                         const int& track_buffer,
                                         const float& track_thresh,
                                         const float& high_thresh,
                                         const float& new_track_thresh,
                                         const float& match_thresh,
                                         bool use_oao,
                                         bool use_bam,
                                         float tau,
                                         float k_x,
                                         float k_y)
    : track_thresh_(track_thresh),
      high_thresh_(high_thresh),
      new_track_thresh_(new_track_thresh),
      match_thresh_(match_thresh),
      max_time_lost_(static_cast<size_t>(frame_rate / 30.0 * track_buffer)),
      frame_id_(0),
      track_id_count_(0),
      use_oao_(use_oao),
      use_bam_(use_bam),
      tau_(tau),
      oam_(std::make_unique<bot_sort::OcclusionAwareModule>(k_x, k_y)),
      oao_(use_oao ? std::make_unique<bot_sort::OcclusionAwareOffset>(tau) : nullptr),
      bam_(use_bam ? std::make_unique<bot_sort::BiasAwareMomentum>() : nullptr)
{
}

byte_track::OAByteTracker::~OAByteTracker()
{
    tracked_stracks_.clear();
    lost_stracks_.clear();
    removed_stracks_.clear();
}

const std::vector<byte_track::OAByteTracker::STrackPtr>& byte_track::OAByteTracker::getTrackedStracks() const
{
    return tracked_stracks_;
}

const std::vector<byte_track::OAByteTracker::STrackPtr>& byte_track::OAByteTracker::getLostStracks() const
{
    return lost_stracks_;
}

size_t byte_track::OAByteTracker::getFrameId() const
{
    return frame_id_;
}

size_t byte_track::OAByteTracker::getMaxTimeLost() const
{
    return max_time_lost_;
}

bool byte_track::OAByteTracker::recoverTrack(const STrackPtr& track, const Object& object)
{
    if (!track)
    {
        return false;
    }

    auto det = std::make_shared<STrack>(object.rect, object.prob, object.feature);
    track->reActivate(*det, frame_id_);

    lost_stracks_ = subStracks(lost_stracks_, std::vector<STrackPtr>{track});
    tracked_stracks_ = jointStracks(tracked_stracks_, std::vector<STrackPtr>{track});
    std::vector<STrackPtr> tracked_out, lost_out;
    removeDuplicateStracks(tracked_stracks_, lost_stracks_, tracked_out, lost_out);
    tracked_stracks_ = std::move(tracked_out);
    lost_stracks_ = std::move(lost_out);
    return true;
}

std::vector<byte_track::OAByteTracker::STrackPtr> byte_track::OAByteTracker::update(const std::vector<Object>& objects,
                                                                                     const cv::Mat& img)
{
    frame_id_++;
    std::vector<STrackPtr> activated_stracks;
    std::vector<STrackPtr> refind_stracks;
    std::vector<STrackPtr> current_lost_stracks;
    std::vector<STrackPtr> current_removed_stracks;

    if (!img.empty())
    {
        img_h_ = img.rows;
        img_w_ = img.cols;
    }

    std::vector<STrackPtr> det_stracks;
    std::vector<STrackPtr> det_low_stracks;
    for (const auto& object : objects)
    {
        auto strack = std::make_shared<STrack>(object.rect, object.prob, object.feature);
        if (object.prob >= track_thresh_)
            det_stracks.push_back(strack);
        else
            det_low_stracks.push_back(strack);
    }

    std::vector<STrackPtr> active_stracks;
    std::vector<STrackPtr> non_active_stracks;
    for (const auto& tracked_strack : tracked_stracks_)
    {
        if (!tracked_strack->isActivated())
            non_active_stracks.push_back(tracked_strack);
        else
            active_stracks.push_back(tracked_strack);
    }

    auto strack_pool = jointStracks(active_stracks, lost_stracks_);
    for (auto& strack : strack_pool)
        strack->predict();

    std::vector<STrackPtr> remain_tracked_stracks;
    std::vector<STrackPtr> remain_det_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_detection_idx, unmatch_track_idx;

        auto dists = calcIouDistance(strack_pool, det_stracks);
        if (use_oao_ && !strack_pool.empty() && !det_stracks.empty())
        {
            std::vector<cv::Rect2f> estimations;
            for (const auto& s : strack_pool)
            {
                const auto& r = s->getRect();
                estimations.emplace_back(r.tl_x(), r.tl_y(), r.width(), r.height());
            }
            cv::Mat iou_mat(static_cast<int>(strack_pool.size()), static_cast<int>(det_stracks.size()), CV_32FC1);
            for (size_t i = 0; i < dists.size(); ++i)
                for (size_t j = 0; j < dists[i].size(); ++j)
                    iou_mat.at<float>(static_cast<int>(i), static_cast<int>(j)) = 1.0f - dists[i][j];
            cv::Mat refined = oao_->refineSpatialConsistency(estimations, iou_mat);
            for (size_t i = 0; i < dists.size(); ++i)
                for (size_t j = 0; j < dists[i].size(); ++j)
                    dists[i][j] = 1.0f - refined.at<float>(static_cast<int>(i), static_cast<int>(j));
        }

        linearAssignment(dists, strack_pool.size(), det_stracks.size(), match_thresh_,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            const auto track = strack_pool[match_idx[0]];
            const auto det = det_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
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

        for (const auto& unmatch_idx : unmatch_detection_idx)
            remain_det_stracks.push_back(det_stracks[unmatch_idx]);

        for (const auto& unmatch_idx : unmatch_track_idx)
            if (strack_pool[unmatch_idx]->getSTrackState() == STrackState::Tracked)
                remain_tracked_stracks.push_back(strack_pool[unmatch_idx]);
    }

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_track_idx, unmatch_detection_idx;

        auto dists = calcIouDistance(remain_tracked_stracks, det_low_stracks);
        linearAssignment(dists, remain_tracked_stracks.size(), det_low_stracks.size(), 0.5f,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            const auto track = remain_tracked_stracks[match_idx[0]];
            const auto det = det_low_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                if (use_bam_)
                {
                    cv::Rect2f estimation(track->getRect().tl_x(), track->getRect().tl_y(),
                                          track->getRect().width(), track->getRect().height());
                    cv::Rect2f detection(det->getRect().tl_x(), det->getRect().tl_y(),
                                         det->getRect().width(), det->getRect().height());
                    cv::Rect2f refined = bam_->refineObservation(estimation, detection, track->getOcclusionCoeff());
                    auto det_refined = std::make_shared<STrack>(
                        Rect<float>(refined.x, refined.y, refined.width, refined.height),
                        det->getScore(),
                        det->getCurrentFeature());
                    track->update(*det_refined, frame_id_);
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

        for (const auto& unmatch_track : unmatch_track_idx)
        {
            const auto track = remain_tracked_stracks[unmatch_track];
            if (track->getSTrackState() != STrackState::Lost)
            {
                track->markAsLost();
                current_lost_stracks.push_back(track);
            }
        }
    }

    {
        std::vector<int> unmatch_detection_idx;
        std::vector<int> unmatch_unconfirmed_idx;
        std::vector<std::vector<int>> matches_idx;

        auto dists = calcIouDistance(non_active_stracks, remain_det_stracks);
        linearAssignment(dists, non_active_stracks.size(), remain_det_stracks.size(), 0.7f,
                         matches_idx, unmatch_unconfirmed_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            non_active_stracks[match_idx[0]]->update(*remain_det_stracks[match_idx[1]], frame_id_);
            activated_stracks.push_back(non_active_stracks[match_idx[0]]);
        }

        for (const auto& unmatch_idx : unmatch_unconfirmed_idx)
        {
            const auto track = non_active_stracks[unmatch_idx];
            track->markAsRemoved();
            current_removed_stracks.push_back(track);
        }

        for (const auto& unmatch_idx : unmatch_detection_idx)
        {
            const auto track = remain_det_stracks[unmatch_idx];
            if (track->getScore() < new_track_thresh_)
                continue;
            track_id_count_++;
            track->activate(frame_id_, track_id_count_);
            activated_stracks.push_back(track);
        }
    }

    for (const auto& lost_strack : lost_stracks_)
    {
        if (frame_id_ - lost_strack->getFrameId() > max_time_lost_)
        {
            lost_strack->markAsRemoved();
            current_removed_stracks.push_back(lost_strack);
        }
    }

    tracked_stracks_ = jointStracks(activated_stracks, refind_stracks);
    lost_stracks_ = subStracks(jointStracks(subStracks(lost_stracks_, tracked_stracks_), current_lost_stracks), removed_stracks_);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    std::vector<STrackPtr> tracked_out, lost_out;
    removeDuplicateStracks(tracked_stracks_, lost_stracks_, tracked_out, lost_out);
    tracked_stracks_ = tracked_out;
    lost_stracks_ = lost_out;

    if (use_bam_ && !activated_stracks.empty() && img_h_ > 0 && img_w_ > 0)
    {
        std::vector<cv::Rect2f> active_bboxes;
        active_bboxes.reserve(activated_stracks.size());
        for (const auto& t : activated_stracks)
        {
            const auto& r = t->getRect();
            active_bboxes.emplace_back(r.tl_x(), r.tl_y(), r.width(), r.height());
        }
        const auto oc_coeffs = oam_->computeOcclusionCoefficients(active_bboxes, img_h_, img_w_);
        for (size_t i = 0; i < activated_stracks.size() && i < oc_coeffs.size(); ++i)
        {
            activated_stracks[i]->setOcclusionCoeff(oc_coeffs[i]);
        }
    }

    std::vector<STrackPtr> output_stracks;
    for (const auto& track : tracked_stracks_)
        if (track->isActivated())
            output_stracks.push_back(track);

    removed_stracks_.clear();
    return output_stracks;
}

std::vector<byte_track::OAByteTracker::STrackPtr> byte_track::OAByteTracker::jointStracks(const std::vector<STrackPtr>& a_tlist,
                                                                                           const std::vector<STrackPtr>& b_tlist) const
{
    std::map<int, int> exists;
    std::vector<STrackPtr> res;
    for (const auto& t : a_tlist)
    {
        exists.emplace(t->getTrackId(), 1);
        res.push_back(t);
    }
    for (const auto& t : b_tlist)
    {
        const int tid = static_cast<int>(t->getTrackId());
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(t);
        }
    }
    return res;
}

std::vector<byte_track::OAByteTracker::STrackPtr> byte_track::OAByteTracker::subStracks(const std::vector<STrackPtr>& a_tlist,
                                                                                         const std::vector<STrackPtr>& b_tlist) const
{
    std::map<int, STrackPtr> stracks;
    for (const auto& t : a_tlist)
        stracks.emplace(static_cast<int>(t->getTrackId()), t);
    for (const auto& t : b_tlist)
    {
        const int tid = static_cast<int>(t->getTrackId());
        if (stracks.count(tid) != 0)
            stracks.erase(tid);
    }

    std::vector<STrackPtr> res;
    for (auto& kv : stracks)
        res.push_back(kv.second);
    return res;
}

void byte_track::OAByteTracker::removeDuplicateStracks(const std::vector<STrackPtr>& a_stracks,
                                                       const std::vector<STrackPtr>& b_stracks,
                                                       std::vector<STrackPtr>& a_res,
                                                       std::vector<STrackPtr>& b_res) const
{
    const auto ious = calcIouDistance(a_stracks, b_stracks);
    std::vector<std::pair<size_t, size_t>> overlapping_combinations;
    for (size_t i = 0; i < ious.size(); ++i)
        for (size_t j = 0; j < ious[i].size(); ++j)
            if (ious[i][j] < 0.15f)
                overlapping_combinations.emplace_back(i, j);

    std::vector<bool> a_overlapping(a_stracks.size(), false), b_overlapping(b_stracks.size(), false);
    for (const auto& [a_idx, b_idx] : overlapping_combinations)
    {
        const int timep = static_cast<int>(a_stracks[a_idx]->getFrameId() - a_stracks[a_idx]->getStartFrameId());
        const int timeq = static_cast<int>(b_stracks[b_idx]->getFrameId() - b_stracks[b_idx]->getStartFrameId());
        if (timep > timeq)
            b_overlapping[b_idx] = true;
        else
            a_overlapping[a_idx] = true;
    }

    for (size_t ai = 0; ai < a_stracks.size(); ++ai)
        if (!a_overlapping[ai]) a_res.push_back(a_stracks[ai]);
    for (size_t bi = 0; bi < b_stracks.size(); ++bi)
        if (!b_overlapping[bi]) b_res.push_back(b_stracks[bi]);
}

void byte_track::OAByteTracker::linearAssignment(const std::vector<std::vector<float>>& cost_matrix,
                                                 const int& cost_matrix_size,
                                                 const int& cost_matrix_size_size,
                                                 const float& thresh,
                                                 std::vector<std::vector<int>>& matches,
                                                 std::vector<int>& a_unmatched,
                                                 std::vector<int>& b_unmatched) const
{
    const size_t M = cost_matrix.size();
    if (M == 0)
    {
        for (int i = 0; i < cost_matrix_size; ++i) a_unmatched.push_back(i);
        for (int i = 0; i < cost_matrix_size_size; ++i) b_unmatched.push_back(i);
        return;
    }
    const size_t N = cost_matrix[0].size();
    if (N == 0)
    {
        for (size_t i = 0; i < M; ++i)
            a_unmatched.push_back(static_cast<int>(i));
        return;
    }

    const size_t n = std::max(M, N);
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));

    double cost_max = 0.0;
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            cost[i][j] = static_cast<double>(cost_matrix[i][j]);
            cost_max = std::max(cost_max, cost[i][j]);
        }
    }
    cost_max += 1.0;

    for (size_t i = M; i < n; ++i)
        for (size_t j = 0; j < N; ++j)
            cost[i][j] = 0.0;
    for (size_t i = 0; i < M; ++i)
        for (size_t j = N; j < n; ++j)
            cost[i][j] = cost_max;
    for (size_t i = M; i < n; ++i)
        for (size_t j = N; j < n; ++j)
            cost[i][j] = 0.0;

    std::vector<double*> cost_ptr(n);
    for (size_t i = 0; i < n; ++i)
        cost_ptr[i] = cost[i].data();

    std::vector<int> x_c(n, -1);
    std::vector<int> y_c(n, -1);
    const int ret = lapjv_internal(static_cast<int>(n), cost_ptr.data(), x_c.data(), y_c.data());
    if (ret != 0)
        throw std::runtime_error("lapjv_internal failed in OAByteTracker::linearAssignment");

    for (size_t i = 0; i < M; ++i)
    {
        if (x_c[i] >= 0 && static_cast<size_t>(x_c[i]) < N &&
            cost_matrix[i][x_c[i]] < thresh)
        {
            matches.push_back({static_cast<int>(i), x_c[i]});
        }
        else
        {
            a_unmatched.push_back(static_cast<int>(i));
        }
    }

    std::vector<bool> col_matched(N, false);
    for (const auto& match : matches)
        col_matched[static_cast<size_t>(match[1])] = true;
    for (size_t j = 0; j < N; ++j)
        if (!col_matched[j])
            b_unmatched.push_back(static_cast<int>(j));
}

std::vector<std::vector<float>> byte_track::OAByteTracker::calcIous(const std::vector<Rect<float>>& a_rect,
                                                                    const std::vector<Rect<float>>& b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.empty() || b_rect.empty())
        return ious;
    ious.resize(a_rect.size(), std::vector<float>(b_rect.size(), 0.0f));
    for (size_t bi = 0; bi < b_rect.size(); ++bi)
        for (size_t ai = 0; ai < a_rect.size(); ++ai)
            ious[ai][bi] = b_rect[bi].calcIoU(a_rect[ai]);
    return ious;
}

std::vector<std::vector<float>> byte_track::OAByteTracker::calcIouDistance(const std::vector<STrackPtr>& a_tracks,
                                                                           const std::vector<STrackPtr>& b_tracks) const
{
    std::vector<Rect<float>> a_rects, b_rects;
    for (const auto& t : a_tracks) a_rects.push_back(t->getRect());
    for (const auto& t : b_tracks) b_rects.push_back(t->getRect());
    const auto ious = calcIous(a_rects, b_rects);
    std::vector<std::vector<float>> cost_matrix;
    for (size_t i = 0; i < ious.size(); ++i)
    {
        std::vector<float> row;
        for (size_t j = 0; j < ious[i].size(); ++j)
            row.push_back(1.0f - ious[i][j]);
        cost_matrix.push_back(row);
    }
    return cost_matrix;
}

double byte_track::OAByteTracker::execLapjv(const std::vector<std::vector<float>>& cost,
                                            std::vector<int>& rowsol,
                                            std::vector<int>& colsol,
                                            bool extend_cost,
                                            float cost_limit,
                                            bool return_cost) const
{
    std::vector<std::vector<float>> cost_c;
    cost_c.assign(cost.begin(), cost.end());
    std::vector<std::vector<float>> cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else if (!extend_cost)
    {
        throw std::runtime_error("The `extend_cost` variable should set True");
    }

    if (extend_cost || cost_limit < std::numeric_limits<float>::max())
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (size_t i = 0; i < cost_c_extended.size(); ++i)
            cost_c_extended[i].resize(n);

        if (cost_limit < std::numeric_limits<float>::max())
        {
            for (size_t i = 0; i < cost_c_extended.size(); ++i)
                for (size_t j = 0; j < cost_c_extended[i].size(); ++j)
                    cost_c_extended[i][j] = cost_limit / 2.0f;
        }
        else
        {
            float cost_max = -1.0f;
            for (size_t i = 0; i < cost_c.size(); ++i)
                for (size_t j = 0; j < cost_c[i].size(); ++j)
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
            for (size_t i = 0; i < cost_c_extended.size(); ++i)
                for (size_t j = 0; j < cost_c_extended[i].size(); ++j)
                    cost_c_extended[i][j] = cost_max + 1.0f;
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); ++i)
            for (size_t j = n_cols; j < cost_c_extended[i].size(); ++j)
                cost_c_extended[i][j] = 0.0f;
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                cost_c_extended[i][j] = cost_c[i][j];

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double** cost_ptr = new double*[n];
    for (int i = 0; i < n; ++i)
        cost_ptr[i] = new double[n];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cost_ptr[i][j] = cost_c[i][j];

    int* x_c = new int[n];
    int* y_c = new int[n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
        throw std::runtime_error("The result of lapjv_internal() is invalid.");

    double opt = 0.0;
    if (n != n_rows)
    {
        for (int i = 0; i < n; ++i)
        {
            if (x_c[i] >= n_cols) x_c[i] = -1;
            if (y_c[i] >= n_rows) y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; ++i) rowsol[i] = x_c[i];
        for (int i = 0; i < n_cols; ++i) colsol[i] = y_c[i];
        if (return_cost)
            for (size_t i = 0; i < rowsol.size(); ++i)
                if (rowsol[i] != -1)
                    opt += cost_ptr[i][rowsol[i]];
    }
    else if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); ++i)
            opt += cost_ptr[i][rowsol[i]];
    }

    for (int i = 0; i < n; ++i) delete[] cost_ptr[i];
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;
    return opt;
}
