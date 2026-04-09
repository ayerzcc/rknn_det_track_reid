#include "tracker/ReIDRecovery.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace byte_track
{

void ReIDRecovery::recordDetectionFeatureStats(size_t crop_invalid,
                                               size_t extract_empty,
                                               size_t feature_count_mismatch)
{
    stats_.det_crop_invalid += crop_invalid;
    stats_.det_extract_empty += extract_empty;
    stats_.det_feature_count_mismatch += feature_count_mismatch;
}

void ReIDRecovery::updateGallery(size_t track_id,
                                 const std::vector<float>& feature,
                                 size_t max_history)
{
    if (feature.empty())
    {
        return;
    }

    auto& history = gallery_[track_id];
    history.push_back(normalizeFeature(feature));
    while (history.size() > max_history)
    {
        history.pop_front();
    }
}

size_t ReIDRecovery::recoverLostTracks(const std::vector<STrackPtr>& lost_tracks,
                                       const std::vector<Object>& detections,
                                       const std::vector<std::array<float, 4>>& det_boxes,
                                       const std::vector<std::vector<float>>& det_features,
                                       const std::vector<ActiveMatch>& active_matches,
                                       size_t frame_id,
                                       size_t recovery_window,
                                       float similarity_threshold,
                                       size_t gallery_history,
                                       const RecoverFn& recover_fn)
{
    std::set<size_t> used_dets;
    for (const auto& match : active_matches)
    {
        used_dets.insert(match.det_index);
    }
    auto& frame_stats = frame_stats_[frame_id];
    for (const auto& match : active_matches)
    {
        frame_stats.active_track_ids.insert(match.track_id);
        frame_stats.used_det_indices.insert(match.det_index);
        frame_stats.active_det_ious[match.det_index] = match.iou;
    }

    std::unordered_map<size_t, STrackPtr> lost_map;
    std::vector<Candidate> candidates;
    std::set<size_t> viable_tracks;

    for (const auto& track : lost_tracks)
    {
        if (track->getSTrackState() != STrackState::Lost)
        {
            continue;
        }

        const size_t gap = frame_id >= track->getFrameId() ? frame_id - track->getFrameId() : 0;
        if (gap > recovery_window)
        {
            ++stats_.over_window;
            ++frame_stats.over_window;
            frame_stats.impacted_tracks.insert(track->getTrackId());
            continue;
        }

        const std::vector<float> track_feature = averageGalleryFeature(track->getTrackId());
        if (track_feature.empty())
        {
            ++stats_.track_feature_empty;
            continue;
        }

        lost_map[track->getTrackId()] = track;
        const auto& rect = track->getRect();
        const std::array<float, 4> track_box = {rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y()};
        bool has_det_feature = false;
        bool has_similarity_match = false;
        bool has_geometry_match = false;

        for (size_t det_index = 0; det_index < detections.size() && det_index < det_features.size() && det_index < det_boxes.size(); ++det_index)
        {
            if (used_dets.count(det_index) > 0)
            {
                ++stats_.det_used_by_active;
                ++frame_stats.det_used_by_active;
                frame_stats.impacted_tracks.insert(track->getTrackId());

                if (!det_features[det_index].empty())
                {
                    const float similarity = cosineSimilarity(track_feature, det_features[det_index]);
                    const float iou = computeIou(track_box, det_boxes[det_index]);
                    const float center_ratio = computeCenterRatio(track_box, det_boxes[det_index]);
                    auto it = std::find_if(frame_stats.blocked_candidates.begin(),
                                           frame_stats.blocked_candidates.end(),
                                           [&](const FrameStats::BlockedCandidate& item) {
                                               return item.track_id == track->getTrackId();
                                           });
                    if (it == frame_stats.blocked_candidates.end())
                    {
                        frame_stats.blocked_candidates.push_back(
                            {track->getTrackId(), det_index, similarity, iou, center_ratio});
                    }
                    else if (similarity > it->similarity)
                    {
                        *it = {track->getTrackId(), det_index, similarity, iou, center_ratio};
                    }
                }
                continue;
            }
            if (det_features[det_index].empty())
            {
                ++stats_.det_feature_missing;
                continue;
            }
            has_det_feature = true;

            const float similarity = cosineSimilarity(track_feature, det_features[det_index]);
            const float iou = computeIou(track_box, det_boxes[det_index]);
            const float center_ratio = computeCenterRatio(track_box, det_boxes[det_index]);
            if (similarity >= similarity_threshold)
            {
                has_similarity_match = true;
                if (iou >= 0.01f || center_ratio <= 2.5f)
                {
                    has_geometry_match = true;
                    candidates.push_back({track->getTrackId(), det_index, similarity, iou, center_ratio});
                }
            }
        }

        if (!has_det_feature)
        {
            ++stats_.det_feature_empty;
            lost_map.erase(track->getTrackId());
            continue;
        }
        if (!has_similarity_match)
        {
            ++stats_.similarity_fail;
            ++frame_stats.similarity_fail;
            frame_stats.impacted_tracks.insert(track->getTrackId());
            lost_map.erase(track->getTrackId());
            continue;
        }
        if (!has_geometry_match)
        {
            ++stats_.geometry_fail;
            lost_map.erase(track->getTrackId());
            continue;
        }
        viable_tracks.insert(track->getTrackId());
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
        if (lhs.iou != rhs.iou) return lhs.iou > rhs.iou;
        return lhs.center_ratio < rhs.center_ratio;
    });

    std::set<size_t> recovered_tracks;
    size_t recovered_count = 0;
    for (const auto& candidate : candidates)
    {
        if (recovered_tracks.count(candidate.track_id) > 0 || used_dets.count(candidate.det_index) > 0)
        {
            continue;
        }

        const auto track_it = lost_map.find(candidate.track_id);
        if (track_it == lost_map.end())
        {
            continue;
        }

        if (!recover_fn(track_it->second, detections[candidate.det_index]))
        {
            continue;
        }

        updateGallery(candidate.track_id, det_features[candidate.det_index], gallery_history);
        recovered_tracks.insert(candidate.track_id);
        used_dets.insert(candidate.det_index);
        ++recovered_count;
        ++stats_.recovered_success;
    }

    for (const auto& track_id : viable_tracks)
    {
        if (recovered_tracks.count(track_id) == 0)
        {
            ++stats_.conflict_fail;
        }
    }

    return recovered_count;
}

std::vector<ReIDRecovery::OverrideDecision> ReIDRecovery::recoverFromLowQualityActiveMatches(
    const std::vector<STrackPtr>& lost_tracks,
    const std::vector<Object>& detections,
    const std::vector<std::array<float, 4>>& det_boxes,
    const std::vector<std::vector<float>>& det_features,
    const std::vector<ActiveMatch>& active_matches,
    size_t frame_id,
    size_t recovery_window,
    float similarity_threshold,
    float active_iou_threshold,
    const RecoverFn& recover_fn) const
{
    struct Candidate
    {
        size_t active_track_id = 0;
        size_t lost_track_id = 0;
        size_t det_index = 0;
        float active_iou = 0.0f;
        float similarity = 0.0f;
        float iou = 0.0f;
        float center_ratio = 0.0f;
    };

    std::unordered_map<size_t, STrackPtr> lost_map;
    for (const auto& track : lost_tracks)
    {
        if (track->getSTrackState() != STrackState::Lost)
            continue;
        const size_t gap = frame_id >= track->getFrameId() ? frame_id - track->getFrameId() : 0;
        if (gap > recovery_window)
            continue;
        if (averageGalleryFeature(track->getTrackId()).empty())
            continue;
        lost_map[track->getTrackId()] = track;
    }

    std::vector<Candidate> candidates;
    for (const auto& active_match : active_matches)
    {
        if (active_match.iou >= active_iou_threshold)
            continue;
        if (active_match.det_index >= det_features.size() || active_match.det_index >= det_boxes.size() || active_match.det_index >= detections.size())
            continue;
        if (det_features[active_match.det_index].empty())
            continue;

        for (const auto& [lost_track_id, track] : lost_map)
        {
            const std::vector<float> track_feature = averageGalleryFeature(lost_track_id);
            const auto& rect = track->getRect();
            const std::array<float, 4> track_box = {rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y()};
            const float similarity = cosineSimilarity(track_feature, det_features[active_match.det_index]);
            const float iou = computeIou(track_box, det_boxes[active_match.det_index]);
            const float center_ratio = computeCenterRatio(track_box, det_boxes[active_match.det_index]);
            if (similarity >= similarity_threshold && (iou >= 0.01f || center_ratio <= 2.5f))
            {
                candidates.push_back({active_match.track_id, lost_track_id, active_match.det_index,
                                      active_match.iou, similarity, iou, center_ratio});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
        if (lhs.active_iou != rhs.active_iou) return lhs.active_iou < rhs.active_iou;
        if (lhs.iou != rhs.iou) return lhs.iou > rhs.iou;
        return lhs.center_ratio < rhs.center_ratio;
    });

    std::set<size_t> used_active_tracks;
    std::set<size_t> used_lost_tracks;
    std::set<size_t> used_dets;
    std::vector<OverrideDecision> decisions;

    for (const auto& candidate : candidates)
    {
        if (used_active_tracks.count(candidate.active_track_id) > 0 ||
            used_lost_tracks.count(candidate.lost_track_id) > 0 ||
            used_dets.count(candidate.det_index) > 0)
            continue;

        const auto it = lost_map.find(candidate.lost_track_id);
        if (it == lost_map.end())
            continue;
        if (!recover_fn(it->second, detections[candidate.det_index]))
            continue;

        used_active_tracks.insert(candidate.active_track_id);
        used_lost_tracks.insert(candidate.lost_track_id);
        used_dets.insert(candidate.det_index);
        decisions.push_back({candidate.active_track_id, candidate.lost_track_id,
                             candidate.det_index, candidate.active_iou, candidate.similarity});
    }

    return decisions;
}

std::vector<float> ReIDRecovery::normalizeFeature(const std::vector<float>& feature)
{
    if (feature.empty())
    {
        return {};
    }

    const float norm = std::sqrt(std::inner_product(feature.begin(), feature.end(), feature.begin(), 0.0f));
    if (norm <= 1e-6f)
    {
        return feature;
    }

    std::vector<float> normalized(feature.size(), 0.0f);
    for (size_t i = 0; i < feature.size(); ++i)
    {
        normalized[i] = feature[i] / norm;
    }
    return normalized;
}

float ReIDRecovery::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || b.empty() || a.size() != b.size())
    {
        return 0.0f;
    }

    const std::vector<float> na = normalizeFeature(a);
    const std::vector<float> nb = normalizeFeature(b);
    return std::inner_product(na.begin(), na.end(), nb.begin(), 0.0f);
}

float ReIDRecovery::computeIou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }
    return inter_area / union_area;
}

float ReIDRecovery::computeCenterRatio(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float ax = 0.5f * (a[0] + a[2]);
    const float ay = 0.5f * (a[1] + a[3]);
    const float bx = 0.5f * (b[0] + b[2]);
    const float by = 0.5f * (b[1] + b[3]);
    const float dist = std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
    const float aw = std::max(1.0f, a[2] - a[0]);
    const float ah = std::max(1.0f, a[3] - a[1]);
    const float bw = std::max(1.0f, b[2] - b[0]);
    const float bh = std::max(1.0f, b[3] - b[1]);
    const float norm = std::max(1.0f, 0.5f * (std::sqrt(aw * aw + ah * ah) + std::sqrt(bw * bw + bh * bh)));
    return dist / norm;
}

std::vector<float> ReIDRecovery::averageGalleryFeature(size_t track_id) const
{
    const auto it = gallery_.find(track_id);
    if (it == gallery_.end() || it->second.empty())
    {
        return {};
    }

    std::vector<float> avg(it->second.front().size(), 0.0f);
    for (const auto& feature : it->second)
    {
        for (size_t i = 0; i < feature.size(); ++i)
        {
            avg[i] += feature[i];
        }
    }
    for (float& value : avg)
    {
        value /= static_cast<float>(it->second.size());
    }
    return normalizeFeature(avg);
}

}  // namespace byte_track
