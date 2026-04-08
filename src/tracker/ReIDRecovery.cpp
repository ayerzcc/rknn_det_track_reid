#include "tracker/ReIDRecovery.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace byte_track
{

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
                                       const std::unordered_map<size_t, size_t>& active_matches,
                                       size_t frame_id,
                                       size_t recovery_window,
                                       float similarity_threshold,
                                       size_t gallery_history,
                                       const RecoverFn& recover_fn)
{
    std::set<size_t> used_dets;
    for (const auto& match : active_matches)
    {
        used_dets.insert(match.second);
    }

    std::unordered_map<size_t, STrackPtr> lost_map;
    std::vector<Candidate> candidates;

    for (const auto& track : lost_tracks)
    {
        if (track->getSTrackState() != STrackState::Lost)
        {
            continue;
        }

        const size_t gap = frame_id >= track->getFrameId() ? frame_id - track->getFrameId() : 0;
        if (gap > recovery_window)
        {
            continue;
        }

        const std::vector<float> track_feature = averageGalleryFeature(track->getTrackId());
        if (track_feature.empty())
        {
            continue;
        }

        lost_map[track->getTrackId()] = track;
        const auto& rect = track->getRect();
        const std::array<float, 4> track_box = {rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y()};

        for (size_t det_index = 0; det_index < detections.size() && det_index < det_features.size() && det_index < det_boxes.size(); ++det_index)
        {
            if (used_dets.count(det_index) > 0 || det_features[det_index].empty())
            {
                continue;
            }

            const float similarity = cosineSimilarity(track_feature, det_features[det_index]);
            const float iou = computeIou(track_box, det_boxes[det_index]);
            const float center_ratio = computeCenterRatio(track_box, det_boxes[det_index]);
            if (similarity >= similarity_threshold && (iou >= 0.01f || center_ratio <= 2.5f))
            {
                candidates.push_back({track->getTrackId(), det_index, similarity, iou, center_ratio});
            }
        }
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
    }

    return recovered_count;
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
