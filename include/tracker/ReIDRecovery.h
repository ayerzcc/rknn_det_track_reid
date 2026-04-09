#pragma once

#include "tracker/Object.h"
#include "tracker/STrack.h"

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace byte_track
{

class ReIDRecovery
{
public:
    using STrackPtr = std::shared_ptr<STrack>;
    using RecoverFn = std::function<bool(const STrackPtr&, const Object&)>;

    struct ActiveMatch
    {
        size_t track_id = 0;
        size_t det_index = 0;
        float iou = 0.0f;
    };

    struct OverrideDecision
    {
        size_t active_track_id = 0;
        size_t lost_track_id = 0;
        size_t det_index = 0;
        float active_iou = 0.0f;
        float similarity = 0.0f;
    };

    struct Stats
    {
        size_t recovered_success = 0;
        size_t over_window = 0;
        size_t track_feature_empty = 0;
        size_t det_feature_empty = 0;
        size_t det_crop_invalid = 0;
        size_t det_extract_empty = 0;
        size_t det_feature_count_mismatch = 0;
        size_t det_used_by_active = 0;
        size_t det_feature_missing = 0;
        size_t similarity_fail = 0;
        size_t geometry_fail = 0;
        size_t conflict_fail = 0;
    };

    struct FrameStats
    {
        struct BlockedCandidate
        {
            size_t track_id = 0;
            size_t det_index = 0;
            float similarity = 0.0f;
            float iou = 0.0f;
            float center_ratio = 0.0f;
        };

        size_t det_used_by_active = 0;
        size_t similarity_fail = 0;
        size_t over_window = 0;
        std::set<size_t> impacted_tracks;
        std::set<size_t> active_track_ids;
        std::set<size_t> used_det_indices;
        std::map<size_t, float> active_det_ious;
        std::vector<BlockedCandidate> blocked_candidates;
    };

    void updateGallery(size_t track_id,
                       const std::vector<float>& feature,
                       size_t max_history);

    void resetStats() { stats_ = {}; frame_stats_.clear(); }
    const Stats& getStats() const { return stats_; }
    const std::map<size_t, FrameStats>& getFrameStats() const { return frame_stats_; }
    void recordDetectionFeatureStats(size_t crop_invalid,
                                     size_t extract_empty,
                                     size_t feature_count_mismatch);

    size_t recoverLostTracks(const std::vector<STrackPtr>& lost_tracks,
                             const std::vector<Object>& detections,
                             const std::vector<std::array<float, 4>>& det_boxes,
                             const std::vector<std::vector<float>>& det_features,
                             const std::vector<ActiveMatch>& active_matches,
                             size_t frame_id,
                             size_t recovery_window,
                             float similarity_threshold,
                             size_t gallery_history,
                             const RecoverFn& recover_fn);

    std::vector<OverrideDecision> recoverFromLowQualityActiveMatches(
        const std::vector<STrackPtr>& lost_tracks,
        const std::vector<Object>& detections,
        const std::vector<std::array<float, 4>>& det_boxes,
        const std::vector<std::vector<float>>& det_features,
        const std::vector<ActiveMatch>& active_matches,
        size_t frame_id,
        size_t recovery_window,
        float similarity_threshold,
        float active_iou_threshold,
        const RecoverFn& recover_fn) const;

private:
    struct Candidate
    {
        size_t track_id = 0;
        size_t det_index = 0;
        float similarity = 0.0f;
        float iou = 0.0f;
        float center_ratio = 0.0f;
    };

    std::unordered_map<size_t, std::deque<std::vector<float>>> gallery_;
    Stats stats_;
    std::map<size_t, FrameStats> frame_stats_;

    static std::vector<float> normalizeFeature(const std::vector<float>& feature);
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    static float computeIou(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static float computeCenterRatio(const std::array<float, 4>& a, const std::array<float, 4>& b);
    std::vector<float> averageGalleryFeature(size_t track_id) const;
};

}  // namespace byte_track
