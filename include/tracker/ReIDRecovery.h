#pragma once

#include "tracker/Object.h"
#include "tracker/STrack.h"

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <unordered_map>
#include <vector>

namespace byte_track
{

class ReIDRecovery
{
public:
    using STrackPtr = std::shared_ptr<STrack>;
    using RecoverFn = std::function<bool(const STrackPtr&, const Object&)>;

    void updateGallery(size_t track_id,
                       const std::vector<float>& feature,
                       size_t max_history);

    size_t recoverLostTracks(const std::vector<STrackPtr>& lost_tracks,
                             const std::vector<Object>& detections,
                             const std::vector<std::array<float, 4>>& det_boxes,
                             const std::vector<std::vector<float>>& det_features,
                             const std::unordered_map<size_t, size_t>& active_matches,
                             size_t frame_id,
                             size_t recovery_window,
                             float similarity_threshold,
                             size_t gallery_history,
                             const RecoverFn& recover_fn);

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

    static std::vector<float> normalizeFeature(const std::vector<float>& feature);
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    static float computeIou(const std::array<float, 4>& a, const std::array<float, 4>& b);
    static float computeCenterRatio(const std::array<float, 4>& a, const std::array<float, 4>& b);
    std::vector<float> averageGalleryFeature(size_t track_id) const;
};

}  // namespace byte_track
