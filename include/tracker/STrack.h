#pragma once

#include "tracker/Rect.h"
#include "tracker/KalmanFilter.h"
#include <opencv2/core.hpp>

#include <cstddef>
#include <deque>
#include <vector>

namespace byte_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

class STrack
{
public:
    STrack(const Rect<float>& rect, const float& score, const std::vector<float>& feat = {});
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;
    const std::vector<float>& getSmoothFeature() const;
    const std::vector<float>& getCurrentFeature() const;
    bool hasFeature() const;
    float getOcclusionCoeff() const;
    void setOcclusionCoeff(float value);

    void activate(const size_t& frame_id, const size_t& track_id);
    void reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id = -1);

    void applyGmc(const cv::Matx23f& H);
    void predict();
    void update(const STrack &new_track, const size_t &frame_id);

    void markAsLost();
    void markAsRemoved();

private:
    void updateFeatures(const std::vector<float>& feat);
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    bool is_activated_;
    float score_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;
    std::vector<float> smooth_feat_;
    std::vector<float> curr_feat_;
    std::deque<std::vector<float>> features_;
    float feat_alpha_;
    size_t feat_history_;
    float occlusion_coeff_;

    void updateRect();
};
}
