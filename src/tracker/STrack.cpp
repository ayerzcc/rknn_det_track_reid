#include "tracker/STrack.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <numeric>

namespace
{
std::vector<float> normalizeFeature(const std::vector<float>& feat)
{
    if (feat.empty())
    {
        return {};
    }

    const float norm = std::sqrt(std::inner_product(feat.begin(), feat.end(), feat.begin(), 0.0f));
    if (norm <= 1e-6f)
    {
        return feat;
    }

    std::vector<float> normalized(feat.size(), 0.0f);
    for (size_t i = 0; i < feat.size(); ++i)
    {
        normalized[i] = feat[i] / norm;
    }
    return normalized;
}
}

byte_track::STrack::STrack(const Rect<float>& rect, const float& score, const std::vector<float>& feat) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0),
    smooth_feat_(),
    curr_feat_(),
    features_(),
    feat_alpha_(0.9f),
    feat_history_(50),
    occlusion_coeff_(0.0f)
{
    if (!feat.empty())
    {
        updateFeatures(feat);
    }
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}
const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

const std::vector<float>& byte_track::STrack::getSmoothFeature() const
{
    return smooth_feat_;
}

const std::vector<float>& byte_track::STrack::getCurrentFeature() const
{
    return curr_feat_;
}

bool byte_track::STrack::hasFeature() const
{
    return !smooth_feat_.empty();
}

float byte_track::STrack::getOcclusionCoeff() const
{
    return occlusion_coeff_;
}

void byte_track::STrack::setOcclusionCoeff(float value)
{
    occlusion_coeff_ = value;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = (frame_id == 1);
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
    if (new_track.hasFeature())
    {
        updateFeatures(new_track.getCurrentFeature());
    }
}

void byte_track::STrack::applyGmc(const cv::Matx23f& H)
{
    const float r00 = H(0, 0), r01 = H(0, 1), tx = H(0, 2);
    const float r10 = H(1, 0), r11 = H(1, 1), ty = H(1, 2);

    const float cx = mean_[0];
    const float cy = mean_[1];
    mean_[0] = r00 * cx + r01 * cy + tx;
    mean_[1] = r10 * cx + r11 * cy + ty;

    const float vx = mean_[4];
    const float vy = mean_[5];
    mean_[4] = r00 * vx + r01 * vy;
    mean_[5] = r10 * vx + r11 * vy;

    KalmanFilter::StateCov R8 = KalmanFilter::StateCov::Identity();
    R8(0, 0) = r00; R8(0, 1) = r01;
    R8(1, 0) = r10; R8(1, 1) = r11;
    R8(4, 4) = r00; R8(4, 5) = r01;
    R8(5, 4) = r10; R8(5, 5) = r11;
    covariance_ = R8 * covariance_ * R8.transpose();

    updateRect();
}

void byte_track::STrack::predict()
{
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    kalman_filter_.predict(mean_, covariance_);
    updateRect();
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
    if (new_track.hasFeature())
    {
        updateFeatures(new_track.getCurrentFeature());
    }
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateFeatures(const std::vector<float>& feat)
{
    if (feat.empty())
    {
        return;
    }

    curr_feat_ = normalizeFeature(feat);
    if (smooth_feat_.empty())
    {
        smooth_feat_ = curr_feat_;
    }
    else
    {
        if (smooth_feat_.size() != curr_feat_.size())
        {
            smooth_feat_ = curr_feat_;
        }
        else
        {
            for (size_t i = 0; i < smooth_feat_.size(); ++i)
            {
                smooth_feat_[i] = feat_alpha_ * smooth_feat_[i] + (1.0f - feat_alpha_) * curr_feat_[i];
            }
            smooth_feat_ = normalizeFeature(smooth_feat_);
        }
    }

    features_.push_back(curr_feat_);
    while (features_.size() > feat_history_)
    {
        features_.pop_front();
    }
}

void byte_track::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}
