#pragma once

#include "tracker/BYTETracker.h"
#include "tracker/Object.h"
#include "tracker/GMC.h"
#include "tracker/OcclusionAware.h"

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace bot_sort
{

class BoTSORT
{
public:
    using STrackPtr = std::shared_ptr<byte_track::STrack>;

    BoTSORT(float track_high_thresh = 0.6f,
            float track_low_thresh  = 0.1f,
            float new_track_thresh  = 0.7f,
            float match_thresh      = 0.5f,
            int   track_buffer      = 30,
            GMCMethod gmc_method    = GMCMethod::ORB,
            int   gmc_downscale     = 2);

    virtual ~BoTSORT() = default;

    virtual std::vector<STrackPtr> update(const std::vector<byte_track::Object>& objects,
                                          const cv::Mat& img = cv::Mat());

    const std::vector<STrackPtr>& getTrackedStracks() const;
    const std::vector<STrackPtr>& getLostStracks() const;
    size_t getFrameId() const;
    size_t getMaxTimeLost() const;
    bool recoverTrack(const STrackPtr& track, const byte_track::Object& object);
    const std::string& getLastDebugJson() const;

protected:
    static std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a,
                                               const std::vector<STrackPtr>& b);
    static std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a,
                                             const std::vector<STrackPtr>& b);
    static void removeDuplicateStracks(std::vector<STrackPtr>& a,
                                       std::vector<STrackPtr>& b);

    static std::vector<std::vector<float>> calcIouDistance(
        const std::vector<STrackPtr>& atracks,
        const std::vector<STrackPtr>& btracks);

    static void linearAssignment(
        const std::vector<std::vector<float>>& cost_matrix,
        float thresh,
        std::vector<std::vector<int>>& matches,
        std::vector<int>& unmatched_rows,
        std::vector<int>& unmatched_cols);

    float track_high_thresh_;
    float track_low_thresh_;
    float new_track_thresh_;
    float match_thresh_;
    size_t max_time_lost_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;

    std::unique_ptr<GMC> gmc_;
    std::string last_debug_json_;
};

class OABoTSORT : public BoTSORT
{
public:
    OABoTSORT(float track_high_thresh = 0.6f,
              float track_low_thresh  = 0.1f,
              float new_track_thresh  = 0.7f,
              float match_thresh      = 0.8f,
              int   track_buffer      = 30,
              GMCMethod gmc_method    = GMCMethod::ORB,
              int   gmc_downscale     = 2,
              bool  use_oao           = true,
              bool  use_bam           = true,
              float oa_tau            = 0.15f,
              float oa_k_x            = 4.24264f,
              float oa_k_y            = 3.0f);

    virtual std::vector<STrackPtr> update(const std::vector<byte_track::Object>& objects,
                                          const cv::Mat& img = cv::Mat()) override;

private:
    bool use_oao_;
    bool use_bam_;

    std::unique_ptr<OcclusionAwareModule> oam_;
    std::unique_ptr<OcclusionAwareOffset> oao_;
    std::unique_ptr<BiasAwareMomentum>    bam_;

    int img_h_ = 0;
    int img_w_ = 0;
};

} // namespace bot_sort
