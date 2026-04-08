#pragma once

#include "tracker/BYTETracker.h"
#include "tracker/OcclusionAware.h"

namespace byte_track
{

class OAByteTracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    OAByteTracker(const int& frame_rate = 30,
                  const int& track_buffer = 30,
                  const float& track_thresh = 0.5f,
                  const float& high_thresh = 0.6f,
                  const float& new_track_thresh = 0.7f,
                  const float& match_thresh = 0.8f,
                  bool use_oao = true,
                  bool use_bam = true,
                  float tau = 0.15f,
                  float k_x = 4.24264f,
                  float k_y = 3.0f);
    ~OAByteTracker();

    std::vector<STrackPtr> update(const std::vector<Object>& objects, const cv::Mat& img = cv::Mat());
    const std::vector<STrackPtr>& getTrackedStracks() const;
    const std::vector<STrackPtr>& getLostStracks() const;
    size_t getFrameId() const;
    size_t getMaxTimeLost() const;
    bool recoverTrack(const STrackPtr& track, const Object& object);

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a_tlist,
                                        const std::vector<STrackPtr>& b_tlist) const;
    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a_tlist,
                                      const std::vector<STrackPtr>& b_tlist) const;
    void removeDuplicateStracks(const std::vector<STrackPtr>& a_stracks,
                                const std::vector<STrackPtr>& b_stracks,
                                std::vector<STrackPtr>& a_res,
                                std::vector<STrackPtr>& b_res) const;
    void linearAssignment(const std::vector<std::vector<float>>& cost_matrix,
                          const int& cost_matrix_size,
                          const int& cost_matrix_size_size,
                          const float& thresh,
                          std::vector<std::vector<int>>& matches,
                          std::vector<int>& b_unmatched,
                          std::vector<int>& a_unmatched) const;
    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr>& a_tracks,
                                                    const std::vector<STrackPtr>& b_tracks) const;
    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>>& a_rect,
                                             const std::vector<Rect<float>>& b_rect) const;
    double execLapjv(const std::vector<std::vector<float>>& cost,
                     std::vector<int>& rowsol,
                     std::vector<int>& colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

private:
    const float track_thresh_;
    const float high_thresh_;
    const float new_track_thresh_;
    const float match_thresh_;
    const size_t max_time_lost_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;

    bool use_oao_;
    bool use_bam_;
    float tau_;
    std::unique_ptr<bot_sort::OcclusionAwareModule> oam_;
    std::unique_ptr<bot_sort::OcclusionAwareOffset> oao_;
    std::unique_ptr<bot_sort::BiasAwareMomentum> bam_;
    int img_h_ = 0;
    int img_w_ = 0;
};

}
