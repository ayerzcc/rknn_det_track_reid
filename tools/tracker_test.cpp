#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <chrono>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector.hpp"
#include "module_test_utils.hpp"
#include "personfeature_extractor.hpp"
#include "tracker/BYTETracker.h"
#include "tracker/OAByteTracker.h"
#include "tracker/BoTSORT.h"
#include "tracker/Object.h"
#include "tracker/ReIDRecovery.h"

namespace
{
enum class TrackerType
{
    ByteTrack,
    OAByteTrack,
    BoTSORT,
    OABoTSORT
};

using STrackPtr = std::shared_ptr<byte_track::STrack>;

struct Options
{
    std::string input_path;
    std::string output_path;
    std::string dump_tracks_path;
    std::string dump_debug_path;
    std::string detections_path;
    std::string model_path;
    TrackerType tracker_type = TrackerType::ByteTrack;
    int frame_rate = 30;
    int max_frames = -1;
    int track_buffer = -1;
    int recovery_window = -1;
    float track_thresh = 0.0f;  // loaded from config
    float high_thresh = 0.0f;   // loaded from config
    float match_thresh = 0.0f;  // loaded from config
    bot_sort::GMCMethod gmc_method = bot_sort::GMCMethod::ORB;
    bool use_oao = true;
    bool use_bam = true;
    bool enable_reid_recovery = true;
    bool show_lost_tracks = true;
};

struct DetectionRecord
{
    float x1 = 0.0f;
    float y1 = 0.0f;
    float y2 = 0.0f;
    float x2 = 0.0f;
    float score = 0.0f;
    int label = 0;
};

struct DetectionInfo
{
    byte_track::Object object;
    std::array<float, 4> box{};
    std::vector<float> feature;
    bool has_feature = false;
};

struct RecoveryCandidate
{
    size_t track_id = 0;
    size_t det_index = 0;
    float similarity = 0.0f;
    float iou = 0.0f;
    float center_ratio = 0.0f;
};

cv::Scalar color_for_track_id(size_t track_id)
{
    static const std::array<cv::Scalar, 12> palette = {
        cv::Scalar(0, 200, 255),
        cv::Scalar(0, 180, 120),
        cv::Scalar(255, 128, 0),
        cv::Scalar(0, 128, 255),
        cv::Scalar(180, 80, 255),
        cv::Scalar(255, 80, 160),
        cv::Scalar(80, 220, 80),
        cv::Scalar(255, 180, 60),
        cv::Scalar(80, 160, 255),
        cv::Scalar(200, 120, 0),
        cv::Scalar(120, 220, 220),
        cv::Scalar(160, 160, 255),
    };
    return palette[track_id % palette.size()];
}

void print_usage()
{
    std::cout
        << "Usage: tracker_test --input <video|image> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --tracker <type>       Tracker type: bytetrack (default), oa_bytetrack, botsort, oabotsort\n"
        << "  --output <path>        Output image/video path\n"
        << "  --dump-tracks <csv>    Dump per-frame active track boxes\n"
        << "  --dump-debug <jsonl>   Dump per-frame debug info\n"
        << "  --detections <txt>     Detection file (skip online detection)\n"
        << "  --model <rknn>         Detection model path\n"
        << "  --fps <n>              Frame rate (default: 30)\n"
        << "  --max-frames <n>       Stop after n frames\n"
        << "  --track-buffer <n>     Track buffer / max lost frames (default: 30)\n"
        << "  --track-thresh <f>     Track confidence threshold (default: 0.5)\n"
        << "  --high-thresh <f>      High confidence threshold (default: 0.6)\n"
        << "  --match-thresh <f>     Match threshold (default: 0.8)\n"
        << "  --recovery-window <n>  ReID recovery window in frames (default: min(max_lost, 10))\n"
        << "  --gmc-method <type>    GMC method for BoTSORT: orb (default), sparse, none\n"
        << "  --no-oao               Disable OAO for OABoTSORT\n"
        << "  --no-bam               Disable BAM for OABoTSORT\n"
        << "  --no-reid-recovery     Disable ReID recovery for BoTSORT/OABoTSORT\n"
        << "  --hide-lost-tracks     Do not visualize short-lived lost tracks\n"
        << "\n"
        << "Detection file format: frame_id x1 y1 x2 y2 score [label]\n";
}

TrackerType parse_tracker_type(const std::string& s)
{
    if (s == "oa_bytetrack") return TrackerType::OAByteTrack;
    if (s == "botsort") return TrackerType::BoTSORT;
    if (s == "oabotsort") return TrackerType::OABoTSORT;
    return TrackerType::ByteTrack;
}

std::string tracker_type_name(TrackerType t)
{
    switch (t) {
        case TrackerType::OAByteTrack: return "OA-ByteTrack";
        case TrackerType::BoTSORT: return "BoTSORT";
        case TrackerType::OABoTSORT: return "OA-BoT-SORT";
        default: return "ByteTrack";
    }
}

bool parse_args(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (arg == "--dump-tracks" && i + 1 < argc) {
            options.dump_tracks_path = argv[++i];
        } else if (arg == "--dump-debug" && i + 1 < argc) {
            options.dump_debug_path = argv[++i];
        } else if (arg == "--detections" && i + 1 < argc) {
            options.detections_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            options.model_path = argv[++i];
        } else if (arg == "--tracker" && i + 1 < argc) {
            options.tracker_type = parse_tracker_type(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            options.frame_rate = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            options.max_frames = std::stoi(argv[++i]);
        } else if (arg == "--track-buffer" && i + 1 < argc) {
            options.track_buffer = std::stoi(argv[++i]);
        } else if (arg == "--track-thresh" && i + 1 < argc) {
            options.track_thresh = std::stof(argv[++i]);
        } else if (arg == "--high-thresh" && i + 1 < argc) {
            options.high_thresh = std::stof(argv[++i]);
        } else if (arg == "--match-thresh" && i + 1 < argc) {
            options.match_thresh = std::stof(argv[++i]);
        } else if (arg == "--recovery-window" && i + 1 < argc) {
            options.recovery_window = std::stoi(argv[++i]);
        } else if (arg == "--gmc-method" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "sparse") options.gmc_method = bot_sort::GMCMethod::SparseOptFlow;
            else if (v == "none") options.gmc_method = bot_sort::GMCMethod::None;
            else options.gmc_method = bot_sort::GMCMethod::ORB;
        } else if (arg == "--no-oao") {
            options.use_oao = false;
        } else if (arg == "--no-bam") {
            options.use_bam = false;
        } else if (arg == "--no-reid-recovery") {
            options.enable_reid_recovery = false;
        } else if (arg == "--hide-lost-tracks") {
            options.show_lost_tracks = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage();
            return false;
        }
    }

    if (options.input_path.empty()) {
        print_usage();
        return false;
    }
    return true;
}

std::unordered_map<int, std::vector<DetectionRecord>> load_detection_file(const std::string& path)
{
    std::unordered_map<int, std::vector<DetectionRecord>> records;
    if (path.empty()) {
        return records;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open detections file: " + path);
    }

    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int frame_id = 0;
        DetectionRecord record;
        if (!(iss >> frame_id >> record.x1 >> record.y1 >> record.x2 >> record.y2 >> record.score)) {
            throw std::runtime_error("Invalid detections file format at line " + std::to_string(line_no));
        }
        if (!(iss >> record.label)) {
            record.label = 0;
        }
        records[frame_id].push_back(record);
    }

    return records;
}

std::vector<byte_track::Object> load_frame_detections_from_file(
    const std::unordered_map<int, std::vector<DetectionRecord>>& records,
    int frame_index)
{
    std::vector<byte_track::Object> objects;
    const auto it = records.find(frame_index);
    if (it == records.end()) {
        return objects;
    }

    for (const auto& record : it->second) {
        const float width = std::max(0.0f, record.x2 - record.x1);
        const float height = std::max(0.0f, record.y2 - record.y1);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }
        objects.emplace_back(byte_track::Rect<float>(record.x1, record.y1, width, height), record.label, record.score);
    }
    return objects;
}

std::vector<byte_track::Object> detect_frame(PersonDetector& detector, const cv::Mat& frame)
{
    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    const int input_width = params_config->getKeyValue<int>("detect_image_width");
    const int input_height = params_config->getKeyValue<int>("detect_image_height");

    module_test::LetterboxInfo letterbox_info;
    cv::Mat detector_input;
    if (!module_test::resize_with_letterbox_rgb(frame, detector_input, cv::Size(input_width, input_height), letterbox_info)) {
        return {};
    }

    PersonDetector::DetectionResult result;
    detector.detect(detector_input, result);

    std::vector<byte_track::Object> objects;
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const auto mapped_box = module_test::map_box_to_original(result.boxes[i], letterbox_info, frame.cols, frame.rows);
        const float width = std::max(0.0f, mapped_box[2] - mapped_box[0]);
        const float height = std::max(0.0f, mapped_box[3] - mapped_box[1]);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }
        objects.emplace_back(byte_track::Rect<float>(mapped_box[0], mapped_box[1], width, height),
                             result.class_ids[i],
                             result.scores[i]);
    }
    return objects;
}

std::array<float, 4> rect_to_xyxy(const byte_track::Rect<float>& rect)
{
    return {rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y()};
}

float compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

float compute_center_ratio(const std::array<float, 4>& a, const std::array<float, 4>& b)
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

std::vector<float> normalize_feature(const std::vector<float>& feature)
{
    if (feature.empty()) {
        return {};
    }

    const float norm = std::sqrt(std::inner_product(feature.begin(), feature.end(), feature.begin(), 0.0f));
    if (norm <= 1e-6f) {
        return feature;
    }

    std::vector<float> normalized(feature.size(), 0.0f);
    for (size_t i = 0; i < feature.size(); ++i) {
        normalized[i] = feature[i] / norm;
    }
    return normalized;
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }

    const std::vector<float> na = normalize_feature(a);
    const std::vector<float> nb = normalize_feature(b);
    return std::inner_product(na.begin(), na.end(), nb.begin(), 0.0f);
}

std::vector<DetectionInfo> build_detection_infos(const std::vector<byte_track::Object>& objects)
{
    std::vector<DetectionInfo> detections;
    detections.reserve(objects.size());

    for (const auto& object : objects) {
        DetectionInfo info{object, rect_to_xyxy(object.rect), {}, false};
        detections.push_back(std::move(info));
    }

    return detections;
}

void attach_features_to_detections(std::vector<DetectionInfo>& detections,
                                   const std::vector<std::vector<float>>& features)
{
    const size_t n = std::min(detections.size(), features.size());
    for (size_t i = 0; i < n; ++i) {
        detections[i].feature = features[i];
        detections[i].has_feature = !features[i].empty();
        detections[i].object.feature = features[i];
    }
}

std::vector<byte_track::ReIDRecovery::ActiveMatch> associate_tracks_to_detections(const std::vector<STrackPtr>& tracks,
                                                                                  const std::vector<DetectionInfo>& detections,
                                                                                  float min_iou)
{
    struct MatchItem {
        size_t track_id;
        size_t det_index;
        float iou;
    };

    std::vector<MatchItem> candidates;
    for (const auto& track : tracks) {
        const auto track_box = rect_to_xyxy(track->getRect());
        for (size_t det_index = 0; det_index < detections.size(); ++det_index) {
            const float iou = compute_iou(track_box, detections[det_index].box);
            if (iou >= min_iou) {
                candidates.push_back({track->getTrackId(), det_index, iou});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const MatchItem& lhs, const MatchItem& rhs) {
        return lhs.iou > rhs.iou;
    });

    std::vector<byte_track::ReIDRecovery::ActiveMatch> matches;
    std::set<size_t> used_tracks;
    std::set<size_t> used_dets;
    for (const auto& candidate : candidates) {
        if (used_tracks.count(candidate.track_id) > 0 || used_dets.count(candidate.det_index) > 0) {
            continue;
        }
        used_tracks.insert(candidate.track_id);
        used_dets.insert(candidate.det_index);
        matches.push_back({candidate.track_id, candidate.det_index, candidate.iou});
    }
    return matches;
}

std::vector<STrackPtr> collect_active_tracks(const std::vector<STrackPtr>& tracks)
{
    std::vector<STrackPtr> active;
    for (const auto& track : tracks) {
        if (track->isActivated() && track->getSTrackState() == byte_track::STrackState::Tracked) {
            active.push_back(track);
        }
    }
    return active;
}

std::vector<STrackPtr> collect_lost_tracks(const std::vector<STrackPtr>& tracks)
{
    std::vector<STrackPtr> lost;
    for (const auto& track : tracks) {
        if (track->getSTrackState() == byte_track::STrackState::Lost) {
            lost.push_back(track);
        }
    }
    return lost;
}

std::vector<STrackPtr> collect_recent_lost_tracks(const std::vector<STrackPtr>& tracks,
                                                  size_t frame_id,
                                                  size_t max_gap)
{
    std::vector<STrackPtr> recent_lost;
    for (const auto& track : tracks) {
        if (track->getSTrackState() != byte_track::STrackState::Lost) {
            continue;
        }
        const size_t gap = frame_id >= track->getFrameId() ? frame_id - track->getFrameId() : 0;
        if (gap <= max_gap) {
            recent_lost.push_back(track);
        }
    }
    return recent_lost;
}

void update_gallery(std::unordered_map<size_t, std::deque<std::vector<float>>>& gallery,
                    size_t track_id,
                    const std::vector<float>& feature,
                    size_t max_history)
{
    if (feature.empty()) {
        return;
    }

    auto& history = gallery[track_id];
    history.push_back(normalize_feature(feature));
    while (history.size() > max_history) {
        history.pop_front();
    }
}

std::vector<float> average_gallery_feature(const std::unordered_map<size_t, std::deque<std::vector<float>>>& gallery,
                                           size_t track_id)
{
    const auto it = gallery.find(track_id);
    if (it == gallery.end() || it->second.empty()) {
        return {};
    }

    std::vector<float> avg(it->second.front().size(), 0.0f);
    for (const auto& feature : it->second) {
        for (size_t i = 0; i < feature.size(); ++i) {
            avg[i] += feature[i];
        }
    }
    for (float& value : avg) {
        value /= static_cast<float>(it->second.size());
    }
    return normalize_feature(avg);
}

template <typename TrackerT>
size_t recover_lost_tracks(TrackerT& tracker,
                           const std::vector<DetectionInfo>& detections,
                           const std::unordered_map<size_t, size_t>& active_matches,
                           std::unordered_map<size_t, std::deque<std::vector<float>>>& gallery,
                           float similarity_threshold,
                           size_t recovery_window,
                           size_t gallery_history)
{
    std::set<size_t> used_dets;
    for (const auto& match : active_matches) {
        used_dets.insert(match.second);
    }

    std::unordered_map<size_t, STrackPtr> lost_map;
    std::vector<RecoveryCandidate> candidates;

    for (const auto& track : tracker.getLostStracks()) {
        if (track->getSTrackState() != byte_track::STrackState::Lost) {
            continue;
        }
        const size_t gap = tracker.getFrameId() >= track->getFrameId() ? tracker.getFrameId() - track->getFrameId() : 0;
        if (gap > recovery_window) {
            continue;
        }

        const std::vector<float> track_feature = average_gallery_feature(gallery, track->getTrackId());
        if (track_feature.empty()) {
            continue;
        }

        lost_map[track->getTrackId()] = track;
        const auto track_box = rect_to_xyxy(track->getRect());
        for (size_t det_index = 0; det_index < detections.size(); ++det_index) {
            if (used_dets.count(det_index) > 0 || !detections[det_index].has_feature) {
                continue;
            }

            const float similarity = cosine_similarity(track_feature, detections[det_index].feature);
            const float iou = compute_iou(track_box, detections[det_index].box);
            const float center_ratio = compute_center_ratio(track_box, detections[det_index].box);
            if (similarity >= similarity_threshold && (iou >= 0.01f || center_ratio <= 2.5f)) {
                candidates.push_back({track->getTrackId(), det_index, similarity, iou, center_ratio});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const RecoveryCandidate& lhs, const RecoveryCandidate& rhs) {
        if (lhs.similarity != rhs.similarity) {
            return lhs.similarity > rhs.similarity;
        }
        if (lhs.iou != rhs.iou) {
            return lhs.iou > rhs.iou;
        }
        return lhs.center_ratio < rhs.center_ratio;
    });

    std::set<size_t> recovered_tracks;
    size_t recovered_count = 0;
    for (const auto& candidate : candidates) {
        if (recovered_tracks.count(candidate.track_id) > 0 || used_dets.count(candidate.det_index) > 0) {
            continue;
        }

        const auto track_it = lost_map.find(candidate.track_id);
        if (track_it == lost_map.end()) {
            continue;
        }

        if (!tracker.recoverTrack(track_it->second, detections[candidate.det_index].object)) {
            continue;
        }

        update_gallery(gallery,
                       candidate.track_id,
                       detections[candidate.det_index].feature,
                       gallery_history);
        recovered_tracks.insert(candidate.track_id);
        used_dets.insert(candidate.det_index);
        ++recovered_count;
    }

    return recovered_count;
}

void draw_tracks(cv::Mat& frame,
                 const std::vector<STrackPtr>& tracks,
                 const cv::Scalar& color,
                 const std::string& prefix = "")
{
    for (const auto& track : tracks) {
        const auto& rect = track->getRect();
        const cv::Rect draw_rect(static_cast<int>(rect.x()),
                                 static_cast<int>(rect.y()),
                                 static_cast<int>(rect.width()),
                                 static_cast<int>(rect.height()));
        if (draw_rect.width <= 0 || draw_rect.height <= 0) {
            continue;
        }

        cv::rectangle(frame, draw_rect, color, 2);
        std::string label = prefix.empty() ? "ID:" : prefix + " ";
        label += std::to_string(track->getTrackId());
        if (track->getScore() > 0.0f) {
            label += cv::format(" %.2f", track->getScore());
        }
        const int baseline = 0;
        const cv::Size txt_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
        const int box_y1 = std::max(0, draw_rect.y - txt_size.height - 6);
        const int box_y2 = std::max(0, draw_rect.y);
        cv::rectangle(frame,
                      cv::Point(draw_rect.x, box_y1),
                      cv::Point(draw_rect.x + txt_size.width + 6, box_y2),
                      color,
                      -1);
        cv::putText(frame,
                    label,
                    cv::Point(draw_rect.x + 3, std::max(15, draw_rect.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);
    }
}

void draw_track(cv::Mat& frame,
                const STrackPtr& track,
                const cv::Scalar& color,
                const std::string& prefix = "")
{
    if (!track) {
        return;
    }

    const std::vector<STrackPtr> single_track{track};
    draw_tracks(frame, single_track, color, prefix);
}

bool write_or_open_video(cv::VideoWriter& writer,
                         const Options& options,
                         const cv::VideoCapture& cap,
                         const cv::Mat& frame)
{
    if (options.output_path.empty()) {
        return true;
    }

    if (!writer.isOpened()) {
        const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0
            ? cap.get(cv::CAP_PROP_FPS) : static_cast<double>(options.frame_rate);
        writer.open(options.output_path,
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps,
                    frame.size());
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output video: " << options.output_path << std::endl;
            return false;
        }
    }

    writer.write(frame);
    return true;
}

std::vector<STrackPtr> run_tracker_update(byte_track::BYTETracker& tracker,
                                          const std::vector<byte_track::Object>& objects,
                                          const cv::Mat&)
{
    return tracker.update(objects);
}

std::vector<STrackPtr> run_tracker_update(byte_track::OAByteTracker& tracker,
                                          const std::vector<byte_track::Object>& objects,
                                          const cv::Mat& frame)
{
    return tracker.update(objects, frame);
}

std::vector<STrackPtr> run_tracker_update(bot_sort::BoTSORT& tracker,
                                          const std::vector<byte_track::Object>& objects,
                                          const cv::Mat& frame)
{
    return tracker.update(objects, frame);
}

std::vector<STrackPtr> run_tracker_update(bot_sort::OABoTSORT& tracker,
                                          const std::vector<byte_track::Object>& objects,
                                          const cv::Mat& frame)
{
    return tracker.update(objects, frame);
}

void dump_tracks(std::ofstream& file, int frame_index, const std::vector<STrackPtr>& tracks)
{
    for (const auto& track : tracks) {
        const auto& rect = track->getRect();
        file << frame_index << ","
             << track->getTrackId() << ","
             << rect.tl_x() << ","
             << rect.tl_y() << ","
             << rect.br_x() << ","
             << rect.br_y() << ","
             << track->getScore() << "\n";
    }
}

void dump_debug(std::ofstream& file,
                int frame_index,
                const std::vector<DetectionInfo>& detections,
                const std::vector<STrackPtr>& active_tracks,
                const std::vector<STrackPtr>& recent_lost,
                size_t recovered_count)
{
    file << "{\"frame\":" << frame_index << ",\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        file << "{\"x1\":" << d.box[0]
             << ",\"y1\":" << d.box[1]
             << ",\"x2\":" << d.box[2]
             << ",\"y2\":" << d.box[3]
             << ",\"score\":" << d.object.prob
             << "}";
        if (i + 1 < detections.size()) file << ",";
    }
    file << "],\"active_tracks\":[";
    for (size_t i = 0; i < active_tracks.size(); ++i) {
        const auto& t = active_tracks[i];
        const auto& r = t->getRect();
        file << "{\"id\":" << t->getTrackId()
             << ",\"x1\":" << r.tl_x()
             << ",\"y1\":" << r.tl_y()
             << ",\"x2\":" << r.br_x()
             << ",\"y2\":" << r.br_y()
             << ",\"score\":" << t->getScore()
             << "}";
        if (i + 1 < active_tracks.size()) file << ",";
    }
    file << "],\"lost_tracks\":[";
    for (size_t i = 0; i < recent_lost.size(); ++i) {
        const auto& t = recent_lost[i];
        const auto& r = t->getRect();
        file << "{\"id\":" << t->getTrackId()
             << ",\"x1\":" << r.tl_x()
             << ",\"y1\":" << r.tl_y()
             << ",\"x2\":" << r.br_x()
             << ",\"y2\":" << r.br_y()
             << "}";
        if (i + 1 < recent_lost.size()) file << ",";
    }
    file << "],\"recovered\":" << recovered_count << "}\n";
}

template <typename TrackerT>
std::string get_tracker_internal_debug(const TrackerT&) { return {}; }

template <>
std::string get_tracker_internal_debug<bot_sort::BoTSORT>(const bot_sort::BoTSORT& tracker)
{
    return tracker.getLastDebugJson();
}

template <>
std::string get_tracker_internal_debug<bot_sort::OABoTSORT>(const bot_sort::OABoTSORT& tracker)
{
    return tracker.getLastDebugJson();
}

template <>
std::string get_tracker_internal_debug<byte_track::BYTETracker>(const byte_track::BYTETracker& tracker)
{
    return tracker.getLastDebugJson();
}

template <typename TrackerT>
int run_botsort_image(TrackerT& tracker,
                      const Options& options,
                      const std::unordered_map<int, std::vector<DetectionRecord>>& detection_records,
                      PersonDetector* detector,
                      PersonFeatureExtractor& extractor)
{
    cv::Mat image = cv::imread(options.input_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << options.input_path << std::endl;
        return EXIT_FAILURE;
    }

    const auto objects = options.detections_path.empty()
        ? detect_frame(*detector, image)
        : load_frame_detections_from_file(detection_records, 0);
    const auto detections = build_detection_infos(objects);

    std::vector<byte_track::Object> tracker_objects;
        tracker_objects.reserve(detections.size());
        for (const auto& detection : detections) {
            tracker_objects.push_back(detection.object);
        }

        run_tracker_update(tracker, tracker_objects, image);
        const auto active_tracks = collect_active_tracks(tracker.getTrackedStracks());
        for (const auto& track : active_tracks) {
            draw_track(image, track, color_for_track_id(track->getTrackId()));
        }

    if (!options.output_path.empty() && !cv::imwrite(options.output_path, image)) {
        std::cerr << "Failed to write output image: " << options.output_path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "tracks: " << active_tracks.size() << std::endl;
    return EXIT_SUCCESS;
}

template <typename TrackerT>
int run_botsort_video(TrackerT& tracker,
                      const Options& options,
                      const std::unordered_map<int, std::vector<DetectionRecord>>& detection_records,
                      PersonDetector* detector,
                      PersonFeatureExtractor& extractor)
{
    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    const float reid_match_threshold = params_config->getKeyValue<float>("person_similarity_threshold");
    const size_t gallery_history = 10;
    const size_t visual_lost_window = 3;
    const size_t recovery_window = (options.recovery_window > 0)
        ? static_cast<size_t>(options.recovery_window)
        : static_cast<size_t>(params_config->getKeyValue<int>("botsort_recovery_window"));
    const size_t recovery_persist_frames = std::max<size_t>(
        1,
        static_cast<size_t>(params_config->getKeyValue<int>("reid_recovery_persist_frames")));
    const size_t max_active_age_frames = std::max<size_t>(
        1,
        static_cast<size_t>(params_config->getKeyValue<int>("reid_recovery_max_active_age_frames")));

    byte_track::ReIDRecovery reid_recovery;
    reid_recovery.resetStats();

    cv::VideoCapture cap(options.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << options.input_path << std::endl;
        return EXIT_FAILURE;
    }

    cv::VideoWriter writer;
    std::ofstream dump_file;
    std::ofstream debug_file;
    if (!options.dump_tracks_path.empty()) {
        dump_file.open(options.dump_tracks_path);
        dump_file << "frame,track_id,x1,y1,x2,y2,score\n";
    }
    if (!options.dump_debug_path.empty()) {
        debug_file.open(options.dump_debug_path);
    }
    int frame_index = 0;
    auto last_tick = std::chrono::steady_clock::now();
    double fps_ema = 0.0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        const auto objects = options.detections_path.empty()
            ? detect_frame(*detector, frame)
            : load_frame_detections_from_file(detection_records, frame_index);
        auto detections = build_detection_infos(objects);

        std::vector<byte_track::Object> tracker_objects;
        tracker_objects.reserve(detections.size());
        for (const auto& detection : detections) {
            tracker_objects.push_back(detection.object);
        }

        run_tracker_update(tracker, tracker_objects, frame);
        auto active_tracks = collect_active_tracks(tracker.getTrackedStracks());
        const auto active_match_infos = associate_tracks_to_detections(active_tracks, detections, 0.1f);
        std::unordered_map<size_t, size_t> active_track_ages;
        active_track_ages.reserve(active_tracks.size());
        for (const auto& track : active_tracks) {
            const size_t age = (track->getFrameId() >= track->getStartFrameId())
                ? track->getFrameId() - track->getStartFrameId()
                : 0;
            active_track_ages[track->getTrackId()] = age;
        }

        if (options.enable_reid_recovery && !active_tracks.empty()) {
            std::vector<std::array<float, 4>> track_boxes;
            track_boxes.reserve(active_tracks.size());
            for (const auto& track : active_tracks) {
                track_boxes.push_back(rect_to_xyxy(track->getRect()));
            }
            const auto track_features = extractor.extract(frame, track_boxes);
            for (size_t i = 0; i < active_tracks.size() && i < track_features.size(); ++i) {
                if (!track_features[i].empty()) {
                    reid_recovery.updateGallery(active_tracks[i]->getTrackId(), track_features[i], gallery_history);
                }
            }
        }

        size_t recovered_count = 0;
        std::set<size_t> suppressed_active_ids;
        if (options.enable_reid_recovery && !tracker.getLostStracks().empty() && !detections.empty()) {
            std::vector<std::array<float, 4>> det_boxes;
            det_boxes.reserve(detections.size());
            for (const auto& detection : detections) {
                det_boxes.push_back(detection.box);
            }

            size_t crop_invalid = 0;
            for (const auto& box : det_boxes) {
                const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(box[0]))));
                const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(box[1]))));
                const int x2 = std::max(0, std::min(frame.cols, static_cast<int>(std::round(box[2]))));
                const int y2 = std::max(0, std::min(frame.rows, static_cast<int>(std::round(box[3]))));
                if (x2 <= x1 || y2 <= y1) {
                    ++crop_invalid;
                }
            }
            const auto det_features = extractor.extract(frame, det_boxes);
            size_t feature_count_mismatch = 0;
            if (det_features.size() != det_boxes.size()) {
                feature_count_mismatch = det_boxes.size() > det_features.size()
                    ? det_boxes.size() - det_features.size()
                    : det_features.size() - det_boxes.size();
            }
            size_t extract_empty = 0;
            for (const auto& feature : det_features) {
                if (feature.empty()) {
                    ++extract_empty;
                }
            }
            reid_recovery.recordDetectionFeatureStats(crop_invalid, extract_empty, feature_count_mismatch);
            attach_features_to_detections(detections, det_features);
            std::vector<byte_track::Object> det_objects;
            std::vector<std::array<float, 4>> det_boxes_with_features;
            std::vector<std::vector<float>> det_feature_vectors;
            det_objects.reserve(detections.size());
            det_boxes_with_features.reserve(detections.size());
            det_feature_vectors.reserve(detections.size());
            for (const auto& detection : detections) {
                det_objects.push_back(detection.object);
                det_boxes_with_features.push_back(detection.box);
                det_feature_vectors.push_back(detection.feature);
            }
            const auto override_decisions = reid_recovery.recoverFromLowQualityActiveMatches(
                tracker.getLostStracks(),
                det_objects,
                det_boxes_with_features,
                det_feature_vectors,
                active_match_infos,
                active_track_ages,
                tracker.getFrameId(),
                recovery_window,
                std::min(reid_match_threshold, 0.7f),
                recovery_persist_frames,
                max_active_age_frames,
                0.5f,
                [&](const STrackPtr& track, const byte_track::Object& object) {
                    return tracker.recoverTrack(track, object);
                });
            for (const auto& decision : override_decisions) {
                suppressed_active_ids.insert(decision.active_track_id);
            }
            std::vector<byte_track::ReIDRecovery::ActiveMatch> active_matches;
            for (const auto& match : active_match_infos) {
                if (suppressed_active_ids.count(match.track_id) == 0) {
                    active_matches.push_back(match);
                }
            }
            recovered_count = reid_recovery.recoverLostTracks(
                tracker.getLostStracks(),
                det_objects,
                det_boxes_with_features,
                det_feature_vectors,
                active_matches,
                tracker.getFrameId(),
                recovery_window,
                reid_match_threshold,
                gallery_history,
                [&](const STrackPtr& track, const byte_track::Object& object) {
                    return tracker.recoverTrack(track, object);
                });
        }
        active_tracks = collect_active_tracks(tracker.getTrackedStracks());
        if (!suppressed_active_ids.empty()) {
            active_tracks.erase(
                std::remove_if(active_tracks.begin(), active_tracks.end(), [&](const STrackPtr& track) {
                    return suppressed_active_ids.count(track->getTrackId()) > 0;
                }),
                active_tracks.end());
        }
        std::vector<STrackPtr> recent_lost;
        if (options.show_lost_tracks) {
            recent_lost = collect_recent_lost_tracks(tracker.getLostStracks(),
                                                     tracker.getFrameId(),
                                                     visual_lost_window);
        }

        cv::Mat vis_frame = frame.clone();
        for (const auto& track : active_tracks) {
            draw_track(vis_frame, track, color_for_track_id(track->getTrackId()));
        }
        if (options.show_lost_tracks) {
            draw_tracks(vis_frame, recent_lost, cv::Scalar(180, 180, 180), "L");
        }

        const auto now_tick = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now_tick - last_tick).count();
        last_tick = now_tick;
        if (dt > 1e-6) {
            const double fps_now = 1.0 / dt;
            fps_ema = (fps_ema <= 0.0) ? fps_now : (0.9 * fps_ema + 0.1 * fps_now);
        }

        const std::string status = cv::format("Frame: %d | Tracks: %zu | Lost: %zu | Recovered: %zu | FPS: %.1f",
                                              frame_index,
                                              active_tracks.size(),
                                              recent_lost.size(),
                                              recovered_count,
                                              fps_ema);
        cv::putText(vis_frame,
                    status,
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(50, 255, 50),
                    2,
                    cv::LINE_AA);

        if (!write_or_open_video(writer, options, cap, vis_frame)) {
            return EXIT_FAILURE;
        }
        if (dump_file.is_open()) {
            dump_tracks(dump_file, frame_index, active_tracks);
        }
        if (debug_file.is_open()) {
            const auto dump_lost = collect_lost_tracks(tracker.getLostStracks());
            dump_debug(debug_file, frame_index, detections, active_tracks, dump_lost, recovered_count);
            const auto tracker_debug = get_tracker_internal_debug(tracker);
            if (!tracker_debug.empty()) debug_file << "{\"frame\":" << frame_index << ",\"tracker_debug\":" << tracker_debug << "}\n";
        }

        std::cout << "frame " << frame_index
                  << ", inputs=" << detections.size()
                  << ", tracks=" << active_tracks.size()
                  << ", lost=" << recent_lost.size()
                  << ", recovered=" << recovered_count
                  << std::endl;
        ++frame_index;
        if (options.max_frames > 0 && frame_index >= options.max_frames) {
            break;
        }
    }

    const auto& reid_stats = reid_recovery.getStats();
    std::cout << "reid_stats"
              << " recovered_success=" << reid_stats.recovered_success
              << " over_window=" << reid_stats.over_window
              << " track_feature_empty=" << reid_stats.track_feature_empty
              << " det_feature_empty=" << reid_stats.det_feature_empty
              << " det_crop_invalid=" << reid_stats.det_crop_invalid
              << " det_extract_empty=" << reid_stats.det_extract_empty
              << " det_feature_count_mismatch=" << reid_stats.det_feature_count_mismatch
              << " det_used_by_active=" << reid_stats.det_used_by_active
              << " det_feature_missing=" << reid_stats.det_feature_missing
              << " similarity_fail=" << reid_stats.similarity_fail
              << " geometry_fail=" << reid_stats.geometry_fail
              << " conflict_fail=" << reid_stats.conflict_fail
              << std::endl;

    std::vector<std::pair<size_t, byte_track::ReIDRecovery::FrameStats>> frame_stats(
        reid_recovery.getFrameStats().begin(), reid_recovery.getFrameStats().end());
    std::sort(frame_stats.begin(), frame_stats.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second.det_used_by_active > rhs.second.det_used_by_active;
    });
    const size_t top_n = std::min<size_t>(10, frame_stats.size());
    for (size_t i = 0; i < top_n; ++i) {
        const auto& item = frame_stats[i];
        std::cout << "reid_frame"
                  << " frame=" << item.first
                  << " det_used_by_active=" << item.second.det_used_by_active
                  << " similarity_fail=" << item.second.similarity_fail
                  << " over_window=" << item.second.over_window
                  << " active_track_ids=";
        bool first_active = true;
        for (size_t track_id : item.second.active_track_ids) {
            if (!first_active) std::cout << ",";
            std::cout << track_id;
            first_active = false;
        }
        std::cout << " used_det_indices=";
        bool first_det = true;
        for (size_t det_index : item.second.used_det_indices) {
            if (!first_det) std::cout << ",";
            std::cout << det_index;
            first_det = false;
        }
        std::cout << " active_det_ious=";
        bool first_iou = true;
        for (const auto& [det_index, iou] : item.second.active_det_ious) {
            if (!first_iou) std::cout << ",";
            std::cout << det_index << ":" << iou;
            first_iou = false;
        }
        std::cout << " impacted_tracks=";
        bool first = true;
        for (size_t track_id : item.second.impacted_tracks) {
            if (!first) std::cout << ",";
            std::cout << track_id;
            first = false;
        }
        std::cout << " blocked_candidates=";
        bool first_blocked = true;
        for (const auto& blocked : item.second.blocked_candidates) {
            if (!first_blocked) std::cout << ";";
            std::cout << blocked.track_id << "->det" << blocked.det_index
                      << "@sim=" << blocked.similarity
                      << ",iou=" << blocked.iou
                      << ",cr=" << blocked.center_ratio;
            first_blocked = false;
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv)
{
    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    Options options;
    if (!parse_args(argc, argv, options)) {
        return options.input_path.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    std::cout << "Tracker: " << tracker_type_name(options.tracker_type) << std::endl;

    const auto detection_records = load_detection_file(options.detections_path);

    std::unique_ptr<PersonDetector> detector;
    if (options.detections_path.empty()) {
        detector = std::make_unique<PersonDetector>();
        if (!detector->initialize(options.model_path)) {
            return EXIT_FAILURE;
        }
        std::cout << "No detection file provided, using PersonDetector online." << std::endl;
    }

    PersonFeatureExtractor extractor;
    if (!extractor.initialize()) {
        return EXIT_FAILURE;
    }

    if (options.tracker_type == TrackerType::ByteTrack)
    {
        const int track_buffer = (options.track_buffer > 0)
            ? options.track_buffer
            : params_config->getKeyValue<int>("bytetrack_track_buffer");
        const float track_thresh = (options.track_thresh > 0.0f)
            ? options.track_thresh
            : params_config->getKeyValue<float>("bytetrack_track_thresh");
        const float high_thresh = (options.high_thresh > 0.0f)
            ? options.high_thresh
            : params_config->getKeyValue<float>("bytetrack_high_thresh");
        const float new_track_thresh = params_config->getKeyValue<float>("bytetrack_new_track_thresh");
        const float match_thresh = (options.match_thresh > 0.0f)
            ? options.match_thresh
            : params_config->getKeyValue<float>("bytetrack_match_thresh");
        byte_track::BYTETracker tracker(options.frame_rate, track_buffer,
                                        track_thresh, high_thresh, new_track_thresh,
                                        match_thresh);
        if (module_test::is_image_file(options.input_path)) {
            return run_botsort_image(tracker, options, detection_records, detector.get(), extractor);
        }
        return run_botsort_video(tracker, options, detection_records, detector.get(), extractor);
    }

    if (options.tracker_type == TrackerType::OAByteTrack)
    {
        const int track_buffer = (options.track_buffer > 0)
            ? options.track_buffer
            : params_config->getKeyValue<int>("oabytetrack_track_buffer");
        const float track_thresh = (options.track_thresh > 0.0f)
            ? options.track_thresh
            : params_config->getKeyValue<float>("oabytetrack_track_thresh");
        const float high_thresh = (options.high_thresh > 0.0f)
            ? options.high_thresh
            : params_config->getKeyValue<float>("oabytetrack_high_thresh");
        const float new_track_thresh = params_config->getKeyValue<float>("oabytetrack_new_track_thresh");
        const float match_thresh = (options.match_thresh > 0.0f)
            ? options.match_thresh
            : params_config->getKeyValue<float>("oabytetrack_match_thresh");
        const bool use_oao = params_config->getKeyValue<int>("oabytetrack_use_oao") != 0;
        const bool use_bam = params_config->getKeyValue<int>("oabytetrack_use_bam") != 0;
        byte_track::OAByteTracker tracker(options.frame_rate, track_buffer,
                                          track_thresh, high_thresh, new_track_thresh, match_thresh,
                                          use_oao, use_bam);
        if (module_test::is_image_file(options.input_path)) {
            return run_botsort_image(tracker, options, detection_records, detector.get(), extractor);
        }
        return run_botsort_video(tracker, options, detection_records, detector.get(), extractor);
    }

    if (options.tracker_type == TrackerType::BoTSORT)
    {
        const int botsort_track_buffer = (options.track_buffer > 0)
            ? options.track_buffer
            : params_config->getKeyValue<int>("botsort_track_buffer");
        const float botsort_track_thresh = (options.track_thresh > 0.0f)
            ? options.track_thresh
            : params_config->getKeyValue<float>("botsort_track_thresh");
        const float botsort_high_thresh = (options.high_thresh > 0.0f)
            ? options.high_thresh
            : params_config->getKeyValue<float>("botsort_high_thresh");
        const float botsort_new_track_thresh = params_config->getKeyValue<float>("botsort_new_track_thresh");
        const float botsort_match_thresh = (options.match_thresh > 0.0f)
            ? options.match_thresh
            : params_config->getKeyValue<float>("botsort_match_thresh");
        options.gmc_method = static_cast<bot_sort::GMCMethod>(
            params_config->getKeyValue<int>("botsort_gmc_method"));
        bot_sort::BoTSORT tracker(botsort_high_thresh, botsort_track_thresh,
                                  botsort_new_track_thresh, botsort_match_thresh,
                                  botsort_track_buffer, options.gmc_method);
        if (module_test::is_image_file(options.input_path)) {
            return run_botsort_image(tracker, options, detection_records, detector.get(), extractor);
        }
        return run_botsort_video(tracker, options, detection_records, detector.get(), extractor);
    }

    const int oabotsort_track_buffer = (options.track_buffer > 0)
        ? options.track_buffer
        : params_config->getKeyValue<int>("oabotsort_track_buffer");
    const float oabotsort_track_thresh = (options.track_thresh > 0.0f)
        ? options.track_thresh
        : params_config->getKeyValue<float>("oabotsort_track_thresh");
    const float oabotsort_high_thresh = (options.high_thresh > 0.0f)
        ? options.high_thresh
        : params_config->getKeyValue<float>("oabotsort_high_thresh");
    const float oabotsort_new_track_thresh = params_config->getKeyValue<float>("oabotsort_new_track_thresh");
    const float oabotsort_match_thresh = (options.match_thresh > 0.0f)
        ? options.match_thresh
        : params_config->getKeyValue<float>("oabotsort_match_thresh");
    options.gmc_method = static_cast<bot_sort::GMCMethod>(
        params_config->getKeyValue<int>("oabotsort_gmc_method"));
    options.use_oao = params_config->getKeyValue<int>("oabotsort_use_oao") != 0;
    options.use_bam = params_config->getKeyValue<int>("oabotsort_use_bam") != 0;
    bot_sort::OABoTSORT tracker(oabotsort_high_thresh, oabotsort_track_thresh,
                                oabotsort_new_track_thresh, oabotsort_match_thresh,
                                oabotsort_track_buffer, options.gmc_method,
                                2, options.use_oao, options.use_bam);
    if (module_test::is_image_file(options.input_path)) {
        return run_botsort_image(tracker, options, detection_records, detector.get(), extractor);
    }
    return run_botsort_video(tracker, options, detection_records, detector.get(), extractor);
}
