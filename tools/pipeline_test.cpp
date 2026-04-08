#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "gallery_manager.hpp"
#include "detector.hpp"
#include "module_test_utils.hpp"
#include "personfeature_extractor.hpp"
#include "tracker/BYTETracker.h"
#include "tracker/BoTSORT.h"
#include "tracker/Object.h"

namespace
{
enum class TrackerType
{
    ByteTrack,
    BoTSORT,
    OABoTSORT
};

using STrackPtr = std::shared_ptr<byte_track::STrack>;

struct Options
{
    std::string input_path;
    std::string output_path;
    std::string detector_model_path;
    std::string reid_model_path;
    std::string query_image_path;
    TrackerType tracker_type = TrackerType::ByteTrack;
    int frame_rate = 30;
    int track_buffer = 30;
    float track_thresh = 0.0f;  // loaded from config
    float high_thresh = 0.0f;   // loaded from config
    float match_thresh = 0.0f;  // loaded from config
    bot_sort::GMCMethod gmc_method = bot_sort::GMCMethod::ORB;
    bool use_oao = true;
    bool use_bam = true;
};

struct DetectionWithFeature
{
    std::array<float, 4> box {};
    float score = 0.0f;
    std::vector<float> feature;
    cv::Mat thumbnail;
};

struct TrackDetectionMatch
{
    int detection_index = -1;
    float iou = 0.0f;
};

void print_usage()
{
    std::cout
        << "Usage: pipeline_test --input <image|video> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --tracker <type>       Tracker: bytetrack (default), botsort, oabotsort\n"
        << "  --output <path>        Output image/video path\n"
        << "  --query <person_crop>  Query person image for ReID\n"
        << "  --det-model <rknn>     Detection model path\n"
        << "  --reid-model <rknn>    ReID model path\n"
        << "  --fps <n>              Frame rate (default: 30)\n"
        << "  --track-buffer <n>     Track buffer / max lost frames (default: 30)\n"
        << "  --track-thresh <f>     Track confidence threshold (default: 0.5)\n"
        << "  --high-thresh <f>      High confidence threshold (default: 0.6)\n"
        << "  --match-thresh <f>     Match threshold (default: 0.8)\n"
        << "  --gmc-method <type>    GMC method: orb (default), sparse, none\n"
        << "  --no-oao               Disable OAO for OABoTSORT\n"
        << "  --no-bam               Disable BAM for OABoTSORT\n";
}

TrackerType parse_tracker_type(const std::string& s)
{
    if (s == "botsort") return TrackerType::BoTSORT;
    if (s == "oabotsort") return TrackerType::OABoTSORT;
    return TrackerType::ByteTrack;
}

std::string tracker_type_name(TrackerType t)
{
    switch (t) {
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
        } else if (arg == "--query" && i + 1 < argc) {
            options.query_image_path = argv[++i];
        } else if (arg == "--det-model" && i + 1 < argc) {
            options.detector_model_path = argv[++i];
        } else if (arg == "--reid-model" && i + 1 < argc) {
            options.reid_model_path = argv[++i];
        } else if (arg == "--tracker" && i + 1 < argc) {
            options.tracker_type = parse_tracker_type(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            options.frame_rate = std::stoi(argv[++i]);
        } else if (arg == "--track-buffer" && i + 1 < argc) {
            options.track_buffer = std::stoi(argv[++i]);
        } else if (arg == "--track-thresh" && i + 1 < argc) {
            options.track_thresh = std::stof(argv[++i]);
        } else if (arg == "--high-thresh" && i + 1 < argc) {
            options.high_thresh = std::stof(argv[++i]);
        } else if (arg == "--match-thresh" && i + 1 < argc) {
            options.match_thresh = std::stof(argv[++i]);
        } else if (arg == "--gmc-method" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "sparse") options.gmc_method = bot_sort::GMCMethod::SparseOptFlow;
            else if (v == "none") options.gmc_method = bot_sort::GMCMethod::None;
            else options.gmc_method = bot_sort::GMCMethod::ORB;
        } else if (arg == "--no-oao") {
            options.use_oao = false;
        } else if (arg == "--no-bam") {
            options.use_bam = false;
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

std::optional<DetectionWithFeature> extract_feature_from_box(PersonFeatureExtractor& extractor,
                                                             const cv::Mat& frame,
                                                             const std::array<float, 4>& box,
                                                             float score)
{
    const cv::Rect roi = module_test::clamp_xyxy_to_rect(box, frame.size());
    if (roi.width <= 1 || roi.height <= 1) {
        return std::nullopt;
    }

    DetectionWithFeature item;
    item.box = box;
    item.score = score;
    cv::Mat crop = frame(roi).clone();
    item.feature = extractor.extractFeature(crop, item.thumbnail);
    return item;
}

std::vector<DetectionWithFeature> detect_and_extract(PersonDetector& detector,
                                                     PersonFeatureExtractor& extractor,
                                                     const cv::Mat& frame)
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

    std::vector<DetectionWithFeature> detections;
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const auto mapped_box = module_test::map_box_to_original(result.boxes[i], letterbox_info, frame.cols, frame.rows);
        auto item = extract_feature_from_box(extractor, frame, mapped_box, result.scores[i]);
        if (item.has_value()) {
            detections.push_back(std::move(item.value()));
        }
    }
    return detections;
}

std::vector<byte_track::Object> to_tracker_objects(const std::vector<DetectionWithFeature>& detections)
{
    std::vector<byte_track::Object> objects;
    objects.reserve(detections.size());
    for (const auto& detection : detections) {
        const float width = std::max(0.0f, detection.box[2] - detection.box[0]);
        const float height = std::max(0.0f, detection.box[3] - detection.box[1]);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }
        objects.emplace_back(byte_track::Rect<float>(detection.box[0], detection.box[1], width, height), 0, detection.score);
    }
    return objects;
}

std::unordered_map<size_t, TrackDetectionMatch> associate_tracks_to_detections(
    const std::vector<STrackPtr>& tracks,
    const std::vector<DetectionWithFeature>& detections)
{
    std::unordered_map<size_t, TrackDetectionMatch> matches;
    std::vector<bool> used_detection(detections.size(), false);

    for (const auto& track : tracks) {
        const auto& rect = track->getRect();
        const std::array<float, 4> track_box = {
            rect.x(),
            rect.y(),
            rect.x() + rect.width(),
            rect.y() + rect.height()
        };

        int best_detection = -1;
        float best_iou = 0.0f;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (used_detection[i]) {
                continue;
            }
            const float iou = compute_iou(track_box, detections[i].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_detection = static_cast<int>(i);
            }
        }

        if (best_detection >= 0 && best_iou > 0.3f) {
            used_detection[best_detection] = true;
            matches[track->getTrackId()] = {best_detection, best_iou};
        }
    }

    return matches;
}

void draw_tracks(cv::Mat& frame,
                 const std::vector<STrackPtr>& tracks,
                 const std::unordered_map<size_t, TrackDetectionMatch>& track_matches,
                 const std::vector<DetectionWithFeature>& detections,
                 int target_track_id,
                 float target_similarity,
                 int lost_count)
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

        const bool is_target = static_cast<int>(track->getTrackId()) == target_track_id;
        const cv::Scalar color = is_target ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 255);
        cv::rectangle(frame, draw_rect, color, is_target ? 3 : 2);

        std::string label = "ID " + std::to_string(track->getTrackId());
        const auto match_it = track_matches.find(track->getTrackId());
        if (match_it != track_matches.end() && match_it->second.detection_index >= 0) {
            const float score = detections[match_it->second.detection_index].score;
            label += cv::format(" %.2f", score);
        }
        if (is_target) {
            label += cv::format(" target %.2f", target_similarity);
        }

        cv::putText(frame,
                    label,
                    cv::Point(draw_rect.x, std::max(20, draw_rect.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2);
    }

    const std::string status = "target_id=" + std::to_string(target_track_id) +
        cv::format(" sim=%.3f lost=%d", target_similarity, lost_count);
    cv::putText(frame, status, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(50, 255, 50), 2);
}

// ---- Tracker helpers ----
std::vector<STrackPtr> run_tracker(
    TrackerType type,
    byte_track::BYTETracker& bt,
    bot_sort::BoTSORT& bs,
    const std::vector<byte_track::Object>& objects,
    const cv::Mat& frame)
{
    if (type == TrackerType::ByteTrack) {
        return bt.update(objects);
    }
    return bs.update(objects, frame);
}

}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_args(argc, argv, options)) {
        return options.input_path.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    std::cout << "Tracker: " << tracker_type_name(options.tracker_type) << std::endl;

    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    const float reid_match_threshold = params_config->getKeyValue<float>("person_similarity_threshold");
    const float gallery_save_threshold = params_config->getKeyValue<float>("gallery_similarity_save_threshold");

    PersonDetector detector;
    if (!detector.initialize(options.detector_model_path)) {
        return EXIT_FAILURE;
    }

    PersonFeatureExtractor extractor;
    if (!extractor.initialize(options.reid_model_path)) {
        return EXIT_FAILURE;
    }

    GalleryManager gallery;
    gallery.reset_gallery("pipeline");
    bool gallery_seeded = false;

    if (!options.query_image_path.empty()) {
        cv::Mat query_image = cv::imread(options.query_image_path);
        if (query_image.empty()) {
            std::cerr << "Failed to read query image: " << options.query_image_path << std::endl;
            return EXIT_FAILURE;
        }
        cv::Mat thumbnail;
        auto query_feature = extractor.extractFeature(query_image, thumbnail);
        gallery.save_to_gallery(query_feature, thumbnail, "pipeline");
        gallery_seeded = true;
        std::cout << "Loaded query feature from: " << options.query_image_path << std::endl;
    }

    // Load tracker thresholds from config (can be overridden by CLI args)
    if (options.track_thresh <= 0.0f)
        options.track_thresh = params_config->getKeyValue<float>("track_thresh");
    if (options.high_thresh <= 0.0f)
        options.high_thresh = params_config->getKeyValue<float>("high_thresh");
    if (options.match_thresh <= 0.0f)
        options.match_thresh = params_config->getKeyValue<float>("match_thresh");

    byte_track::BYTETracker bt(options.frame_rate, options.track_buffer,
                                          options.track_thresh, options.high_thresh,
                                          options.match_thresh);
    bot_sort::OABoTSORT bs(options.high_thresh, options.track_thresh,
                            options.high_thresh, options.match_thresh,
                            options.track_buffer, options.gmc_method,
                            2, options.use_oao, options.use_bam);
    int target_track_id = -1;
    int lost_count = 0;
    float target_similarity = 0.0f;

    auto process_frame = [&](const cv::Mat& input_frame, cv::Mat& vis_frame, int frame_index) -> bool {
        const auto detections = detect_and_extract(detector, extractor, input_frame);
        const auto objects = to_tracker_objects(detections);
        const auto tracks = run_tracker(options.tracker_type, bt, bs, objects, input_frame);
        const auto track_matches = associate_tracks_to_detections(tracks, detections);

        vis_frame = input_frame.clone();
        target_similarity = 0.0f;

        if (target_track_id >= 0) {
            bool target_found = false;
            for (const auto& track : tracks) {
                if (static_cast<int>(track->getTrackId()) != target_track_id) {
                    continue;
                }

                target_found = true;
                lost_count = 0;
                const auto match_it = track_matches.find(track->getTrackId());
                if (match_it != track_matches.end()) {
                    const auto& detection = detections[match_it->second.detection_index];
                    target_similarity = gallery.get_max_similarity(detection.feature);
                    if (target_similarity >= gallery_save_threshold) {
                        gallery.save_to_gallery(detection.feature, detection.thumbnail, "pipeline");
                    }
                }
                break;
            }

            if (!target_found) {
                ++lost_count;
            }
        }

        if (target_track_id < 0 || lost_count > 0) {
            int best_track_id = -1;
            float best_similarity = 0.0f;

            for (const auto& track : tracks) {
                const auto match_it = track_matches.find(track->getTrackId());
                if (match_it == track_matches.end()) {
                    continue;
                }

                const auto& detection = detections[match_it->second.detection_index];
                const float similarity = gallery.get_max_similarity(detection.feature);
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_track_id = static_cast<int>(track->getTrackId());
                }
            }

            if (best_track_id >= 0 && best_similarity >= reid_match_threshold) {
                target_track_id = best_track_id;
                target_similarity = best_similarity;
                lost_count = 0;
            }
        }

        if (!gallery_seeded && target_track_id < 0 && options.query_image_path.empty() && !detections.empty()) {
            const auto best_it = std::max_element(detections.begin(), detections.end(), [](const DetectionWithFeature& lhs, const DetectionWithFeature& rhs) {
                return lhs.score < rhs.score;
            });
            if (best_it != detections.end()) {
                gallery.save_to_gallery(best_it->feature, best_it->thumbnail, "pipeline");
                gallery_seeded = true;

                int best_track_id = -1;
                float best_iou = 0.0f;
                for (const auto& track : tracks) {
                    const auto& rect = track->getRect();
                    const std::array<float, 4> track_box = {
                        rect.x(),
                        rect.y(),
                        rect.x() + rect.width(),
                        rect.y() + rect.height()
                    };
                    const float iou = compute_iou(track_box, best_it->box);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_track_id = static_cast<int>(track->getTrackId());
                    }
                }

                if (best_track_id >= 0) {
                    target_track_id = best_track_id;
                    target_similarity = 1.0f;
                    lost_count = 0;
                    std::cout << "frame " << frame_index << ": initialized target track id " << target_track_id << " from top-score detection" << std::endl;
                }
            }
        }

        draw_tracks(vis_frame, tracks, track_matches, detections, target_track_id, target_similarity, lost_count);
        std::cout << "frame " << frame_index
                  << " detections=" << detections.size()
                  << " tracks=" << tracks.size()
                  << " target_id=" << target_track_id
                  << " target_similarity=" << target_similarity
                  << " lost=" << lost_count
                  << std::endl;
        return true;
    };

    if (module_test::is_image_file(options.input_path)) {
        cv::Mat image = cv::imread(options.input_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << options.input_path << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat vis_image;
        if (!process_frame(image, vis_image, 0)) {
            return EXIT_FAILURE;
        }

        if (!options.output_path.empty() && !cv::imwrite(options.output_path, vis_image)) {
            std::cerr << "Failed to write output image: " << options.output_path << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    cv::VideoCapture cap(options.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << options.input_path << std::endl;
        return EXIT_FAILURE;
    }

    cv::VideoWriter writer;
    const bool write_video = !options.output_path.empty();
    int frame_index = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cv::Mat vis_frame;
        if (!process_frame(frame, vis_frame, frame_index)) {
            return EXIT_FAILURE;
        }

        if (write_video) {
            if (!writer.isOpened()) {
                const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : static_cast<double>(options.frame_rate);
                writer.open(options.output_path,
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps,
                            vis_frame.size());
                if (!writer.isOpened()) {
                    std::cerr << "Failed to open output video: " << options.output_path << std::endl;
                    return EXIT_FAILURE;
                }
            }
            writer.write(vis_frame);
        }

        ++frame_index;
    }

    return EXIT_SUCCESS;
}
