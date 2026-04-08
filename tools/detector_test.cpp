#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector.hpp"
#include "module_test_utils.hpp"

namespace
{
struct Options
{
    std::string input_path;
    std::string output_path;
    std::string model_path;
};

void print_usage()
{
    std::cout
        << "Usage: detector_test --input <image|video> [--output <path>] [--model <rknn>]\n";
}

bool parse_args(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            options.model_path = argv[++i];
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

void draw_detection(cv::Mat& frame, const std::array<float, 4>& box, float score)
{
    const cv::Rect rect = module_test::clamp_xyxy_to_rect(box, frame.size());
    if (rect.width <= 0 || rect.height <= 0) {
        return;
    }

    cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
    const std::string label = cv::format("person %.3f", score);
    const int text_y = std::max(20, rect.y - 8);
    cv::putText(frame, label, cv::Point(rect.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

bool run_detector_on_frame(PersonDetector& detector, const cv::Mat& frame, cv::Mat& vis_frame)
{
    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    const int input_width = params_config->getKeyValue<int>("detect_image_width");
    const int input_height = params_config->getKeyValue<int>("detect_image_height");

    module_test::LetterboxInfo letterbox_info;
    cv::Mat detector_input;
    if (!module_test::resize_with_letterbox_rgb(frame, detector_input, cv::Size(input_width, input_height), letterbox_info)) {
        return false;
    }

    PersonDetector::DetectionResult result;
    detector.detect(detector_input, result);
    vis_frame = frame.clone();

    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const auto mapped_box = module_test::map_box_to_original(result.boxes[i], letterbox_info, frame.cols, frame.rows);
        draw_detection(vis_frame, mapped_box, result.scores[i]);
    }

    std::cout << "detections: " << result.boxes.size() << std::endl;
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_args(argc, argv, options)) {
        return options.input_path.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    PersonDetector detector;
    if (!detector.initialize(options.model_path)) {
        return EXIT_FAILURE;
    }

    if (module_test::is_image_file(options.input_path)) {
        cv::Mat image = cv::imread(options.input_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << options.input_path << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat vis_image;
        if (!run_detector_on_frame(detector, image, vis_image)) {
            return EXIT_FAILURE;
        }

        if (!options.output_path.empty()) {
            if (!cv::imwrite(options.output_path, vis_image)) {
                std::cerr << "Failed to write output image: " << options.output_path << std::endl;
                return EXIT_FAILURE;
            }
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
        if (!run_detector_on_frame(detector, frame, vis_frame)) {
            return EXIT_FAILURE;
        }

        if (write_video) {
            if (!writer.isOpened()) {
                const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
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

        std::cout << "frame " << frame_index++ << " processed" << std::endl;
    }

    return EXIT_SUCCESS;
}
