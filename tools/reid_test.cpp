#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "module_test_utils.hpp"
#include "personfeature_extractor.hpp"

namespace
{
struct Options
{
    std::string query_path;
    std::string candidate_path;
    std::string gallery_dir;
    std::string model_path;
};

struct MatchResult
{
    std::string path;
    float similarity = 0.0f;
};

void print_usage()
{
    std::cout
        << "Usage: reid_test --query <image> [--candidate <image>] [--gallery <dir>] [--model <rknn>]\n";
}

bool parse_args(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--query" && i + 1 < argc) {
            options.query_path = argv[++i];
        } else if (arg == "--candidate" && i + 1 < argc) {
            options.candidate_path = argv[++i];
        } else if (arg == "--gallery" && i + 1 < argc) {
            options.gallery_dir = argv[++i];
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

    if (options.query_path.empty() || (options.candidate_path.empty() && options.gallery_dir.empty())) {
        print_usage();
        return false;
    }
    return true;
}

std::vector<float> extract_feature(PersonFeatureExtractor& extractor, const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    cv::Mat thumbnail;
    return extractor.extractFeature(image, thumbnail);
}
}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_args(argc, argv, options)) {
        return options.query_path.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    PersonFeatureExtractor extractor;
    if (!extractor.initialize(options.model_path)) {
        return EXIT_FAILURE;
    }

    const auto query_feature = extract_feature(extractor, options.query_path);

    if (!options.candidate_path.empty()) {
        const auto candidate_feature = extract_feature(extractor, options.candidate_path);
        const float similarity = module_test::cosine_similarity(query_feature, candidate_feature);
        std::cout << "query: " << options.query_path << std::endl;
        std::cout << "candidate: " << options.candidate_path << std::endl;
        std::cout << "similarity: " << similarity << std::endl;
    }

    if (!options.gallery_dir.empty()) {
        std::vector<MatchResult> matches;
        for (const auto& entry : std::filesystem::directory_iterator(options.gallery_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string path = entry.path().string();
            if (!module_test::is_image_file(path)) {
                continue;
            }

            const auto gallery_feature = extract_feature(extractor, path);
            matches.push_back({path, module_test::cosine_similarity(query_feature, gallery_feature)});
        }

        std::sort(matches.begin(), matches.end(), [](const MatchResult& lhs, const MatchResult& rhs) {
            return lhs.similarity > rhs.similarity;
        });

        std::cout << "gallery results:" << std::endl;
        for (const auto& match : matches) {
            std::cout << match.path << " -> " << match.similarity << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
