#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace module_test
{
struct LetterboxInfo
{
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

inline bool resize_with_letterbox_rgb(const cv::Mat& src,
                                      cv::Mat& dst,
                                      const cv::Size& target_size,
                                      LetterboxInfo& info)
{
    if (src.empty() || target_size.width <= 0 || target_size.height <= 0) {
        return false;
    }

    dst = cv::Mat::zeros(target_size, CV_8UC3);
    info.scale = std::min(static_cast<float>(target_size.width) / static_cast<float>(src.cols),
                          static_cast<float>(target_size.height) / static_cast<float>(src.rows));

    const cv::Size scaled_size(static_cast<int>(std::round(src.cols * info.scale)),
                               static_cast<int>(std::round(src.rows * info.scale)));
    info.pad_x = (target_size.width - scaled_size.width) / 2;
    info.pad_y = (target_size.height - scaled_size.height) / 2;

    cv::Mat scaled;
    cv::resize(src, scaled, scaled_size, 0.0, 0.0, cv::INTER_LINEAR);
    scaled.copyTo(dst(cv::Rect(info.pad_x, info.pad_y, scaled_size.width, scaled_size.height)));
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    return true;
}

inline std::array<float, 4> map_box_to_original(const std::array<float, 4>& box,
                                                const LetterboxInfo& info,
                                                int image_width,
                                                int image_height)
{
    std::array<float, 4> mapped {};
    mapped[0] = std::clamp((box[0] - static_cast<float>(info.pad_x)) / info.scale, 0.0f, static_cast<float>(image_width));
    mapped[1] = std::clamp((box[1] - static_cast<float>(info.pad_y)) / info.scale, 0.0f, static_cast<float>(image_height));
    mapped[2] = std::clamp((box[2] - static_cast<float>(info.pad_x)) / info.scale, 0.0f, static_cast<float>(image_width));
    mapped[3] = std::clamp((box[3] - static_cast<float>(info.pad_y)) / info.scale, 0.0f, static_cast<float>(image_height));
    return mapped;
}

inline cv::Rect clamp_xyxy_to_rect(const std::array<float, 4>& box, const cv::Size& image_size)
{
    const int x1 = std::clamp(static_cast<int>(std::floor(box[0])), 0, image_size.width - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(box[1])), 0, image_size.height - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(box[2])), 0, image_size.width);
    const int y2 = std::clamp(static_cast<int>(std::ceil(box[3])), 0, image_size.height);
    const int width = std::max(0, x2 - x1);
    const int height = std::max(0, y2 - y1);
    return cv::Rect(x1, y1, width, height);
}

inline bool is_image_file(const std::string& path)
{
    const std::string ext = std::filesystem::path(path).extension().string();
    const std::string lower = [&ext]() {
        std::string value = ext;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }();

    return lower == ".jpg" || lower == ".jpeg" || lower == ".png" || lower == ".bmp" || lower == ".webp";
}

inline float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a <= 0.0f || norm_b <= 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}
}  // namespace module_test
