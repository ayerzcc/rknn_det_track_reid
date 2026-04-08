#pragma once

#include "tracker/Rect.h"
#include <vector>

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int label;
    float prob;
    std::vector<float> feature;

    Object(const Rect<float> &_rect,
           const int &_label,
           const float &_prob,
           const std::vector<float>& _feature = {});
};
}
