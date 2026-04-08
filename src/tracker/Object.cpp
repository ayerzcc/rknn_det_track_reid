#include "tracker/Object.h"

byte_track::Object::Object(const Rect<float> &_rect,
                           const int &_label,
                           const float &_prob,
                           const std::vector<float>& _feature) : rect(_rect), label(_label), prob(_prob), feature(_feature)
{
}
