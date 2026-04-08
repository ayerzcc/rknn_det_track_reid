#include "tracker/Rect.h"

#include <algorithm>

template <typename T>
byte_track::Rect<T>::Rect(const T &x, const T &y, const T &width, const T &height) :
    tlwh({x, y, width, height})
{
}

template <typename T>
byte_track::Rect<T>::~Rect()
{
}

template <typename T>
const T& byte_track::Rect<T>::x() const
{
    return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::y() const
{
    return tlwh[1];
}

template <typename T>
const T& byte_track::Rect<T>::width() const
{
    return tlwh[2];
}

template <typename T>
const T& byte_track::Rect<T>::height() const
{
    return tlwh[3];
}

template <typename T>
T& byte_track::Rect<T>::x()
{
    return tlwh[0];
}

template <typename T>
T& byte_track::Rect<T>::y()
{
    return tlwh[1];
}

template <typename T>
T& byte_track::Rect<T>::width()
{
    return tlwh[2];
}

template <typename T>
T& byte_track::Rect<T>::height()
{
    return tlwh[3];
}

template <typename T>
const T& byte_track::Rect<T>::tl_x() const
{
    return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::tl_y() const
{
    return tlwh[1];
}

template <typename T>
T byte_track::Rect<T>::br_x() const
{
    return tlwh[0] + tlwh[2];
}

template <typename T>
T byte_track::Rect<T>::br_y() const
{
    return tlwh[1] + tlwh[3];
}

template <typename T>
byte_track::Tlbr<T> byte_track::Rect<T>::getTlbr() const
{
    return {
        tlwh[0],
        tlwh[1],
        tlwh[0] + tlwh[2],
        tlwh[1] + tlwh[3],
    };
}

template <typename T>
byte_track::Xyah<T> byte_track::Rect<T>::getXyah() const
{
    return {
        tlwh[0] + tlwh[2] / 2,
        tlwh[1] + tlwh[3] / 2,
        tlwh[2] / tlwh[3],
        tlwh[3],
    };
}

template<typename T>
float byte_track::Rect<T>::calcIoU(const Rect<T>& other) const
{
    const float xx1 = std::max(tlwh[0], other.tlwh[0]);
    const float yy1 = std::max(tlwh[1], other.tlwh[1]);
    const float xx2 = std::min(tlwh[0] + tlwh[2], other.tlwh[0] + other.tlwh[2]);
    const float yy2 = std::min(tlwh[1] + tlwh[3], other.tlwh[1] + other.tlwh[3]);

    const float iw = std::max(0.0f, xx2 - xx1);
    const float ih = std::max(0.0f, yy2 - yy1);
    const float inter = iw * ih;

    const float area_a = std::max(0.0f, static_cast<float>(tlwh[2])) * std::max(0.0f, static_cast<float>(tlwh[3]));
    const float area_b = std::max(0.0f, static_cast<float>(other.tlwh[2])) * std::max(0.0f, static_cast<float>(other.tlwh[3]));
    const float union_area = area_a + area_b - inter;
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }
    return inter / union_area;
}

template<typename T>
byte_track::Rect<T> byte_track::generate_rect_by_tlbr(const byte_track::Tlbr<T>& tlbr)
{
    return byte_track::Rect<T>(tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]);
}

template<typename T>
byte_track::Rect<T> byte_track::generate_rect_by_xyah(const byte_track::Xyah<T>& xyah)
{
    const auto width = xyah[2] * xyah[3];
    return byte_track::Rect<T>(xyah[0] - width / 2, xyah[1] - xyah[3] / 2, width, xyah[3]);
}

// explicit instantiation
template class byte_track::Rect<int>;
template class byte_track::Rect<float>;

template byte_track::Rect<int> byte_track::generate_rect_by_tlbr<int>(const byte_track::Tlbr<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_tlbr<float>(const byte_track::Tlbr<float>&);

template byte_track::Rect<int> byte_track::generate_rect_by_xyah<int>(const byte_track::Xyah<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_xyah<float>(const byte_track::Xyah<float>&);
