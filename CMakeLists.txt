cmake_minimum_required(VERSION 3.14)
project(rover_vision_module_lab)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(RKNN_LIB /usr/lib/librknnrt.so CACHE FILEPATH "Path to librknnrt.so")
if(NOT EXISTS ${RKNN_LIB})
    message(FATAL_ERROR "RKNN runtime library not found: ${RKNN_LIB}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/tracker
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

set(TRACKER_SRCS
    src/tracker/BYTETracker.cpp
    src/tracker/KalmanFilter.cpp
    src/tracker/Object.cpp
    src/tracker/Rect.cpp
    src/tracker/STrack.cpp
    src/tracker/lapjv.cpp
    src/tracker/GMC.cpp
    src/tracker/OcclusionAware.cpp
    src/tracker/OAByteTracker.cpp
    src/tracker/BoTSORT.cpp
    src/tracker/ReIDRecovery.cpp
)

set(CORE_SRCS
    src/detector.cpp
    src/personfeature_extractor.cpp
    src/gallery_manager.cpp
    ${TRACKER_SRCS}
)

add_library(vision_module_core STATIC ${CORE_SRCS})
target_link_libraries(vision_module_core
    PUBLIC
        ${OpenCV_LIBS}
        ${RKNN_LIB}
        Eigen3::Eigen
)

add_executable(detector_test tools/detector_test.cpp)
target_link_libraries(detector_test vision_module_core)

add_executable(tracker_test tools/tracker_test.cpp)
target_link_libraries(tracker_test vision_module_core)

add_executable(reid_test tools/reid_test.cpp)
target_link_libraries(reid_test vision_module_core)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=cortex-a76 -mtune=cortex-a76")
endif()
