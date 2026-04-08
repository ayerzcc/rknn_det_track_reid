"""
全局运动补偿 (Global Motion Compensation) 模块
================================================
核心功能: 估计帧间全局仿射变换，补偿四足机器人
          行走/奔跑时的相机自身运动导致的前景漂移。

支持三种 GMC 方法:
  1. ORB 特征匹配 (默认，速度快)
  2. ECC 图像配准 (精度高，速度慢)
  3. 稀疏光流 (平衡)
"""

import cv2
import numpy as np
from copy import copy


class GMC:
    """
    全局运动补偿器。

    通过估计帧间全局仿射变换矩阵来补偿相机自身运动,
    是 BoT-SORT 在四足机器人场景下的关键组件。
    """

    def __init__(self, method='orb', downscale=2):
        """
        Args:
            method: 'orb' | 'ecc' | 'sparseOptFlow' | 'none'
            downscale: 下采样倍数，加速计算
        """
        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'ecc':
            # ECC 参数
            number_of_iterations = 25
            termination_eps = 1e-5
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps
            )

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )

        elif self.method == 'none':
            pass
        else:
            raise ValueError(f"不支持的 GMC 方法: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        """
        估计帧间全局运动变换矩阵。

        Args:
            raw_frame: 当前帧 (BGR)
            detections: 当前帧检测框 (可选，用于掩码前景)

        Returns:
            H: 2x3 仿射变换矩阵。若估计失败则返回单位矩阵。
        """
        if self.method == 'orb':
            return self._apply_orb(raw_frame, detections)
        elif self.method == 'ecc':
            return self._apply_ecc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self._apply_sparse_optflow(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def _apply_orb(self, raw_frame, detections=None):
        """ORB 特征点匹配估计全局运动"""
        h, w = raw_frame.shape[:2]

        # 下采样
        frame = cv2.resize(raw_frame, (w // self.downscale, h // self.downscale))
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        # 创建掩码，排除检测到的前景区域
        mask = np.zeros_like(frame_gray)
        mask[int(0.02 * h / self.downscale):int(0.98 * h / self.downscale),
             int(0.02 * w / self.downscale):int(0.98 * w / self.downscale)] = 255

        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int32)
                mask[max(0, tlbr[1]):min(mask.shape[0], tlbr[3]),
                     max(0, tlbr[0]):min(mask.shape[1], tlbr[2])] = 0

        # 检测特征点
        keypoints = self.detector.detect(frame_gray, mask)
        keypoints, descriptors = self.extractor.compute(frame_gray, keypoints)

        H = np.eye(2, 3)

        if not self.initializedFirstFrame:
            self.prevFrame = frame_gray.copy()
            self.prevKeyPoints = copy(keypoints)
            self.prevDescriptors = copy(descriptors)
            self.initializedFirstFrame = True
            return H

        if descriptors is None or self.prevDescriptors is None:
            self.prevFrame = frame_gray.copy()
            self.prevKeyPoints = copy(keypoints)
            self.prevDescriptors = copy(descriptors)
            return H

        # KNN 匹配
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, k=2)

        # Lowe's ratio test
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([w, h]) / self.downscale

        for pair in knnMatches:
            # OpenCV 不能保证每个查询点都返回 2 个近邻，少于 2 个时直接跳过。
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.9 * n.distance:
                prevKp = self.prevKeyPoints[m.queryIdx]
                currKp = keypoints[m.trainIdx]
                spatialDist = (
                    prevKp.pt[0] - currKp.pt[0],
                    prevKp.pt[1] - currKp.pt[1],
                )
                if (abs(spatialDist[0]) < maxSpatialDistance[0] and
                        abs(spatialDist[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDist)
                    matches.append(m)

        if len(matches) < 4:
            self.prevFrame = frame_gray.copy()
            self.prevKeyPoints = copy(keypoints)
            self.prevDescriptors = copy(descriptors)
            return H

        # 估计仿射变换
        prevPoints = np.array([self.prevKeyPoints[m.queryIdx].pt for m in matches])
        currPoints = np.array([keypoints[m.trainIdx].pt for m in matches])

        H, inliers = cv2.estimateAffinePartial2D(
            prevPoints, currPoints, cv2.RANSAC
        )

        if H is None:
            H = np.eye(2, 3)

        # 缩放补偿
        if self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale

        self.prevFrame = frame_gray.copy()
        self.prevKeyPoints = copy(keypoints)
        self.prevDescriptors = copy(descriptors)

        return H

    def _apply_ecc(self, raw_frame, detections=None):
        """ECC 图像配准估计全局运动"""
        h, w = raw_frame.shape[:2]

        frame = cv2.resize(raw_frame, (w // self.downscale, h // self.downscale))
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        H = np.eye(2, 3, dtype=np.float32)

        if not self.initializedFirstFrame:
            self.prevFrame = frame_gray.copy()
            self.initializedFirstFrame = True
            return H

        try:
            (cc, H) = cv2.findTransformECC(
                self.prevFrame, frame_gray,
                H, self.warp_mode, self.criteria,
                None, 1
            )
        except cv2.error:
            pass  # 失败时返回单位矩阵

        if self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale

        self.prevFrame = frame_gray.copy()
        return H

    def _apply_sparse_optflow(self, raw_frame, detections=None):
        """稀疏光流估计全局运动"""
        h, w = raw_frame.shape[:2]

        frame = cv2.resize(raw_frame, (w // self.downscale, h // self.downscale))
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        H = np.eye(2, 3)

        if not self.initializedFirstFrame:
            self.prevFrame = frame_gray.copy()
            self.initializedFirstFrame = True
            return H

        # 创建掩码排除前景
        mask = np.zeros_like(frame_gray)
        mask[int(0.02 * h / self.downscale):int(0.98 * h / self.downscale),
             int(0.02 * w / self.downscale):int(0.98 * w / self.downscale)] = 255

        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int32)
                mask[max(0, tlbr[1]):min(mask.shape[0], tlbr[3]),
                     max(0, tlbr[0]):min(mask.shape[1], tlbr[2])] = 0

        # 寻找角点
        keypoints = cv2.goodFeaturesToTrack(
            frame_gray, mask=mask, **self.feature_params
        )

        if keypoints is None or len(keypoints) < 4:
            self.prevFrame = frame_gray.copy()
            return H

        # 光流追踪
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame_gray, keypoints, None
        )

        prevPoints = keypoints[status[:, 0] == 1]
        currPoints = matchedKeypoints[status[:, 0] == 1]

        if len(prevPoints) < 4:
            self.prevFrame = frame_gray.copy()
            return H

        # 用 RANSAC 估计仿射变换
        H, inliers = cv2.estimateAffinePartial2D(
            prevPoints, currPoints, cv2.RANSAC
        )

        if H is None:
            H = np.eye(2, 3)

        if self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale

        self.prevFrame = frame_gray.copy()
        return H
