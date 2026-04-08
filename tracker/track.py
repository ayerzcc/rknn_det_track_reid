"""
轨迹状态管理
============
每条轨迹 (STrack) 维护:
  - 卡尔曼滤波状态
  - ReID 特征向量 (EMA 平滑)
  - 生命周期状态机: New → Tracked → Lost → Removed
"""

import numpy as np
from collections import deque

from .kalman_filter import KalmanFilter


class TrackState:
    """轨迹状态枚举"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """轨迹基类"""
    _count = 0

    @staticmethod
    def _next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New
    score = 0
    frame_id = 0
    start_frame = 0
    time_since_update = 0


class STrack(BaseTrack):
    """
    单条轨迹 — BoT-SORT 核心数据结构。

    包含:
      - 卡尔曼滤波器状态
      - ReID 外观特征 (EMA 滑动平均)
      - 检测边界框
      - 状态管理
    """

    shared_kalman = KalmanFilter()

    def __init__(self, tlbr, score, feat=None, feat_history=50):
        """
        Args:
            tlbr: [x1, y1, x2, y2] 边界框
            score: 检测置信度
            feat: ReID 特征向量 (可选)
            feat_history: 特征历史长度
        """
        # 位置
        self._tlbr = np.asarray(tlbr, dtype=np.float64)
        self.score = score
        self.is_activated = False
        self.tracklet_len = 0

        # 卡尔曼
        self.kalman_filter = None
        self.mean = None
        self.covariance = None

        # ReID 特征
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9  # EMA 平滑系数
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        """
        更新 ReID 特征 (EMA 指数移动平均)。

        四足机器人场景下由于视角、光照剧烈变化，
        EMA 平滑有助于保持特征稳定性。
        """
        feat /= np.linalg.norm(feat) + 1e-6  # L2 归一化
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = (
                self.alpha * self.smooth_feat
                + (1 - self.alpha) * feat
            )
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6
        self.features.append(feat)

    def predict(self):
        """卡尔曼预测"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # 丢失时将高度速度置零
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        """批量预测（向量化加速）"""
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        
        # 确保协方差矩阵的形状是 (N, 8, 8)
        if len(multi_covariance.shape) == 2:
            multi_covariance = multi_covariance[np.newaxis, ...]

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """
        批量施加全局运动补偿。

        将 GMC 估计的仿射变换应用到所有轨迹的
        卡尔曼状态上，补偿相机自身运动。

        Args:
            stracks: 轨迹列表
            H: 2x3 仿射变换矩阵
        """
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]  # 旋转/缩放
        t = H[:2, 2]   # 平移

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            # 对位置中心施加仿射变换
            mean[:2] = R @ mean[:2] + t
            # 对速度施加旋转
            mean[4:6] = R @ mean[4:6]

            # 对协方差施加旋转
            R8 = np.eye(8)
            R8[:2, :2] = R
            R8[4:6, 4:6] = R
            cov = R8 @ cov @ R8.T

            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """激活新轨迹"""
        self.kalman_filter = kalman_filter
        self.track_id = self._next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlbr_to_xyah(self._tlbr)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活丢失的轨迹"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlbr_to_xyah(new_track.tlbr)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = self._next_id()

    def update(self, new_track, frame_id):
        """用新检测更新已跟踪的轨迹"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlbr = new_track.tlbr
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlbr_to_xyah(new_tlbr)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self._tlbr = new_tlbr

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    # ---- 坐标转换 ---- #

    @property
    def tlbr(self):
        """返回 [x1, y1, x2, y2]"""
        if self.mean is None:
            return self._tlbr.copy()
        ret = self.xyah_to_tlbr(self.mean[:4])
        return ret

    @property
    def tlwh(self):
        """返回 [x, y, w, h]"""
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlbr_to_xyah(tlbr):
        """[x1,y1,x2,y2] → [cx, cy, aspect_ratio, height]"""
        ret = np.asarray(tlbr, dtype=np.float64)
        w = ret[2] - ret[0]
        h = ret[3] - ret[1]
        cx = ret[0] + w / 2
        cy = ret[1] + h / 2
        a = w / (h + 1e-6)
        return np.array([cx, cy, a, h])

    @staticmethod
    def xyah_to_tlbr(xyah):
        """[cx, cy, aspect_ratio, height] → [x1, y1, x2, y2]"""
        cx, cy, a, h = xyah
        w = a * h
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ])

    def __repr__(self):
        return f'STrack(id={self.track_id}, state={self.state})'
