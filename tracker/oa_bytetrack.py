"""
OA-ByteTrack (Occlusion-Aware ByteTrack)
========================================
论文: "Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking"
CVPR 2026

基于 ByteTrack 的遮挡感知改进版本，集成:
  - OAM: 遮挡感知模块
  - OAO: 遮挡感知偏移 (用于高置信度检测关联)
  - BAM: 偏置感知动量 (用于低置信度检测更新)
"""

import numpy as np

from .track import STrack, BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from . import matching
from .occlusion_aware import (
    OcclusionAwareModule, 
    OcclusionAwareOffset, 
    BiasAwareMomentum
)


class OAByteTrack:
    """
    Occlusion-Aware ByteTrack
    
    在 ByteTrack 基础上集成遮挡感知能力：
      1. 第一次关联：使用 OAO 精细化空间一致性
      2. 第二次关联：使用 BAM 优化卡尔曼更新
      3. 保存轨迹的遮挡系数用于后续更新
    """
    
    def __init__(self, cfg):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        BaseTrack.reset_id()
        
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        
        tracker_cfg = cfg.get('TRACKER', {})
        match_cfg = tracker_cfg.get('MATCHING', {})
        oa_cfg = tracker_cfg.get('OCCLUSION_AWARE', {})
        
        # ByteTrack 参数
        self.track_high_thresh = match_cfg.get('TRACK_HIGH_THRESH', 0.6)
        self.track_low_thresh = match_cfg.get('TRACK_LOW_THRESH', 0.1)
        self.new_track_thresh = match_cfg.get('NEW_TRACK_THRESH', 0.7)
        self.match_thresh = match_cfg.get('MATCH_THRESH', 0.8)
        
        self.buffer_size = tracker_cfg.get('TRACK_BUFFER', 30)
        self.max_time_lost = self.buffer_size
        
        # OA-SORT 参数
        self.use_oao = oa_cfg.get('USE_OAO', True)
        self.use_bam = oa_cfg.get('USE_BAM', True)
        self.tau = oa_cfg.get('TAU', 0.15)  # DanceTrack
        
        print(f"[OA-ByteTrack] track_high_thresh={self.track_high_thresh}, "
              f"track_low_thresh={self.track_low_thresh}, "
              f"new_track_thresh={self.new_track_thresh}, "
              f"OAO={self.use_oao}, BAM={self.use_bam}, tau={self.tau}")
        self.k_x = oa_cfg.get('K_X', 3 * np.sqrt(2))
        self.k_y = oa_cfg.get('K_Y', 3)
        
        # 初始化 OA 模块
        if self.use_oao or self.use_bam:
            self.oam = OcclusionAwareModule(k_x=self.k_x, k_y=self.k_y)
            self.oao = OcclusionAwareOffset(tau=self.tau) if self.use_oao else None
            self.bam = BiasAwareMomentum() if self.use_bam else None
        
        # 图像尺寸（用于生成高斯地图）
        self.img_h = None
        self.img_w = None
    
    def update(self, detections, img=None, features=None):
        """
        更新跟踪器
        
        Args:
            detections: (N, 5) 检测结果 [x1, y1, x2, y2, score]
            img: 原始图像（用于获取尺寸）
            features: ReID 特征（未使用）
        
        Returns:
            output_stracks: 激活的轨迹列表
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 更新图像尺寸
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]
        
        if len(detections) == 0:
            detections = np.empty((0, 5))
        
        scores = detections[:, 4]
        bboxes = detections[:, :4]
        
        # 分离高/低置信度检测
        remain_inds = scores > self.track_high_thresh
        inds_low = scores > self.track_low_thresh
        inds_second = np.logical_and(inds_low, np.logical_not(remain_inds))
        
        dets_first = bboxes[remain_inds]
        scores_first = scores[remain_inds]
        
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        
        # 创建检测轨迹对象
        detections_first = []
        for i in range(len(dets_first)):
            detections_first.append(
                STrack(dets_first[i], scores_first[i])
            )
        
        detections_second = []
        for i in range(len(dets_second)):
            detections_second.append(
                STrack(dets_second[i], scores_second[i])
            )
        
        # 分离已确认和未确认轨迹
        unconfirmed = []
        tracked_stracks = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked_stracks.append(t)
        
        # 联合跟踪和丢失轨迹进行预测
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        # ========== 第一次关联：高置信度检测 ==========
        # 计算估计位置
        estimations = np.array([s.tlbr for s in strack_pool])
        
        # 计算 IoU 矩阵
        iou_dists = matching.iou_distance(strack_pool, detections_first)
        
        # 使用 OAO 精细化空间一致性（如果启用）
        if self.use_oao and len(estimations) > 0 and len(detections_first) > 0:
            spatial_scores = self.oao.refine_spatial_consistency(
                estimations, 1 - iou_dists
            )
            # 转换回代价（cost = 1 - score）
            dists = 1 - spatial_scores
        else:
            dists = iou_dists
        
        # 匈牙利匹配
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )
        
        # 更新匹配的轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_first[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ========== 第二次关联：低置信度检测 ==========
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        
        dists2 = matching.iou_distance(r_tracked_stracks, detections_second)
        matches2, u_track2, u_detection2 = matching.linear_assignment(
            dists2, thresh=0.5
        )
        
        # 更新匹配的轨迹（使用 BAM）
        for itracked, idet in matches2:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            
            if track.state == TrackState.Tracked:
                # 使用 BAM 精细化更新
                if self.use_bam and hasattr(track, 'occlusion_coeff'):
                    estimation = track.tlbr
                    detection = det.tlbr
                    oc_last_obs = track.occlusion_coeff
                    
                    # 计算精细化观测
                    refined_obs = self.bam.refine_observation(
                        estimation, detection, oc_last_obs
                    )
                    
                    # 使用精细化观测更新轨迹
                    det_refined = STrack(refined_obs, det.score)
                    track.update(det_refined, self.frame_id)
                else:
                    track.update(det, self.frame_id)
                
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # 标记未匹配的轨迹为丢失
        for it in u_track2:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # ========== 处理未确认轨迹 ==========
        detections_remain = [detections_first[i] for i in u_detection]
        dists3 = matching.iou_distance(unconfirmed, detections_remain)
        matches3, u_unconfirmed, u_detection3 = matching.linear_assignment(
            dists3, thresh=0.7
        )
        
        for itracked, idet in matches3:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # ========== 初始化新轨迹 ==========
        for inew in u_detection3:
            track = detections_remain[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        
        # ========== 移除长时间丢失的轨迹 ==========
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        # ========== 更新遮挡系数 ==========
        # 为所有激活的轨迹计算遮挡系数
        if self.use_bam and len(activated_stracks) > 0:
            active_bboxes = np.array([t.tlbr for t in activated_stracks])
            if len(active_bboxes) > 0:
                oc_coeffs = self.oam.compute_occlusion_coefficients(
                    active_bboxes, self.img_h, self.img_w
                )
                for i, track in enumerate(activated_stracks):
                    track.occlusion_coeff = oc_coeffs[i]
        
        # 更新轨迹列表
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_stracks
        )
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks
        )
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.removed_stracks = self.removed_stracks[-1000:]
        
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        output_stracks = [
            t for t in self.tracked_stracks if t.is_activated
        ]
        return output_stracks
    
    def recover_track(self, track, detection, frame_id):
        """
        通过 ReID 恢复丢失的轨迹
        
        Args:
            track: 要恢复的轨迹 (STrack)
            detection: (5,) 匹配的检测 [x1, y1, x2, y2, score]
            frame_id: 当前帧 ID
        """
        # 从丢失列表中移除
        if track in self.lost_stracks:
            self.lost_stracks.remove(track)
        
        # 更新轨迹状态
        track.mean, track.covariance = self.kalman_filter.update(
            track.mean, track.covariance,
            track.tlbr_to_xyah(detection[:4])
        )
        track._tlbr = detection[:4]
        track.score = detection[4]
        track.frame_id = frame_id
        track.tracklet_len = 0
        track.state = TrackState.Tracked
        track.is_activated = True
        
        # 添加回追踪列表
        if track not in self.tracked_stracks:
            self.tracked_stracks.append(track)


def joint_stracks(tlista, tlistb):
    """合并两个轨迹列表（去重）"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if t.track_id not in exists:
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """从 tlista 中移除 tlistb 中的轨迹"""
    ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in ids_b]


def remove_duplicate_stracks(stracksa, stracksb):
    """移除重复轨迹"""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
