"""
OA-BoT-SORT 追踪器
==================
融合版本：OA-ByteTrack 的遮挡感知 + BoT-SORT 的 GMC 运动补偿

核心策略:
  1. GMC 运动补偿：补偿相机自身移动
  2. 第一次关联：IoU + OAO(遮挡偏移)
  3. 第二次关联：IoU + BAM(偏置动量)
  4. 纯 IoU 匹配，不使用 ReID（ReID 由外部 ReIDRecovery 管理）

适合场景:
  - 相机移动 + 目标遮挡同时存在
  - 四足机器人手持/移动相机场景
"""

import numpy as np

from .track import STrack, BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .gmc import GMC
from . import matching
from .occlusion_aware import (
    OcclusionAwareModule, 
    OcclusionAwareOffset, 
    BiasAwareMomentum
)


class OABoTSORT:
    """
    OA-BoT-SORT: 融合遮挡感知与 GMC 运动补偿
    
    结合:
      - OA-ByteTrack: 遮挡感知 (OAO + BAM)
      - BoT-SORT: GMC 运动补偿
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
        gmc_cfg = tracker_cfg.get('GMC', {})
        oa_cfg = tracker_cfg.get('OCCLUSION_AWARE', {})

        # ByteTrack 参数
        self.track_high_thresh = match_cfg.get('TRACK_HIGH_THRESH', 0.6)
        self.track_low_thresh = match_cfg.get('TRACK_LOW_THRESH', 0.1)
        self.new_track_thresh = match_cfg.get('NEW_TRACK_THRESH', 0.7)
        self.match_thresh = match_cfg.get('MATCH_THRESH', 0.8)

        self.buffer_size = tracker_cfg.get('TRACK_BUFFER', 30)
        self.max_time_lost = self.buffer_size

        # GMC 参数
        gmc_method = gmc_cfg.get('METHOD', 'orb')
        gmc_downscale = gmc_cfg.get('DOWNSCALE', 2)
        self.gmc = GMC(method=gmc_method, downscale=gmc_downscale)
        
        # OA-SORT 参数
        self.use_oao = oa_cfg.get('USE_OAO', True)
        self.use_bam = oa_cfg.get('USE_BAM', True)
        self.tau = oa_cfg.get('TAU', 0.15)
        self.k_x = oa_cfg.get('K_X', 3 * np.sqrt(2))
        self.k_y = oa_cfg.get('K_Y', 3)
        
        # 初始化 OA 模块
        if self.use_oao or self.use_bam:
            self.oam = OcclusionAwareModule(k_x=self.k_x, k_y=self.k_y)
            self.oao = OcclusionAwareOffset(tau=self.tau) if self.use_oao else None
            self.bam = BiasAwareMomentum() if self.use_bam else None
        
        # 图像尺寸
        self.img_h = None
        self.img_w = None
        
        print(f"[OA-BoT-SORT] track_high_thresh={self.track_high_thresh}, "
              f"track_low_thresh={self.track_low_thresh}, "
              f"GMC={gmc_method}, OAO={self.use_oao}, BAM={self.use_bam}")

    def update(self, detections, img=None, features=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 更新图像尺寸
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]
        
        # 应用 GMC 运动补偿
        if img is not None and self.gmc is not None:
            H = self.gmc.apply(img, detections)
            if H is not None:
                # 对所有轨迹应用运动补偿
                STrack.multi_gmc(self.tracked_stracks, H)
                STrack.multi_gmc(self.lost_stracks, H)

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
        
        # ========== 第一次关联：高置信度检测 + OAO ==========
        estimations = np.array([s.tlbr for s in strack_pool])
        iou_dists = matching.iou_distance(strack_pool, detections_first)
        
        # 使用 OAO 精细化空间一致性
        if self.use_oao and len(estimations) > 0 and len(detections_first) > 0:
            spatial_scores = self.oao.refine_spatial_consistency(
                estimations, 1 - iou_dists
            )
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
        
        # ========== 第二次关联：低置信度检测 + BAM ==========
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
                if self.use_bam and hasattr(track, 'occlusion_coeff'):
                    estimation = track.tlbr
                    detection = det.tlbr
                    oc_last_obs = track.occlusion_coeff
                    
                    refined_obs = self.bam.refine_observation(
                        estimation, detection, oc_last_obs
                    )
                    
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
        if self.use_bam and len(activated_stracks) > 0:
            active_bboxes = np.array([t.tlbr for t in activated_stracks])
            if len(active_bboxes) > 0 and self.img_h is not None:
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
        """通过 ReID 恢复丢失的轨迹"""
        if track in self.lost_stracks:
            self.lost_stracks.remove(track)
        
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
        
        if track not in self.tracked_stracks:
            self.tracked_stracks.append(track)


def joint_stracks(tlista, tlistb):
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
    ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in ids_b]


def remove_duplicate_stracks(stracksa, stracksb):
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
