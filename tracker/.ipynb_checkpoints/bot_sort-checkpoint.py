"""
BoT-SORT 追踪器 — 四足机器人适配版 (优化版)
=========================================
核心策略调整:
  1. 第一次关联: 纯 IoU 匹配 (不依赖 ReID)
  2. 第二次关联: 低置信 + 纯 IoU
  3. ReID 仅用于重激活丢失轨迹 (事后补救)
"""

import numpy as np

from .track import STrack, BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .gmc import GMC
from . import matching


class BoTSORT:
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

        self.track_high_thresh = match_cfg.get('TRACK_HIGH_THRESH', 0.6)
        self.track_low_thresh = match_cfg.get('TRACK_LOW_THRESH', 0.1)
        self.new_track_thresh = match_cfg.get('NEW_TRACK_THRESH', 0.7)
        self.match_thresh = match_cfg.get('MATCH_THRESH', 0.5)

        self.buffer_size = tracker_cfg.get('TRACK_BUFFER', 30)
        self.max_time_lost = self.buffer_size

        gmc_method = gmc_cfg.get('METHOD', 'orb')
        gmc_downscale = gmc_cfg.get('DOWNSCALE', 2)
        self.gmc = GMC(method=gmc_method, downscale=gmc_downscale)
        
        print(f"[BoT-SORT] track_high_thresh={self.track_high_thresh}, "
              f"track_low_thresh={self.track_low_thresh}, "
              f"new_track_thresh={self.new_track_thresh}, "
              f"GMC={gmc_method}")

    def update(self, detections, img=None, features=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) == 0:
            detections = np.empty((0, 5))
        if features is not None and len(features) == 0:
            features = None

        scores = detections[:, 4]
        bboxes = detections[:, :4]

        remain_inds = scores > self.track_high_thresh
        inds_low = scores > self.track_low_thresh
        inds_second = np.logical_and(inds_low, np.logical_not(remain_inds))

        dets_first = bboxes[remain_inds]
        scores_first = scores[remain_inds]
        feats_first = features[remain_inds] if features is not None else None

        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        detections_first = []
        for i in range(len(dets_first)):
            feat = feats_first[i] if feats_first is not None else None
            detections_first.append(
                STrack(dets_first[i], scores_first[i], feat=feat)
            )

        detections_second = []
        for i in range(len(dets_second)):
            detections_second.append(
                STrack(dets_second[i], scores_second[i])
            )

        unconfirmed = []
        tracked_stracks = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked_stracks.append(t)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        STrack.multi_predict(strack_pool)

        # Step 4: 第一次关联 - 纯 IoU 匹配 (不使用 ReID)
        iou_dists = matching.iou_distance(strack_pool, detections_first)
        matches, u_track, u_detection = matching.linear_assignment(
            iou_dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_first[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 5: 第二次关联 - 纯 IoU
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        iou_dists2 = matching.iou_distance(r_tracked_stracks, detections_second)
        matches2, u_track2, u_detection2 = matching.linear_assignment(
            iou_dists2, thresh=0.5
        )

        for itracked, idet in matches2:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track2:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 6: 处理未确认轨迹
        detections_remain = [detections_first[i] for i in u_detection]
        iou_dists3 = matching.iou_distance(unconfirmed, detections_remain)
        matches3, u_unconfirmed, u_detection3 = matching.linear_assignment(
            iou_dists3, thresh=0.7
        )

        for itracked, idet in matches3:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 7: 初始化新轨迹
        for inew in u_detection3:
            track = detections_remain[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 8: 使用 ReID 重激活丢失轨迹 (仅用于补救)
        if features is not None and len(self.lost_stracks) > 0:
            lost_with_features = [t for t in self.lost_stracks if t.smooth_feat is not None]
            det_with_features = [d for d in detections_first if d.curr_feat is not None]
            
            if len(lost_with_features) > 0 and len(det_with_features) > 0:
                iou_dists_lost = matching.iou_distance(lost_with_features, det_with_features)
                reid_dists_lost = matching.embedding_distance(lost_with_features, det_with_features)
                
                # 仅当 IoU 接近时才使用 ReID
                for i, track in enumerate(lost_with_features):
                    for j, det in enumerate(det_with_features):
                        if iou_dists_lost[i, j] < 0.3:  # IoU 足够接近
                            if reid_dists_lost[i, j] < 0.3:  # ReID 也匹配
                                track.re_activate(det, self.frame_id, new_id=False)
                                refind_stracks.append(track)
                                if track in self.lost_stracks:
                                    self.lost_stracks.remove(track)

        # 管理丢失/移除轨迹
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks
        )
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks
        )
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
