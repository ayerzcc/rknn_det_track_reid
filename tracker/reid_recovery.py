"""
ReID 恢复模块
==============
用于丢失轨迹的重识别恢复。

架构:
  检测 → 跟踪(IoU) → [丢失轨迹] → ReID匹配 → 恢复跟踪

特点:
  - 仅在轨迹丢失时才调用 ReID，大幅减少计算量
  - 维护特征库用于长期身份匹配
  - 支持跨镜头重识别
"""

import numpy as np
from collections import OrderedDict


class ReIDRecovery:
    """
    ReID 恢复管理器
    
    负责:
      1. 维护已确认轨迹的特征库
      2. 对丢失轨迹进行 ReID 匹配
      3. 返回匹配结果供追踪器恢复
    """
    
    def __init__(self, reid_extractor, cfg):
        """
        Args:
            reid_extractor: FeatureExtractor 实例
            cfg: 配置字典
        """
        self.reid_extractor = reid_extractor
        
        reid_cfg = cfg.get('REID', {})
        self.reid_thresh = reid_cfg.get('REID_THRESH', 0.3)  # ReID 匹配阈值
        self.gallery_size = reid_cfg.get('GALLERY_SIZE', 100)  # 特征库大小
        
        # 特征库: track_id -> {'features': deque, 'last_frame': int, 'bbox': array}
        self.gallery = OrderedDict()
        
        # 统计信息
        self.stats = {
            'total_recoveries': 0,
            'reid_calls': 0,
            'successful_matches': 0
        }
    
    def update_gallery(self, track_id, feature, frame_id, bbox):
        """
        更新特征库
        
        Args:
            track_id: 轨迹 ID
            feature: ReID 特征向量
            frame_id: 当前帧 ID
            bbox: 边界框 [x1, y1, x2, y2]
        """
        if track_id not in self.gallery:
            self.gallery[track_id] = {
                'features': [],
                'last_frame': frame_id,
                'bbox': bbox.copy()
            }
        
        entry = self.gallery[track_id]
        entry['features'].append(feature.copy())
        entry['last_frame'] = frame_id
        entry['bbox'] = bbox.copy()
        
        # 限制特征历史长度
        if len(entry['features']) > 10:
            entry['features'] = entry['features'][-10:]
        
        # 限制特征库大小
        if len(self.gallery) > self.gallery_size:
            # 移除最旧的条目
            oldest_id = next(iter(self.gallery))
            del self.gallery[oldest_id]
    
    def get_track_feature(self, track_id):
        """
        获取轨迹的平均特征
        
        Args:
            track_id: 轨迹 ID
        
        Returns:
            avg_feature: 平均特征向量，如果不存在返回 None
        """
        if track_id not in self.gallery:
            return None
        
        features = self.gallery[track_id]['features']
        if len(features) == 0:
            return None
        
        # 计算平均特征
        avg_feature = np.mean(features, axis=0)
        avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-6)
        return avg_feature
    
    def try_recover(self, frame, detections, lost_tracks, frame_id):
        """
        尝试恢复丢失的轨迹
        
        Args:
            frame: 当前帧图像
            detections: (N, 5) 检测结果 [x1, y1, x2, y2, score]
            lost_tracks: 丢失轨迹列表 (STrack 对象)
            frame_id: 当前帧 ID
        
        Returns:
            recovery_map: {lost_track_id: matched_det_idx} 匹配结果
        """
        if len(detections) == 0 or len(lost_tracks) == 0:
            return {}
        
        if self.reid_extractor is None:
            return {}
        
        self.stats['reid_calls'] += 1
        
        # 提取检测框的 ReID 特征
        det_features = self.reid_extractor.extract(frame, detections[:, :4])
        
        # 收集丢失轨迹的特征
        lost_track_ids = []
        lost_features = []
        
        for track in lost_tracks:
            feat = self.get_track_feature(track.track_id)
            if feat is not None:
                lost_track_ids.append(track.track_id)
                lost_features.append(feat)
        
        if len(lost_features) == 0:
            return {}
        
        lost_features = np.array(lost_features)  # (M, D)
        
        # 计算特征距离矩阵
        # dist[i, j] = 1 - cosine_similarity(lost[i], det[j])
        similarity = lost_features @ det_features.T  # (M, N)
        dist_matrix = 1 - similarity
        
        # 匹配: 找到距离最小且小于阈值的配对
        recovery_map = {}
        matched_dets = set()
        
        # 按距离排序，优先匹配最近的
        for i, track_id in enumerate(lost_track_ids):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]
            
            if min_dist < self.reid_thresh and min_idx not in matched_dets:
                recovery_map[track_id] = min_idx
                matched_dets.add(min_idx)
                self.stats['successful_matches'] += 1
        
        if recovery_map:
            self.stats['total_recoveries'] += len(recovery_map)
        
        return recovery_map
    
    def compute_reid_distance(self, track_features, det_features):
        """
        计算 ReID 特征距离
        
        Args:
            track_features: (M, D) 轨迹特征
            det_features: (N, D) 检测特征
        
        Returns:
            dist_matrix: (M, N) 距离矩阵
        """
        if len(track_features) == 0 or len(det_features) == 0:
            return np.array([])
        
        # Cosine distance
        similarity = track_features @ det_features.T
        return 1 - similarity
    
    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def clear_gallery(self):
        """清空特征库"""
        self.gallery.clear()
