"""
IoU + ReID 融合匹配模块
=======================
BoT-SORT 的核心关联策略:
  1. 一次匹配: 高置信检测 ↔ 已追踪轨迹 (IoU + ReID 加权)
  2. 二次匹配: 低置信检测 ↔ 未匹配轨迹 (纯 IoU)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def iou_batch(bboxes1, bboxes2):
    bboxes1 = np.asarray(bboxes1).reshape(-1, 4)
    bboxes2 = np.asarray(bboxes2).reshape(-1, 4)

    xx1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0].reshape(1, -1))
    yy1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1].reshape(1, -1))
    xx2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2].reshape(1, -1))
    yy2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3].reshape(1, -1))

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1.reshape(-1, 1) + area2.reshape(1, -1) - inter
    iou = inter / (union + 1e-6)
    return iou


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    if len(tracks) == 0 or len(detections) == 0:
        return cost_matrix

    track_features = np.asarray(
        [t.smooth_feat for t in tracks], dtype=np.float32
    )
    det_features = np.asarray(
        [d.curr_feat for d in detections], dtype=np.float32
    )

    cost_matrix = cdist(track_features, det_features, metric=metric)
    return cost_matrix


def fuse_iou_reid(iou_dist, reid_dist, iou_weight=0.7):
    proximity_mask = iou_dist > 0.5
    appearance_mask = reid_dist > 0.4
    fused = iou_weight * iou_dist + (1 - iou_weight) * reid_dist
    fused[proximity_mask | appearance_mask] = 1.0
    return fused


def linear_assignment(cost_matrix, thresh=0.8):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_dets = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append([r, c])
        if r in unmatched_tracks:
            unmatched_tracks.remove(r)
        if c in unmatched_dets:
            unmatched_dets.remove(c)

    return np.array(matches).reshape(-1, 2), unmatched_tracks, unmatched_dets


def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlbrs = np.array([t.tlbr for t in atracks])
    btlbrs = np.array([t.tlbr for t in btracks])
    ious = iou_batch(atlbrs, btlbrs)
    cost_matrix = 1 - ious
    return cost_matrix.astype(np.float32)
