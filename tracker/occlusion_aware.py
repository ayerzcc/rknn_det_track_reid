"""
Occlusion-Aware Module (OAM)
============================
论文: "Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking"
CVPR 2026

核心组件:
  1. Depth Ordering: 基于边界框底边判断遮挡关系
  2. Occlusion Coefficient: 计算遮挡系数
  3. Gaussian Map: 使用高斯权重减少背景影响
"""

import numpy as np


class OcclusionAwareModule:
    """
    遮挡感知模块 (OAM)
    
    计算 bounding box 的遮挡系数，用于描述遮挡严重程度。
    """
    
    def __init__(self, k_x=3*np.sqrt(2), k_y=3, thre_occ=5):
        """
        Args:
            k_x: 高斯地图水平方向缩放因子 (论文推荐: DanceTrack=3*sqrt(2), MOT17=2)
            k_y: 高斯地图垂直方向缩放因子 (论文推荐: DanceTrack=3, MOT17=2)
            thre_occ: 遮挡触发阈值 (像素)，避免边界框波动
        """
        self.k_x = k_x
        self.k_y = k_y
        self.thre_occ = thre_occ
    
    def compute_occlusion_coefficients(self, bboxes, img_h=None, img_w=None):
        """
        计算精细化遮挡系数
        
        Args:
            bboxes: (N, 4) bounding boxes [x1, y1, x2, y2]
            img_h: 图像高度 (可选，用于生成高斯地图)
            img_w: 图像宽度 (可选)
        
        Returns:
            oc_hat: (N,) 精细化遮挡系数，范围 [0, 1]
        """
        if len(bboxes) == 0:
            return np.array([])
        
        # 转换为 numpy 数组，并兜底异常轨迹框，避免 GMC/卡尔曼预测后的非法坐标导致崩溃。
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes[np.newaxis, :]

        bboxes = bboxes.copy()
        x1 = np.minimum(bboxes[:, 0], bboxes[:, 2])
        y1 = np.minimum(bboxes[:, 1], bboxes[:, 3])
        x2 = np.maximum(bboxes[:, 0], bboxes[:, 2])
        y2 = np.maximum(bboxes[:, 1], bboxes[:, 3])
        bboxes[:, 0] = x1
        bboxes[:, 1] = y1
        bboxes[:, 2] = x2
        bboxes[:, 3] = y2

        N = len(bboxes)
        oc_hat = np.zeros(N, dtype=np.float32)
        valid_mask = np.isfinite(bboxes).all(axis=1) & (x2 > x1) & (y2 > y1)
        if not np.any(valid_mask):
            return oc_hat

        valid_indices = np.where(valid_mask)[0]
        bboxes_valid = bboxes[valid_mask]
        
        # 计算 IoU 矩阵
        iou_matrix = self._compute_iou_matrix(bboxes_valid)
        
        # 如果没有重叠，直接返回零向量
        if iou_matrix.max() == 0:
            return oc_hat
        
        # 获取需要处理的边界框索引（有重叠的）
        overlap_mask = iou_matrix.max(axis=1) > 0
        overlap_indices = np.where(overlap_mask)[0]
        
        if len(overlap_indices) == 0:
            return oc_hat
        
        # 计算高斯地图
        if img_h is None or img_w is None:
            # 使用边界框的最大范围
            img_h = max(int(np.ceil(bboxes_valid[:, 3].max())) + 1, 1)
            img_w = max(int(np.ceil(bboxes_valid[:, 2].max())) + 1, 1)
        
        gm = self._compute_gaussian_map(bboxes_valid[overlap_indices], img_h, img_w)
        
        # 计算深度排序和遮挡关系
        bottoms = bboxes_valid[:, 3]  # y2 (bottom)
        areas = (
            (bboxes_valid[:, 2] - bboxes_valid[:, 0]) *
            (bboxes_valid[:, 3] - bboxes_valid[:, 1])
        )
        
        # 遮挡关系矩阵: valid_mask[i, j] = True 表示 i 被 j 遮挡
        # 条件: bottom[i] > bottom[j] + thre_occ 且 IoU > 0
        valid_mask = (
            (bottoms[:, None] > bottoms[None, :] + self.thre_occ) & 
            (iou_matrix > 0)
        )
        
        # 为每个边界框计算遮挡系数
        for i in overlap_indices:
            # 找到遮挡该对象的所有其他对象
            occluders = np.where(valid_mask[i])[0]
            
            if len(occluders) == 0:
                continue
            
            # 裁剪局部高斯地图
            x1, y1, x2, y2 = bboxes_valid[i].astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            local_gm = gm[y1:y2, x1:x2]
            h, w = local_gm.shape
            
            # 创建遮挡地图
            occlusion_map = np.zeros((h, w), dtype=np.float32)
            
            # 对每个遮挡对象
            for j in occluders:
                # 计算重叠区域
                x1_j, y1_j, x2_j, y2_j = bboxes_valid[j]
                
                # 裁剪到当前边界框内的重叠区域
                t_clip = max(0, int(max(y1, y1_j) - y1))
                b_clip = min(h, int(min(y2, y2_j) - y1))
                l_clip = max(0, int(max(x1, x1_j) - x1))
                r_clip = min(w, int(min(x2, x2_j) - x1))
                
                if t_clip < b_clip and l_clip < r_clip:
                    occlusion_map[t_clip:b_clip, l_clip:r_clip] = 1.0
            
            # 计算精细化遮挡系数
            if areas[i] > 0:
                oc_hat[valid_indices[i]] = np.sum(local_gm * occlusion_map) / areas[i]
        
        return oc_hat
    
    def _compute_iou_matrix(self, bboxes):
        """计算 IoU 矩阵"""
        N = len(bboxes)
        iou_matrix = np.zeros((N, N), dtype=np.float32)
        
        for i in range(N):
            for j in range(i+1, N):
                iou = self._compute_iou(bboxes[i], bboxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _compute_iou(self, box1, box2):
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_gaussian_map(self, bboxes, img_h, img_w):
        """
        计算高斯地图 (Gaussian Map)
        
        Args:
            bboxes: (N, 4) bounding boxes
            img_h: 图像高度
            img_w: 图像宽度
        
        Returns:
            gm: (img_h, img_w) 高斯地图
        """
        img_h = max(int(img_h), 1)
        img_w = max(int(img_w), 1)
        gm = np.zeros((img_h, img_w), dtype=np.float32)
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # 计算中心点和尺寸
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # 计算标准差
            sigma_x = w / self.k_x
            sigma_y = h / self.k_y
            
            # 生成高斯核（仅在边界框范围内）
            x_range = np.arange(max(0, int(x1)), min(img_w, int(x2)))
            y_range = np.arange(max(0, int(y1)), min(img_h, int(y2)))
            
            if len(x_range) == 0 or len(y_range) == 0:
                continue
            
            X, Y = np.meshgrid(x_range, y_range)
            
            # 计算高斯值
            gaussian = np.exp(
                -((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2))
            )
            
            # 更新高斯地图（取最大值）
            y_start = max(0, int(y1))
            y_end = min(img_h, int(y2))
            x_start = max(0, int(x1))
            x_end = min(img_w, int(x2))
            
            gm[y_start:y_end, x_start:x_end] = np.maximum(
                gm[y_start:y_end, x_start:x_end], 
                gaussian
            )
        
        return gm


class OcclusionAwareOffset:
    """
    遮挡感知偏移 (OAO)
    
    将遮挡系数集成到空间一致性度量中，用于缓解代价混淆。
    """
    
    def __init__(self, tau=0.15):
        """
        Args:
            tau: 平衡系数 (论文推荐: DanceTrack=0.15, SportsMOT=0.2, MOT17=0.1)
        """
        self.tau = tau
        self.oam = OcclusionAwareModule()
    
    def refine_spatial_consistency(self, estimations, iou_matrix):
        """
        精细化空间一致性分数
        
        Args:
            estimations: (M, 4) 轨迹估计的边界框
            iou_matrix: (M, N) 轨迹和检测之间的 IoU 矩阵
        
        Returns:
            S: (M, N) 精细化后的空间一致性分数
        """
        if len(estimations) == 0:
            return iou_matrix
        
        # 计算估计的遮挡系数
        oc_est = self.oam.compute_occlusion_coefficients(estimations)
        
        # 计算精细化分数: S = tau * (1 - Oc) + (1 - tau) * IoU
        # 扩展遮挡系数维度以匹配 IoU 矩阵 (M, N)
        oc_expanded = oc_est[:, np.newaxis]  # (M, 1) -> 广播为 (M, N)
        
        S = self.tau * (1 - oc_expanded) + (1 - self.tau) * iou_matrix
        
        return S


class BiasAwareMomentum:
    """
    偏置感知动量 (BAM)
    
    用于优化卡尔曼滤波更新，减少不准确检测的影响。
    """
    
    def __init__(self):
        self.oam = OcclusionAwareModule()
    
    def compute_momentum(self, estimation, detection, oc_last_obs):
        """
        计算偏置感知动量
        
        Args:
            estimation: (4,) 当前估计位置
            detection: (4,) 关联的低置信度检测
            oc_last_obs: 标量，轨迹最近观测的遮挡系数
        
        Returns:
            bam: 标量，动量值 [0, 1]
        """
        # 计算 IoU
        iou = self._compute_iou(estimation, detection)
        
        # BAM = IoU * (1 - Oc)
        bam = iou * (1 - oc_last_obs)
        
        return bam
    
    def _compute_iou(self, box1, box2):
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def refine_observation(self, estimation, detection, oc_last_obs):
        """
        精细化观测值
        
        Args:
            estimation: (4,) 当前估计位置
            detection: (4,) 关联的检测
            oc_last_obs: 最近观测的遮挡系数
        
        Returns:
            refined_obs: (4,) 精细化后的观测
        """
        bam = self.compute_momentum(estimation, detection, oc_last_obs)
        
        # Z_hat = BAM * Z + (1 - BAM) * H * X
        # 这里简化为线性插值
        refined_obs = bam * detection + (1 - bam) * estimation
        
        return refined_obs
