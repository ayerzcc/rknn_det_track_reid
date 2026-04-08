"""
改进的卡尔曼滤波器 —— 适配四足机器人步态震荡
==============================================
相比标准 SORT 卡尔曼滤波:
  1. 降低位置/速度方差权重 → 对四足抖动更鲁棒
  2. 状态向量使用 [cx, cy, a, h, vx, vy, va, vh]
     cx,cy = 边界框中心; a = 宽高比; h = 高度
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    8-dim 状态空间卡尔曼滤波器，用于边界框追踪。

    状态向量: [cx, cy, a, h, vx, vy, va, vh]
    观测向量: [cx, cy, a, h]
    """

    # 噪声权重 — 四足场景适配（较标准 DeepSORT 更保守）
    _std_weight_position = 1.0 / 20   # 位置噪声 (原 1/20)
    _std_weight_velocity = 1.0 / 160  # 速度噪声 (原 1/160)

    def __init__(self):
        ndim, dt = 4, 1.0

        # 状态转移矩阵 F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 观测矩阵 H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement):
        """
        从首次检测初始化轨迹状态。

        Args:
            measurement: [cx, cy, a, h] 边界框参数

        Returns:
            (mean, covariance): 初始状态向量与协方差
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],   # cx
            2 * self._std_weight_position * measurement[3],   # cy
            1e-2,                                              # a
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,                                              # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        卡尔曼预测步骤。

        Args:
            mean: 当前状态向量 (8,)
            covariance: 当前协方差矩阵 (8,8)

        Returns:
            (mean, covariance): 预测后的状态
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态空间投影到观测空间。

        Args:
            mean: 状态向量 (8,)
            covariance: 协方差矩阵 (8,8)

        Returns:
            (mean, covariance): 投影到观测空间的分布
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        批量预测（向量化加速）。

        Args:
            mean: (N, 8) 状态向量
            covariance: (N, 8, 8) 协方差矩阵

        Returns:
            (mean, covariance): 预测后
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones(len(mean)),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones(len(mean)),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.array([np.diag(s) for s in sqr])

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.matmul(covariance, self._motion_mat.T)
        covariance = np.matmul(self._motion_mat, covariance) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        卡尔曼更新步骤。

        Args:
            mean: 预测状态 (8,)
            covariance: 预测协方差 (8,8)
            measurement: 观测值 [cx, cy, a, h]

        Returns:
            (mean, covariance): 更新后的状态
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Cholesky 分解求卡尔曼增益
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """
        计算状态分布与观测之间的门控距离。

        Args:
            mean: 状态向量 (8,)
            covariance: 协方差 (8,8)
            measurements: (N, 4) 观测值
            only_position: 是否只用位置
            metric: 'maha' (马氏距离) 或 'gaussian'

        Returns:
            distances: (N,) 门控距离
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True,
                check_finite=False, overwrite_b=True
            )
            return np.sum(z * z, axis=0)
        else:
            raise ValueError(f"不支持的度量: {metric}")
