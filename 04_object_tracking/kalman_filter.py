"""2D bounding-box Kalman filter with constant-velocity motion model."""

from __future__ import annotations

import numpy as np


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes using constant velocity motion."""

    # Standard motion and observation variances from ByteTrack
    _std_weight_position = 1.0 / 20.0
    _std_weight_velocity = 1.0 / 160.0

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create new track state from unconfirmed measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step."""
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
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return projected_mean, projected_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter measurement update step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        kalman_gain = np.linalg.solve(
            projected_cov,
            np.dot(covariance, self._update_mat.T).T,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def xyxy_to_xyah(box: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) box to (center_x, center_y, aspect_ratio, height)."""
    box = np.asarray(box, dtype=np.float32)
    w = max(1e-4, box[2] - box[0])
    h = max(1e-4, box[3] - box[1])
    x = box[0] + w / 2.0
    y = box[1] + h / 2.0
    a = w / h
    return np.array([x, y, a, h], dtype=np.float32)


def xyah_to_xyxy(xyah: np.ndarray) -> tuple[int, int, int, int]:
    """Convert (center_x, center_y, aspect_ratio, height) to integer (x1, y1, x2, y2)."""
    x, y, a, h = xyah[:4]
    w = a * h
    x1 = int(round(x - w / 2.0))
    y1 = int(round(y - h / 2.0))
    x2 = int(round(x + w / 2.0))
    y2 = int(round(y + h / 2.0))
    return x1, y1, x2, y2
