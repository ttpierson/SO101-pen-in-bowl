"""SO-101 forward / inverse kinematics in plain NumPy, read from the SO-ARM100 URDF.

No Isaac dependency, so policies can use it anywhere. The tool point (TCP) is the middle
of the jaw gap, expressed in ``gripper_link``: the approach direction is that link's -Z,
and the jaws close along its X axis (fixed jaw inner face at x=-7.9 mm, moving jaw at
+7.9 mm when the gripper joint is 0, fingertips at z=-104 mm). The URDF's
``gripper_frame_link`` sits on the fixed jaw's face, not in the gap, so it is not used.
All poses are in the robot base frame (``base_link``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_URDF = os.environ.get("SO101_URDF", "/data/so101/so101_new_calib.urdf")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Middle of the jaw gap, 6 mm above the fingertips, in gripper_link (slightly toward the moving jaw).
TCP_IN_GRIPPER_LINK = np.array([0.002, -0.0002, -0.098])


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class SO101Kinematics:
    """FK and damped-least-squares IK for the 5 arm joints of the SO-101."""

    def __init__(self, urdf_path: str = DEFAULT_URDF) -> None:
        joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
        self._chain = []  # (origin 4x4, axis or None) for each joint, base -> tool
        for name in ARM_JOINTS:
            j = joints[name]
            o = j.find("origin")
            T = np.eye(4)
            T[:3, :3] = _rpy_to_matrix(*map(float, o.get("rpy").split()))
            T[:3, 3] = list(map(float, o.get("xyz").split()))
            axis = j.find("axis")
            self._chain.append((T, np.array(list(map(float, axis.get("xyz").split())))))
        self.lower = np.array([float(joints[n].find("limit").get("lower")) for n in ARM_JOINTS])
        self.upper = np.array([float(joints[n].find("limit").get("upper")) for n in ARM_JOINTS])

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the gripper_link pose (4x4) in the base frame."""
        T = np.eye(4)
        for (origin, axis), qi in zip(self._chain, q):
            R = np.eye(4)
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ origin @ R
        return T

    def tcp(self, q: np.ndarray) -> np.ndarray:
        """Return the tool point (middle of the jaw gap) in the base frame."""
        G = self.fk(q)
        return G[:3, :3] @ TCP_IN_GRIPPER_LINK + G[:3, 3]

    def _residual(self, q, target_pos, w_down, grasp_axis, w_roll):
        grip = self.fk(q)
        res = [grip[:3, :3] @ TCP_IN_GRIPPER_LINK + grip[:3, 3] - target_pos]
        if w_down > 0:
            res.append(w_down * (grip[:3, 2] - [0.0, 0.0, 1.0]))  # approach (-Z) points straight down
        if grasp_axis is not None and w_roll > 0:
            # Jaws close along gripper_link X; that must be perpendicular to the object's long axis.
            res.append(np.atleast_1d(w_roll * grip[:3, 0] @ grasp_axis))
        return np.concatenate(res)

    def ik(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        *,
        grasp_axis: np.ndarray | None = None,
        w_down: float = 0.05,
        w_roll: float = 0.05,
        iters: int = 200,
    ) -> tuple[np.ndarray, float]:
        """Solve for joints putting the tool at ``target_pos`` pointing down; return (q, position error m)."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        best_q, best_cost = None, np.inf
        seeds = [np.asarray(q_seed, dtype=np.float64), np.array([np.arctan2(target_pos[1], target_pos[0]), 0.0, 0.5, 1.0, 0.0])]
        for q in seeds:
            q = np.clip(q.copy(), self.lower, self.upper)
            lam = 1e-3
            for _ in range(iters):
                r = self._residual(q, target_pos, w_down, grasp_axis, w_roll)
                J = np.empty((r.size, 5))
                for k in range(5):
                    dq = np.zeros(5)
                    dq[k] = 1e-6
                    J[:, k] = (self._residual(q + dq, target_pos, w_down, grasp_axis, w_roll) - r) / 1e-6
                step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ r)
                q_new = np.clip(q + step, self.lower, self.upper)
                if np.sum(self._residual(q_new, target_pos, w_down, grasp_axis, w_roll) ** 2) < np.sum(r**2):
                    q, lam = q_new, max(lam / 3, 1e-6)
                else:
                    lam *= 5
                if np.linalg.norm(step) < 1e-7:
                    break
            cost = np.sum(self._residual(q, target_pos, w_down, grasp_axis, w_roll) ** 2)
            if cost < best_cost:
                best_q, best_cost = q, cost
        return best_q, float(np.linalg.norm(self.tcp(best_q) - target_pos))
