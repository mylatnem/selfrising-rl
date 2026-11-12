"""初始跌倒姿态: 四个方向, 与参考动作一一对应。

参考工程里定义的四个初态(roll/pitch 各 ±90°), 以及 15° 的随机扰动增强。
这里用纯 numpy 实现, 不依赖 mujoco, 方便单元测试。
"""

from __future__ import annotations

import math

import numpy as np

# name -> (roll_deg, pitch_deg)
POSES: dict[str, tuple[float, float]] = {
    "roll_pos": (90.0, 0.0),
    "roll_neg": (-90.0, 0.0),
    "pitch_pos": (0.0, 90.0),
    "pitch_neg": (0.0, -90.0),
}

POSE_NAMES: tuple[str, ...] = tuple(POSES)


def quat_from_euler(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """欧拉角(度) -> 四元数 (w, x, y, z), 与 MuJoCo free joint 的 qpos[3:7] 一致。

    采用 R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 的约定。
    """
    r, p, y = (math.radians(v) * 0.5 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def sample_pose(
    rng: np.random.Generator,
    pose: str | None = None,
    augmentation_deg: float = 15.0,
    yaw_range_deg: float = 20.0,
) -> tuple[str, np.ndarray, tuple[float, float]]:
    """随机抽一个初始姿态并加扰动。

    返回 (姿态名, 四元数, (roll_deg, pitch_deg))。
    augmentation_deg=0 时得到精确的跌倒姿态。
    """
    if pose is None:
        name = POSE_NAMES[int(rng.integers(len(POSE_NAMES)))]
    else:
        if pose not in POSES:
            raise ValueError(f"未知姿态 {pose!r}, 可选: {', '.join(POSE_NAMES)}")
        name = pose

    base_roll, base_pitch = POSES[name]
    roll = base_roll + float(rng.uniform(-augmentation_deg, augmentation_deg))
    pitch = base_pitch + float(rng.uniform(-augmentation_deg, augmentation_deg))
    yaw = float(rng.uniform(-yaw_range_deg, yaw_range_deg))
    return name, quat_from_euler(roll, pitch, yaw), (roll, pitch)
