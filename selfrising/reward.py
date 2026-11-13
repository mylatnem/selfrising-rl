"""奖励函数: 拆成可读的若干项, 每一项都有明确目的。

设计奖励时先问三个问题:
1. 这一项在鼓励什么行为?
2. 策略能怎么"作弊"来白拿这一项?
3. 去掉它, 学出来的动作会变成什么样?

默认权重参考了 StackForce 导出配置里的 recovery_* 项(upright=1, progress=0.2,
action_rate=-0.02), 并补了高度与关节速度两项。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RewardConfig:
    # 站直奖励: 基座 z 轴与世界 z 轴的点积, [-1, 1]
    w_upright: float = 1.0
    # 高度: 偏离目标高度(0.08 m)的二次惩罚
    w_height: float = 0.5
    height_scale: float = 100.0
    height_target: float = 0.08
    # 进展: 只在"比上一步更直"时给分, 防止原地抖动白拿分
    w_progress: float = 0.2
    # 动作变化率: 抑制高频抖动, 也是上真机最关键的平滑项
    w_action_rate: float = -0.02
    # 关节速度: 抑制甩动
    w_joint_vel: float = -0.001
    # 成功/失败
    w_success: float = 10.0
    w_fail: float = -5.0
    # 成功判定
    # 参考 StackForce reference-learning.json 的 success 段: 除了倾角和保持时间,
    # 还要求两个关节都回到接近 0 (goal 姿态) 且基座基本静止, 否则"站着但还在扭"也算成功。
    success_tilt_deg: float = 14.32
    success_joint_tol: float = 0.12      # rad, 关节角与 0 的允许误差
    success_base_speed: float = 0.25     # m/s, 基座线速度上限
    hold_seconds: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_reward(
    *,
    upright: float,
    prev_upright: float,
    height: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    joint_vel: np.ndarray,
    success: bool,
    failed: bool,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """返回 (总奖励, 各项明细)。明细会写进 info, 便于排查"为什么学不会"。"""
    terms: dict[str, float] = {}

    terms["upright"] = cfg.w_upright * float(upright)
    terms["height"] = -cfg.w_height * cfg.height_scale * (float(height) - cfg.height_target) ** 2
    terms["progress"] = cfg.w_progress * max(0.0, float(upright) - float(prev_upright))
    terms["action_rate"] = cfg.w_action_rate * float(np.sum((np.asarray(action) - np.asarray(prev_action)) ** 2))
    terms["joint_vel"] = cfg.w_joint_vel * float(np.sum(np.asarray(joint_vel) ** 2))

    if success:
        terms["success"] = cfg.w_success
    if failed:
        terms["fail"] = cfg.w_fail

    total = float(sum(terms.values()))
    return total, terms
