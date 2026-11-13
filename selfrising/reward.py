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
    # 高度项默认关闭。
    # 曾经写 -0.5, 但 height 取的是 qpos[2] —— 那是 foot 这个 body 坐标系原点的 z,
    # 原点在几何体之外, 站直时反而会掉到 -0.05, 于是"站起来"每步要挨 0.85 的罚,
    # 策略学会的是"躺着别动"。要用这个量必须先把它换成有物理意义的高度。
    w_height: float = 0.0
    height_scale: float = 100.0
    height_target: float = 0.08
    # 终态整形: 站直之后把两个关节收回 0 (成功判定的 goal 姿态)。
    # 形状用"线性斜坡"而不是高斯: 高斯在 1.5 rad 处已经衰减到 1e-17, 梯度为零,
    # 策略会站在 q=±1.57(限位) 上安心拿 upright 分, 永远不知道还有这一项。
    # 现在 q²/(2*limit²) 让奖励在全程都有梯度, 越靠近 0 越赚。
    # 用 upright² 门控: 只在"已经基本站直"时才给, 避免策略为了拿分把手臂折起来。
    w_joint_goal: float = 0.75
    joint_limit: float = 1.5708
    # 进展: 只在"比上一步更直"时给分, 防止原地抖动白拿分
    w_progress: float = 0.2
    # 动作变化率: 抑制高频抖动, 也是上真机最关键的平滑项
    w_action_rate: float = -0.02
    # 关节速度: 抑制甩动
    w_joint_vel: float = -0.001
    # 成功/失败
    w_success: float = 10.0
    # 关键: 回合是"成功即结束"的, 而奖励是每步给的。提前成功 = 主动放弃后面每步的分,
    # 于是"差一点站住、撑满全场"反而更划算(实测 1154 分 vs 246 分), 策略会理性地拒绝成功。
    # 这个系数让成功时把"剩余步数 × 每步满分"一次性补给你, 成功就不再比赖着差。
    # 设为 0 可关闭(退回旧行为)。
    w_success_horizon: float = 1.0
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
    joints: np.ndarray | None = None,
    remaining_steps: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """返回 (总奖励, 各项明细)。明细会写进 info, 便于排查"为什么学不会"。"""
    terms: dict[str, float] = {}

    terms["upright"] = cfg.w_upright * float(upright)
    terms["height"] = -cfg.w_height * cfg.height_scale * (float(height) - cfg.height_target) ** 2
    terms["progress"] = cfg.w_progress * max(0.0, float(upright) - float(prev_upright))
    terms["action_rate"] = cfg.w_action_rate * float(np.sum((np.asarray(action) - np.asarray(prev_action)) ** 2))
    terms["joint_vel"] = cfg.w_joint_vel * float(np.sum(np.asarray(joint_vel) ** 2))

    if joints is not None:
        q = np.asarray(joints, dtype=np.float64)
        gate = max(0.0, float(upright)) ** 2
        # 1.0 (关节在 0) -> 0.0 (关节顶到限位), 线性、处处有梯度
        closeness = 1.0 - float(np.sum(q * q)) / (2.0 * cfg.joint_limit ** 2)
        terms["joint_goal"] = cfg.w_joint_goal * gate * max(0.0, closeness)

    if success:
        # 成功即结束, 用"剩余步数 × 每步满分"补偿被砍掉的未来收益, 否则早点成功是亏的
        ceiling = cfg.w_upright + cfg.w_joint_goal
        terms["success"] = cfg.w_success + cfg.w_success_horizon * max(0.0, remaining_steps) * ceiling
    if failed:
        terms["fail"] = cfg.w_fail

    total = float(sum(terms.values()))
    return total, terms
