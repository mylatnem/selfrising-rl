"""评测: 按姿态分组统计成功率。

为什么要按姿态分开看?
因为"平均 75%"这种数字会骗人 —— 可能是三个姿态都会、一个姿态完全不会。
按姿态拆开才能看出策略是不是只学会了某一个方向。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .poses import POSE_NAMES
from .rollout import run_episode, summarize


def evaluate(
    act_fn: Callable[[np.ndarray, object], np.ndarray],
    make_env: Callable[[], object],
    *,
    poses: tuple[str, ...] = POSE_NAMES,
    episodes_per_pose: int = 5,
    seed: int = 0,
    env=None,
) -> dict:
    """act_fn(obs, env) -> action。返回 {per_pose, overall}。"""
    own_env = env is None
    env = env or make_env()
    per_pose: dict[str, dict] = {}

    for pose in poses:
        rows = [
            run_episode(env, act_fn, seed=seed + i, pose=pose)
            for i in range(episodes_per_pose)
        ]
        per_pose[pose] = summarize(rows)

    if own_env:
        env.close()

    all_success = [v["success_rate"] for v in per_pose.values()]
    overall = {
        "success_rate": float(np.mean(all_success)) if all_success else 0.0,
        "worst_pose_success_rate": float(np.min(all_success)) if all_success else 0.0,
        "mean_return": float(np.mean([v["mean_return"] for v in per_pose.values()])) if per_pose else 0.0,
    }
    return {"per_pose": per_pose, "overall": overall}


def format_report(result: dict) -> str:
    lines = [
        f"{'姿态':<12}{'成功率':>8}{'平均回报':>12}{'平均用时(s)':>14}",
        "-" * 46,
    ]
    for pose, stats in result["per_pose"].items():
        lines.append(
            f"{pose:<12}{stats['success_rate'] * 100:>7.0f}%{stats['mean_return']:>12.1f}{stats['mean_seconds']:>14.2f}"
        )
    o = result["overall"]
    lines.append("-" * 46)
    lines.append(f"{'总体':<12}{o['success_rate'] * 100:>7.0f}%{o['mean_return']:>12.1f}")
    lines.append(f"最差姿态成功率: {o['worst_pose_success_rate'] * 100:.0f}%")
    return "\n".join(lines)


def save_report(result: dict, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    if extra:
        payload["meta"] = extra
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sb3_predict_fn(model, deterministic: bool = True):
    """把 SB3 模型包成 act_fn(obs, env) -> action。"""

    def _act(obs, _env=None):
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action)

    return _act
