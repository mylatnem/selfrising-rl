"""统一的单回合 rollout, 供脚本基线、评估、调试共用。"""

from __future__ import annotations

from typing import Callable

import numpy as np


def run_episode(
    env,
    act_fn: Callable[[np.ndarray, object], np.ndarray],
    *,
    seed: int | None = None,
    pose: str | None = None,
    max_steps: int | None = None,
) -> dict:
    """跑一个完整回合。

    act_fn(obs, env) -> action, 需要 env 时(如脚本策略要读 env.time)直接取。
    """
    options = {"pose": pose} if pose else None
    obs, info = env.reset(seed=seed, options=options)

    total_reward = 0.0
    steps = 0
    while True:
        action = act_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "return": total_reward,
        "success": bool(info.get("is_success", False)),
        "seconds": float(info.get("time", steps * getattr(env, "effective_dt", 0.02))),
        "steps": steps,
        "final_tilt_deg": float(info.get("tilt_deg", 90.0)),
        "pose": info.get("pose", pose or "random"),
    }


def summarize(rows: list[dict]) -> dict:
    """把多次 rollout 汇成一行统计。"""
    if not rows:
        return {"episodes": 0, "success_rate": 0.0, "mean_return": 0.0, "mean_seconds": 0.0}
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
    }
