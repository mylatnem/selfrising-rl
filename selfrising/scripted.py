"""把参考动作(reference-learning.json)变成可执行的脚本策略.

这是整份工程的"对照组": 不开环、不学习, 只按时间轴把关节目标角播一遍。
它的成功率就是 RL 要超越的基线。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import REFERENCE_JSON


@dataclass
class Motion:
    name: str
    times: np.ndarray          # (K,) 秒
    targets: np.ndarray        # (K, 2) 弧度
    roll_pitch_deg: tuple[float, float]


class ScriptedController:
    """线性插值回放参考轨迹, 输出增量动作(与环境动作空间对齐)。"""

    def __init__(self, motion: str = "roll_pos", reference_path: str | Path | None = None) -> None:
        path = Path(reference_path or REFERENCE_JSON)
        data = json.loads(path.read_text(encoding="utf-8"))
        motions = {m["name"]: m for m in data["motions"]}
        if motion not in motions:
            raise ValueError(f"参考动作里没有 {motion!r}, 可选: {', '.join(motions)}")
        raw = motions[motion]

        times = np.array([float(k["time"]) for k in raw["keyframes"]], dtype=np.float64)
        targets = np.array([k["targets"] for k in raw["keyframes"]], dtype=np.float64)
        order = np.argsort(times)

        self.motion = Motion(
            name=motion,
            times=times[order],
            targets=targets[order],
            roll_pitch_deg=tuple(float(v) for v in raw.get("rollPitchDegrees", (0.0, 0.0))),
        )
        self.channels = data["channels"]
        self.action_delta = float(self.channels[0]["delta"])
        self.success = data.get("success", {})

    # ---------------------------------------------------------------- 轨迹
    def target_at(self, t: float) -> np.ndarray:
        """在 t 时刻的参考关节目标角。"""
        if t <= self.motion.times[0]:
            return self.motion.targets[0].copy()
        if t >= self.motion.times[-1]:
            return self.motion.targets[-1].copy()
        i = int(np.searchsorted(self.motion.times, t) - 1)
        t0, t1 = self.motion.times[i], self.motion.times[i + 1]
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return (1.0 - w) * self.motion.targets[i] + w * self.motion.targets[i + 1]

    def action(self, env) -> np.ndarray:
        """把"参考目标 - 当前目标"折算成 [-1, 1] 的增量动作。"""
        desired = self.target_at(env.time)
        delta = (desired - env.target) / max(self.action_delta, 1e-6)
        return np.clip(delta, -1.0, 1.0)


class PoseScriptedPolicy:
    """按当前姿态自动选用对应参考轨迹的脚本策略。

    可以直接当作 act_fn(obs, env) 传给 evaluate(), 用于和 RL 结果对比。
    """

    def __init__(self, reference_path: str | Path | None = None) -> None:
        self.reference_path = reference_path
        self._controllers: dict[str, ScriptedController] = {}

    def __call__(self, _obs, env) -> np.ndarray:
        pose = getattr(env, "current_pose", "roll_pos")
        ctrl = self._controllers.get(pose)
        if ctrl is None:
            ctrl = ScriptedController(motion=pose, reference_path=self.reference_path)
            self._controllers[pose] = ctrl
        return ctrl.action(env)
