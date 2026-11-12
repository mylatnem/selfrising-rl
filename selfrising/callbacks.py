"""SB3 回调: 定期按姿态评测、记录曲线、保存最优模型。

直接看 train/episode_reward 不够 —— 回报高不等于成功率涨。
这里每个评测点都把成功率拆到四个姿态并写进 TensorBoard。
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from .evaluate import evaluate, sb3_predict_fn


class PoseEvalCallback(BaseCallback):
    def __init__(
        self,
        make_env,
        eval_freq: int = 20_000,
        episodes_per_pose: int = 3,
        best_model_path: str | Path | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.make_env = make_env
        self.eval_freq = int(eval_freq)
        self.episodes_per_pose = int(episodes_per_pose)
        self.best_model_path = Path(best_model_path) if best_model_path else None
        self.best_success = -1.0
        self._eval_env = None

    def _on_training_start(self) -> None:
        self._eval_env = self.make_env()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        result = evaluate(
            sb3_predict_fn(self.model),
            self.make_env,
            episodes_per_pose=self.episodes_per_pose,
            seed=10_000,
            env=self._eval_env,
        )
        overall = result["overall"]
        self.logger.record("eval/success_rate", overall["success_rate"])
        self.logger.record("eval/worst_pose_success_rate", overall["worst_pose_success_rate"])
        self.logger.record("eval/mean_return", overall["mean_return"])
        for pose, stats in result["per_pose"].items():
            self.logger.record(f"eval/{pose}_success", stats["success_rate"])

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps:>8} "
                f"success={overall['success_rate'] * 100:5.1f}% "
                f"worst={overall['worst_pose_success_rate'] * 100:5.1f}% "
                + " ".join(f"{p}={s['success_rate'] * 100:.0f}%" for p, s in result["per_pose"].items())
            )

        if self.best_model_path is not None and overall["success_rate"] > self.best_success:
            self.best_success = overall["success_rate"]
            self.model.save(str(self.best_model_path))
            if self.verbose:
                print(f"[eval] 新的最优模型 -> {self.best_model_path} (success={self.best_success * 100:.0f}%)")
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
