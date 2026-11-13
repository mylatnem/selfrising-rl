#!/usr/bin/env python
"""PPO 训练入口。

用法(Linux):
    python -m scripts.train_ppo --config configs/ppo_fast.json
    python -m scripts.train_ppo --config configs/ppo_full.json --timesteps 2000000
    python -m scripts.train_ppo --timesteps 200000 --n-envs 4 --pose roll_pos

所有超参都写在 configs/*.json 里, 命令行可以覆盖常用几项。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from selfrising.callbacks import PoseEvalCallback
from selfrising.env import SelfRisingEnv
from selfrising.evaluate import evaluate, format_report, save_report, sb3_predict_fn
from selfrising.paths import CONFIGS_DIR, runs_dir


class EnvFactory:
    """可被 pickle 的 env 工厂(SubprocVecEnv 需要)。"""

    def __init__(self, env_kwargs: dict) -> None:
        self.env_kwargs = env_kwargs

    def __call__(self):
        return Monitor(SelfRisingEnv(**self.env_kwargs))


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIGS_DIR / p.name if (CONFIGS_DIR / p.name).exists() else Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def build_model_args(cfg: dict) -> dict:
    return {
        "learning_rate": float(cfg.get("learning_rate", 5e-4)),
        "n_steps": int(cfg.get("n_steps", 48)),
        "batch_size": int(cfg.get("batch_size", 256)),
        "n_epochs": int(cfg.get("n_epochs", 5)),
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "clip_range": float(cfg.get("clip_range", 0.2)),
        "ent_coef": float(cfg.get("ent_coef", 0.0)),
        "vf_coef": float(cfg.get("vf_coef", 1.0)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "target_kl": cfg.get("target_kl"),
        "policy_kwargs": {
            "net_arch": list(cfg.get("net_arch", [64, 64])),
            "activation_fn": torch.nn.ELU,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SelfRising PPO 训练")
    parser.add_argument("--config", default="ppo_fast.json", help="configs/ 里的配置文件名")
    parser.add_argument("--timesteps", type=int, default=None, help="总步数, 覆盖配置")
    parser.add_argument("--n-envs", type=int, default=None, help="并行环境数, 覆盖配置")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pose", default=None, help="只在这一个姿态上训练(课程学习第一步)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None, help="从 .zip 继续训练")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_name = args.run_name or cfg.get("name", "ppo")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    n_envs = args.n_envs or int(cfg.get("n_envs", 8))
    total_timesteps = args.timesteps or int(cfg.get("total_timesteps", 300_000))

    env_kwargs = dict(cfg.get("env", {}))
    if args.pose:
        env_kwargs.update({"fixed_pose": args.pose, "random_pose": False})

    run_dir = runs_dir(run_name)
    env_cfg = cfg.get("env", {})
    aug = float(env_cfg.get("pose_augmentation_deg", 15.0))
    print("=" * 62)
    print(f"运行目录 : {run_dir}")
    print(f"姿态     : {args.pose or f'四个姿态随机(±{aug:g}° 扰动)'}")
    print(f"并行环境 : {n_envs}")
    print(f"总步数   : {total_timesteps:,}")
    # 这几项决定 sim2real 的难度, 打印出来免得对照配置时看错
    if env_cfg.get("domain_randomize") or env_cfg.get("action_delay") or env_cfg.get("obs_noise"):
        print(
            f"真机扰动 : 域随机化={env_cfg.get('domain_randomize', False)} "
            f"动作延迟={env_cfg.get('action_delay', 0)} 观测噪声={env_cfg.get('obs_noise', 0.0)}"
        )
    print("=" * 62)

    factory = EnvFactory(env_kwargs)
    if n_envs == 1:
        vec_env = DummyVecEnv([factory])
    else:
        vec_env = SubprocVecEnv([factory] * n_envs, start_method="forkserver")
    vec_env.seed(seed)

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, device=args.device)
        print(f"从 {args.resume} 继续训练")
    else:
        model = PPO(
            cfg.get("policy", "MlpPolicy"),
            vec_env,
            verbose=1,
            seed=seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
            **build_model_args(cfg),
        )

    callbacks = [
        PoseEvalCallback(
            EnvFactory(env_kwargs),
            eval_freq=int(cfg.get("eval_freq", 20_000)),
            episodes_per_pose=int(cfg.get("eval_episodes_per_pose", 3)),
            best_model_path=run_dir / "best_model",
        ),
        CheckpointCallback(
            save_freq=max(10_000 // max(n_envs, 1), 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"\n模型已保存: {final_path}.zip")

    # 训练结束做一次完整评估
    result = evaluate(
        sb3_predict_fn(model),
        EnvFactory(env_kwargs),
        episodes_per_pose=int(cfg.get("final_eval_per_pose", 10)),
        seed=777,
    )
    print("\n" + format_report(result))
    save_report(result, run_dir / "eval_final.json", extra={"timesteps": total_timesteps, "n_envs": n_envs})
    print(f"\n评估结果: {run_dir / 'eval_final.json'}")
    print(f"查看曲线: tensorboard --logdir {run_dir / 'tb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
