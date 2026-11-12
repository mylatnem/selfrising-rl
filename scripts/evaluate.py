#!/usr/bin/env python
"""评测入口: 同一套指标, 既能评脚本基线, 也能评训练好的模型。

    python -m scripts.evaluate --episodes 10                     # 脚本基线
    python -m scripts.evaluate --model runs/ppo_fast/best_model  # RL 模型
    python -m scripts.evaluate --model ... --json runs/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfrising.env import SelfRisingEnv
from selfrising.evaluate import evaluate, format_report, save_report
from selfrising.poses import POSE_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测脚本基线或训练好的模型")
    parser.add_argument("--model", default=None, help="SB3 模型路径(可省 .zip), 省略则评测脚本基线")
    parser.add_argument("--episodes", type=int, default=10, help="每个姿态的回合数")
    parser.add_argument("--poses", default=None, help="逗号分隔, 默认四个姿态")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="结果写入的 json 路径")
    parser.add_argument("--env-json", default=None, help='环境参数覆盖, 如 \'{"action_delay":1}\'')
    args = parser.parse_args(argv)

    env_kwargs = json.loads(args.env_json) if args.env_json else {}
    poses = tuple(args.poses.split(",")) if args.poses else POSE_NAMES

    def make_env():
        return SelfRisingEnv(**env_kwargs)

    if args.model:
        from stable_baselines3 import PPO

        from selfrising.evaluate import sb3_predict_fn

        model_path = args.model[:-4] if args.model.endswith(".zip") else args.model
        model = PPO.load(model_path, device="cpu")
        act_fn = sb3_predict_fn(model)
        label = f"RL 模型: {model_path}"
        meta = {"model": model_path, "episodes": args.episodes, "env": env_kwargs}
    else:
        from selfrising.scripted import PoseScriptedPolicy

        act_fn = PoseScriptedPolicy()
        label = "脚本基线(参考动作回放)"
        meta = {"model": "scripted", "episodes": args.episodes, "env": env_kwargs}

    result = evaluate(act_fn, make_env, poses=poses, episodes_per_pose=args.episodes, seed=args.seed)
    print(f"\n{label}")
    print(f"每个姿态 {args.episodes} 回合, seed={args.seed}\n")
    print(format_report(result))

    if args.json:
        save_report(result, args.json, extra=meta)
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
