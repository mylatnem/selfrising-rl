#!/usr/bin/env python
"""脚本基线评测: 不用任何学习, 只按参考动作回放。

这是你训练 PPO 之前必须先看到的数字 —— 如果脚本都站不起来,
说明模型或参考动作有问题的可能性更大, 而不是"RL 学不会"。

    python -m scripts.run_scripted
    python -m scripts.run_scripted --episodes 20 --json runs/scripted.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfrising.env import SelfRisingEnv
from selfrising.evaluate import evaluate, format_report, save_report
from selfrising.poses import POSE_NAMES
from selfrising.scripted import PoseScriptedPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脚本基线评测")
    parser.add_argument("--episodes", type=int, default=5, help="每个姿态回合数")
    parser.add_argument("--poses", default=None, help="逗号分隔, 默认四个姿态")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    poses = tuple(args.poses.split(",")) if args.poses else POSE_NAMES

    def make_env():
        return SelfRisingEnv(random_pose=False, pose_augmentation_deg=0.0)

    result = evaluate(
        PoseScriptedPolicy(), make_env, poses=poses, episodes_per_pose=args.episodes, seed=args.seed
    )
    print("\n脚本基线(参考动作回放, 无扰动)")
    print(f"每个姿态 {args.episodes} 回合\n")
    print(format_report(result))

    if args.json:
        save_report(result, args.json, extra={"model": "scripted", "episodes": args.episodes})
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
