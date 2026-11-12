#!/usr/bin/env python
"""可视化: 在 MuJoCo 窗口里看策略怎么站起来。

    python -m scripts.play                                   # 脚本基线
    python -m scripts.play --model runs/ppo_fast/best_model  # 训练好的模型
    python -m scripts.play --pose pitch_neg --episodes 3

注意: 需要图形界面。在无显示的 Linux 虚拟机上请用 --record 导出 GIF 后再看。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfrising.env import SelfRisingEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MuJoCo 可视化")
    parser.add_argument("--model", default=None, help="SB3 模型路径, 省略则用脚本基线")
    parser.add_argument("--pose", default=None, help="固定姿态, 省略则四个姿态循环")
    parser.add_argument("--episodes", type=int, default=2, help="每个姿态播几个回合")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", default=None, help="导出 gif 路径(替代窗口)")
    parser.add_argument("--slowdown", type=float, default=1.0, help=">1 放慢播放")
    args = parser.parse_args(argv)

    if args.model:
        from stable_baselines3 import PPO

        from selfrising.evaluate import sb3_predict_fn

        model = PPO.load(args.model[:-4] if args.model.endswith(".zip") else args.model, device="cpu")
        act_fn = sb3_predict_fn(model)
        label = "RL 策略"
    else:
        from selfrising.scripted import PoseScriptedPolicy

        act_fn = PoseScriptedPolicy()
        label = "脚本基线"

    env = SelfRisingEnv(random_pose=False, pose_augmentation_deg=0.0)
    poses = [args.pose] if args.pose else list(("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"))

    print(f"{label}, 姿态: {poses}")
    frames: list = []

    if args.record:
        import imageio.v2 as imageio

        for pose in poses:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + ep, options={"pose": pose})
                done = False
                while not done:
                    obs, _r, term, trunc, info = env.step(act_fn(obs, env))
                    frames.append(env.render())
                    done = term or trunc
                print(f"  {pose} 回合{ep + 1}: {'成功' if info['is_success'] else '失败'} "
                      f"{info['time']:.2f}s")
        out = Path(args.record)
        imageio.mimsave(str(out), frames, fps=int(round(1 / env.effective_dt)))
        print(f"已导出 {out} ({len(frames)} 帧)")
        env.close()
        return 0

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for pose in poses:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + ep, options={"pose": pose})
                done = False
                while not done and viewer.is_running():
                    start = time.time()
                    obs, _r, term, trunc, info = env.step(act_fn(obs, env))
                    viewer.sync()
                    done = term or trunc
                    wait = env.effective_dt * args.slowdown - (time.time() - start)
                    if wait > 0:
                        time.sleep(wait)
                if not viewer.is_running():
                    break
                print(f"  {pose} 回合{ep + 1}: {'成功' if info['is_success'] else '失败'} "
                      f"{info['time']:.2f}s")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
