#!/usr/bin/env python
"""自检脚本: 第一次运行工程时先跑它。

它不需要 torch / stable-baselines3, 只验证:
1. 依赖是否装好
2. 模型能否加载、名字能否对上
3. 环境的观测/动作维度、物理步进是否正常
4. 参考动作能不能在仿真里跑完一回合(判断模型与参考数据是否匹配)

任何一步失败都会给出中文提示, 不会抛一堆看不懂的栈。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

FAIL = "\033[31m[FAIL]\033[0m"
OK = "\033[32m[ OK ]\033[0m"
WARN = "\033[33m[WARN]\033[0m"


def main() -> int:
    problems: list[str] = []

    print("=" * 62)
    print("SelfRising RL 工程自检")
    print("=" * 62)

    # 1) 依赖 -------------------------------------------------------
    try:
        import gymnasium
        import mujoco

        print(f"{OK} mujoco {mujoco.__version__} / gymnasium {gymnasium.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 依赖导入失败: {exc}")
        print("       请先执行: pip install -r requirements.txt")
        return 1

    # 2) 模型 -------------------------------------------------------
    from selfrising.model import build_index, load_model
    from selfrising.paths import MODEL_XML

    print(f"\n模型文件: {MODEL_XML}")
    try:
        model = load_model(MODEL_XML)
        idx = build_index(model)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 模型加载/索引失败: {exc}")
        return 1

    print(f"{OK} nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep}")
    print(f"     基座 body={idx.base_body} 关节 qpos={idx.qpos_adr} dof={idx.dof_adr}")
    if abs(model.opt.timestep - 0.001) > 1e-9:
        print(f"{WARN} timestep 不是 1ms, 参考动作的节奏会偏")

    # 3) 环境 -------------------------------------------------------
    from selfrising.env import OBS_DIM, SelfRisingEnv

    try:
        env = SelfRisingEnv()
        obs, info = env.reset(seed=0)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 环境构建/复位失败: {exc}")
        return 1

    print(f"{OK} 环境: obs={env.observation_space.shape} act={env.action_space.shape} "
          f"frame_skip={env.frame_skip} 控制频率={1 / env.effective_dt:.0f}Hz")
    if env.observation_space.shape[0] != OBS_DIM:
        problems.append("观测维度与 OBS_DIM 不一致")
    if not np.all(np.isfinite(obs)):
        problems.append("复位后的观测出现 NaN/Inf")

    # 4) 随机动作跑几步 ---------------------------------------------
    try:
        total = 0.0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("观测或奖励出现 NaN/Inf")
            if terminated or truncated:
                obs, info = env.reset(seed=1)
        print(f"{OK} 随机动作 50 步正常, 累计奖励 {total:+.2f}, 当前倾角 {env.tilt_deg:.1f}°")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 随机动作步进失败: {exc}")
        problems.append(str(exc))

    # 5) 参考动作回放 ------------------------------------------------
    print("\n参考动作回放(每个姿态一回合):")
    from selfrising.poses import POSE_NAMES
    from selfrising.rollout import run_episode
    from selfrising.scripted import ScriptedController

    for pose in POSE_NAMES:
        try:
            penv = SelfRisingEnv(fixed_pose=pose, random_pose=False, pose_augmentation_deg=0.0)
            ctrl = ScriptedController(motion=pose)
            result = run_episode(penv, lambda obs, e: ctrl.action(e), seed=0)
            penv.close()
            flag = "成功" if result["success"] else "未成功"
            print(f"   {pose:<10} {flag}  用时 {result['seconds']:.2f}s  "
                  f"末态倾角 {result['final_tilt_deg']:.1f}°  回报 {result['return']:+.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"   {pose:<10} {FAIL} {exc}")
            problems.append(f"{pose}: {exc}")

    # 6) 结论 -------------------------------------------------------
    print("\n" + "=" * 62)
    if problems:
        print(f"{FAIL} 自检发现 {len(problems)} 个问题:")
        for p in problems:
            print(f"   - {p}")
        print("=" * 62)
        return 1

    print(f"{OK} 全部通过。下一步:")
    print("   python -m scripts.run_scripted          # 看脚本基线成功率")
    print("   python -m scripts.train_ppo --config configs/ppo_fast.json")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
