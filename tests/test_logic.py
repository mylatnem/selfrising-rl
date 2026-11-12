#!/usr/bin/env python
"""纯逻辑单元测试: 不需要 MuJoCo, 也不需要 GPU。

    python tests/test_logic.py

覆盖: 四元数、姿态采样、参考轨迹插值、奖励函数、rollout/evaluate 的统计逻辑。
(这些是"算错了一眼看不出来"的部分, 值得单独测。)
真实物理的正确性由 scripts/selftest.py 在装了 MuJoCo 的机器上验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from selfrising.evaluate import evaluate, format_report
from selfrising.poses import POSES, quat_from_euler, sample_pose
from selfrising.reward import RewardConfig, compute_reward
from selfrising.rollout import run_episode, summarize
from selfrising.scripted import PoseScriptedPolicy, ScriptedController


def test_quat_from_euler() -> None:
    q = quat_from_euler(90, 0, 0)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    assert np.allclose(np.abs(q), np.abs(expected), atol=1e-9), q

    q = quat_from_euler(0, 90, 0)
    assert np.allclose(np.abs(q), np.abs([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]), atol=1e-9), q

    q = quat_from_euler(0, 0, 0)
    assert np.allclose(q, [1, 0, 0, 0]), q


def test_sample_pose() -> None:
    rng = np.random.default_rng(0)
    name, quat, (roll, pitch) = sample_pose(rng, pose="roll_pos", augmentation_deg=0.0, yaw_range_deg=0.0)
    assert name == "roll_pos"
    assert abs(roll - 90.0) < 1e-9 and abs(pitch - 0.0) < 1e-9
    assert np.allclose(np.abs(quat), np.abs(quat_from_euler(90, 0, 0)), atol=1e-9)

    # 同一个 seed 必须给出同一个姿态(可复现性)
    a = sample_pose(np.random.default_rng(42), augmentation_deg=15.0)
    b = sample_pose(np.random.default_rng(42), augmentation_deg=15.0)
    assert a[0] == b[0] and np.allclose(a[1], b[1])

    # 四个姿态都能采到
    seen = {sample_pose(np.random.default_rng(i), augmentation_deg=0.0)[0] for i in range(50)}
    assert seen == set(POSES), seen


def test_scripted_interpolation() -> None:
    ctrl = ScriptedController(motion="roll_pos")
    first = ctrl.target_at(-1.0)
    last = ctrl.target_at(99.0)
    assert np.allclose(first, ctrl.motion.targets[0])
    assert np.allclose(last, ctrl.motion.targets[-1])

    # 中点必须落在相邻关键帧之间
    t0, t1 = ctrl.motion.times[0], ctrl.motion.times[1]
    mid = ctrl.target_at(0.5 * (t0 + t1))
    lo = np.minimum(ctrl.motion.targets[0], ctrl.motion.targets[1])
    hi = np.maximum(ctrl.motion.targets[0], ctrl.motion.targets[1])
    assert np.all(mid >= lo - 1e-9) and np.all(mid <= hi + 1e-9), mid


def test_scripted_action_clip() -> None:
    class FakeEnv:
        time = 0.3
        target = np.zeros(2)

    ctrl = ScriptedController(motion="roll_pos")
    a = ctrl.action(FakeEnv())
    assert a.shape == (2,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_reward_terms() -> None:
    cfg = RewardConfig()
    total, terms = compute_reward(
        upright=1.0, prev_upright=0.5, height=0.08, action=np.zeros(2),
        prev_action=np.zeros(2), joint_vel=np.zeros(2), success=False, failed=False, cfg=cfg,
    )
    assert "upright" in terms and "progress" in terms
    assert abs(terms["upright"] - 1.0) < 1e-9
    assert abs(terms["progress"] - 0.2 * 0.5) < 1e-9
    assert abs(total - sum(terms.values())) < 1e-9

    _, t_success = compute_reward(
        upright=1.0, prev_upright=1.0, height=0.08, action=np.zeros(2),
        prev_action=np.zeros(2), joint_vel=np.zeros(2), success=True, failed=False, cfg=cfg,
    )
    assert t_success["success"] == cfg.w_success

    _, t_fail = compute_reward(
        upright=0.0, prev_upright=0.0, height=-0.2, action=np.zeros(2),
        prev_action=np.zeros(2), joint_vel=np.zeros(2), success=False, failed=True, cfg=cfg,
    )
    assert t_fail["fail"] == cfg.w_fail


class FakeEnv:
    """最小可用的假环境, 用来测 rollout/evaluate 的统计逻辑。"""

    effective_dt = 0.02

    def __init__(self, succeed_at: int | None = 3, timeout_steps: int = 10) -> None:
        self.succeed_at = succeed_at
        self.timeout_steps = timeout_steps
        self.steps_done = 0
        self.time = 0.0
        self.current_pose = "roll_pos"
        self.target = np.zeros(2)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.steps_done = 0
        self.time = 0.0
        if options and options.get("pose"):
            self.current_pose = options["pose"]
        return np.zeros(10, dtype=np.float32), {"pose": self.current_pose, "tilt_deg": 90.0}

    def step(self, action):
        self.steps_done += 1
        self.time += self.effective_dt
        success = self.succeed_at is not None and self.steps_done >= self.succeed_at
        truncated = (not success) and self.steps_done >= self.timeout_steps
        info = {
            "is_success": success,
            "tilt_deg": 0.0 if success else 90.0,
            "time": self.time,
            "pose": self.current_pose,
        }
        return np.zeros(10, dtype=np.float32), 1.0, bool(success), bool(truncated), info

    def close(self) -> None:
        self.closed = True


def test_run_episode() -> None:
    env = FakeEnv(succeed_at=3)
    result = run_episode(env, lambda obs, e: np.zeros(2), seed=0, pose="pitch_pos")
    assert result["success"] is True
    assert result["steps"] == 3
    assert abs(result["seconds"] - 0.06) < 1e-9
    assert result["pose"] == "pitch_pos"

    env = FakeEnv(succeed_at=None)
    result = run_episode(env, lambda obs, e: np.zeros(2), seed=0)
    assert result["success"] is False


def test_summarize() -> None:
    rows = [
        {"success": True, "return": 10.0, "seconds": 2.0},
        {"success": False, "return": 4.0, "seconds": 14.0},
    ]
    s = summarize(rows)
    assert s["episodes"] == 2
    assert abs(s["success_rate"] - 0.5) < 1e-9
    assert abs(s["mean_return"] - 7.0) < 1e-9


def test_evaluate_and_report() -> None:
    factory = lambda: FakeEnv(succeed_at=2)  # noqa: E731
    result = evaluate(
        lambda obs, env: np.zeros(2), factory, poses=("roll_pos", "pitch_neg"), episodes_per_pose=3
    )
    assert set(result["per_pose"]) == {"roll_pos", "pitch_neg"}
    assert abs(result["overall"]["success_rate"] - 1.0) < 1e-9
    text = format_report(result)
    assert "roll_pos" in text and "总体" in text


def test_pose_scripted_policy() -> None:
    policy = PoseScriptedPolicy()
    env = FakeEnv()
    policy(np.zeros(10), env)
    action = policy(np.zeros(10), env)
    assert action.shape == (2,)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
