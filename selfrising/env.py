"""Gymnasium 环境: 把 MuJoCo 模型包装成一个 RL 问题。

控制结构(和实物一致):
    timestep = 1 ms  -> 每个控制步跑 frame_skip=20 次物理步进 -> 50 Hz 决策
动作: 2 维, [-1, 1], 表示"在上一目标角基础上再转多少"(增量位置控制)
观测: 10 维, 见 OBS_FIELDS
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .model import (
    build_index,
    body_tilt_deg,
    body_upright,
    gravity_in_body_frame,
    load_model,
)
from .paths import MODEL_XML
from .poses import POSE_NAMES, sample_pose
from .reward import RewardConfig, compute_reward

# 观测分量说明(写进文档, 也方便调试时对照)
OBS_FIELDS = (
    "g_b.x", "g_b.y", "g_b.z",          # 重力在机体系的分量(姿态)
    "q_servo1", "q_servo2",             # 两个关节角
    "dq_servo1", "dq_servo2",           # 两个关节角速度
    "target_servo1", "target_servo2",   # 上一控制目标角
    "base_height",                      # 基座高度
)
OBS_DIM = len(OBS_FIELDS)


class SelfRisingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        control_dt: float = 0.02,
        action_delta: float = 0.08,
        action_limit: float = 1.55,
        reset_height: float = 0.08,
        settle_seconds: float = 0.2,
        random_pose: bool = True,
        fixed_pose: str | None = None,
        pose_augmentation_deg: float = 15.0,
        domain_randomize: bool = False,
        dr_mass: float = 0.10,
        dr_friction: float = 0.20,
        dr_gain: float = 0.10,
        action_delay: int = 0,
        obs_noise: float = 0.0,
        max_episode_seconds: float = 14.0,
        reward: dict | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.model = load_model(xml_path or MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.idx = build_index(self.model)

        # ---- 时间与控制 ----
        self.model.opt.timestep = float(self.model.opt.timestep)
        self.control_dt = float(control_dt)
        self.frame_skip = int(round(self.control_dt / self.model.opt.timestep))
        if self.frame_skip < 1:
            raise ValueError(
                f"control_dt={self.control_dt} 小于物理步长 {self.model.opt.timestep}"
            )
        self.effective_dt = self.frame_skip * self.model.opt.timestep

        self.action_delta = float(action_delta)
        self.action_limit = float(action_limit)
        self.reset_height = float(reset_height)
        self.settle_seconds = float(settle_seconds)

        # ---- 初始姿态 ----
        self.random_pose = bool(random_pose)
        self.fixed_pose = fixed_pose
        if fixed_pose is not None:
            self.random_pose = False
            if fixed_pose not in POSE_NAMES:
                raise ValueError(f"fixed_pose 必须是 {POSE_NAMES} 之一, 收到 {fixed_pose!r}")
        self.pose_augmentation_deg = float(pose_augmentation_deg)

        # ---- 域随机化 / 噪声 / 延迟 ----
        self.domain_randomize = bool(domain_randomize)
        self.dr_mass = float(dr_mass)
        self.dr_friction = float(dr_friction)
        self.dr_gain = float(dr_gain)
        self.action_delay = int(action_delay)
        self.obs_noise = float(obs_noise)

        self.max_episode_seconds = float(max_episode_seconds)

        # ---- 奖励 ----
        self.reward_cfg = RewardConfig(**(reward or {}))

        # 保存模型原始参数, 供域随机化复位
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()
        self._base_gain = self.model.actuator_gainprm.copy()

        # ---- 状态缓冲 ----
        self.time = 0.0
        self.step_count = 0
        self.target = np.zeros(self.idx.n_joints, dtype=np.float64)
        self.prev_action = np.zeros(self.idx.n_joints, dtype=np.float64)
        self.prev_upright = 1.0
        self.hold_time = 0.0
        self.no_contact_time = 0.0
        self.current_pose = "roll_pos"
        self._delay_buf = [np.zeros(self.idx.n_joints) for _ in range(self.action_delay)]
        self._last_terms: dict[str, float] = {}

        # ---- 空间 ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.idx.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1.6, -1.6, -20, -20, -1.6, -1.6, -0.1], dtype=np.float32),
            high=np.array([1, 1, 1, 1.6, 1.6, 20, 20, 1.6, 1.6, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------------ 观测
    def _obs(self) -> np.ndarray:
        g_b = gravity_in_body_frame(self.data, self.idx.base_body)
        q = self.data.qpos[list(self.idx.qpos_adr)]
        dq = self.data.qvel[list(self.idx.dof_adr)]
        height = float(self.data.qpos[self.idx.free_qpos_adr + 2])

        obs = np.concatenate([g_b, q, dq, self.target, [height]]).astype(np.float64)
        if self.obs_noise > 0:
            scale = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.05, 0.05, 0.0, 0.0, 0.002])
            obs = obs + self.np_random.normal(0.0, 1.0, OBS_DIM) * self.obs_noise * scale
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    @property
    def upright(self) -> float:
        return body_upright(self.data, self.idx.base_body)

    @property
    def tilt_deg(self) -> float:
        return body_tilt_deg(self.data, self.idx.base_body)

    # ------------------------------------------------------------------ 复位
    def _apply_domain_randomization(self) -> None:
        rng = self.np_random
        if not self.domain_randomize:
            self.model.body_mass[:] = self._base_mass
            self.model.geom_friction[:] = self._base_friction
            self.model.actuator_gainprm[:] = self._base_gain
            return
        self.model.body_mass[:] = self._base_mass * (1.0 + rng.uniform(-self.dr_mass, self.dr_mass))
        self.model.geom_friction[:, 0] = self._base_friction[:, 0] * (
            1.0 + rng.uniform(-self.dr_friction, self.dr_friction)
        )
        self.model.actuator_gainprm[:, 0] = self._base_gain[:, 0] * (
            1.0 + rng.uniform(-self.dr_gain, self.dr_gain)
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._apply_domain_randomization()

        pose_opt = None
        if options and options.get("pose"):
            pose_opt = str(options["pose"])
        elif self.fixed_pose:
            pose_opt = self.fixed_pose

        name, quat, _ = sample_pose(
            self.np_random,
            pose=pose_opt,
            augmentation_deg=self.pose_augmentation_deg if self.random_pose or pose_opt is None else 0.0,
        )
        self.current_pose = name

        fq = self.idx.free_qpos_adr
        self.data.qpos[fq + 0] = self.np_random.uniform(-0.01, 0.01)
        self.data.qpos[fq + 1] = self.np_random.uniform(-0.01, 0.01)
        self.data.qpos[fq + 2] = self.reset_height
        self.data.qpos[fq + 3 : fq + 7] = quat
        self.data.qpos[list(self.idx.qpos_adr)] = self.np_random.uniform(-0.15, 0.15, self.idx.n_joints)
        self.data.qvel[:] = 0.0

        self.target[:] = 0.0
        self.data.ctrl[list(self.idx.actuator_ids)] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 落定: 用零目标保持一小段时间, 让机器人先稳定在跌倒姿态上
        for _ in range(max(1, int(round(self.settle_seconds / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.time = 0.0
        self.step_count = 0
        self.hold_time = 0.0
        self.no_contact_time = 0.0
        self.prev_action = np.zeros(self.idx.n_joints)
        self.prev_upright = self.upright
        self._delay_buf = [np.zeros(self.idx.n_joints) for _ in range(self.action_delay)]
        self._last_terms = {}

        return self._obs(), {"pose": self.current_pose, "tilt_deg": self.tilt_deg}

    # ------------------------------------------------------------------ 步进
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(self.idx.n_joints), -1.0, 1.0)

        # 动作延迟: 模拟真机上的通信/执行延迟
        if self.action_delay > 0:
            self._delay_buf.append(action)
            if len(self._delay_buf) > self.action_delay:
                action = self._delay_buf.pop(0)

        self.target = np.clip(
            self.target + action * self.action_delta, -self.action_limit, self.action_limit
        )
        self.data.ctrl[list(self.idx.actuator_ids)] = self.target

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.time += self.effective_dt
        self.step_count += 1

        upright = self.upright
        tilt = self.tilt_deg
        height = float(self.data.qpos[self.idx.free_qpos_adr + 2])
        dq = self.data.qvel[list(self.idx.dof_adr)]

        # 成功: 倾角达标并保持 hold_seconds
        upright_enough = tilt <= self.reward_cfg.success_tilt_deg
        self.hold_time = self.hold_time + self.effective_dt if upright_enough else 0.0
        success = self.hold_time >= self.reward_cfg.hold_seconds

        # 失败: 长时间不接触地面(被弹飞/穿模), 或数值炸掉
        # 注: qpos[2] 是 foot 这个 body 坐标系原点的 z, 不是机器人的高度
        #     (原点在几何体之外), 所以不能拿它判断"掉出地面", 用接触点数更可靠。
        in_contact = self.data.ncon > 0
        self.no_contact_time = 0.0 if in_contact else self.no_contact_time + self.effective_dt
        failed = bool(
            self.no_contact_time > 0.5
            or not np.all(np.isfinite(self.data.qpos))
            or np.max(np.abs(self.data.qvel)) > 100.0
        )

        reward, terms = compute_reward(
            upright=upright,
            prev_upright=self.prev_upright,
            height=height,
            action=action,
            prev_action=self.prev_action,
            joint_vel=dq,
            success=success,
            failed=failed,
            cfg=self.reward_cfg,
        )
        self._last_terms = terms
        self.prev_upright = upright
        self.prev_action = action.copy()

        terminated = bool(success or failed)
        truncated = bool(self.time >= self.max_episode_seconds) and not terminated

        info = {
            "pose": self.current_pose,
            "is_success": success,
            "tilt_deg": tilt,
            "upright": upright,
            "time": self.time,
            "reward_terms": terms,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ 渲染
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
