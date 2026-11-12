"""MuJoCo 模型加载与索引解析.

设计原则: 所有维度/索引都从模型里按"名字"查出来, 绝不硬编码 7/9 这种数字。
这样即使以后换模型、加关节, 环境代码也不用改。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = ("servo1_joint", "servo2_joint")
ACTUATOR_NAMES = ("servo1_pos", "servo2_pos")
BASE_BODY_NAME = "foot"


@dataclass(frozen=True)
class ModelIndex:
    """模型里各处关键索引, 环境用它来读写 qpos/qvel/ctrl。"""

    base_body: int
    free_joint: int
    free_qpos_adr: int
    joint_ids: tuple[int, ...]
    qpos_adr: tuple[int, ...]
    dof_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]

    @property
    def n_joints(self) -> int:
        return len(self.joint_ids)


def load_model(xml_path: Path | str) -> mujoco.MjModel:
    """从 XML 加载模型, 失败时给出可读的报错。"""
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到模型文件: {path}\n"
            "请确认 assets/robo1.xml 与 assets/meshes/*.stl 都在工程里。"
        )
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001 - 转成对用户友好的信息
        raise RuntimeError(
            f"MuJoCo 加载模型失败: {path}\n原始错误: {exc}\n"
            "常见原因: meshes 目录缺失、meshdir 写错、XML 语法错误。"
        ) from exc


def _name2id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise ValueError(
            f"模型里找不到名字为 {name!r} 的 {'joint' if objtype == mujoco.mjtObj.mjOBJ_JOINT else '对象'}。"
            f"请检查 assets/robo1.xml。"
        )
    return int(idx)


def build_index(model: mujoco.MjModel) -> ModelIndex:
    base_body = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)

    joint_ids = tuple(_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES)
    actuator_ids = tuple(_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES)

    # 基座的自由关节: 每个 body 的第一个关节
    jnt_adr = int(model.body_jntadr[base_body])
    if jnt_adr < 0 or model.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            f"body {BASE_BODY_NAME!r} 应该挂一个 free joint(整个机器人会倒), "
            "但模型里不是。请检查 robo1.xml 的 worldbody。"
        )

    index = ModelIndex(
        base_body=base_body,
        free_joint=jnt_adr,
        free_qpos_adr=int(model.jnt_qposadr[jnt_adr]),
        joint_ids=joint_ids,
        qpos_adr=tuple(int(model.jnt_qposadr[j]) for j in joint_ids),
        dof_adr=tuple(int(model.jnt_dofadr[j]) for j in joint_ids),
        actuator_ids=actuator_ids,
    )

    # 断言: 这套环境假设 2 自由度 + 自由基座
    expected_nq = 7 + index.n_joints
    expected_nv = 6 + index.n_joints
    if model.nq != expected_nq or model.nv != expected_nv:
        raise ValueError(
            f"模型维度不符合预期: nq={model.nq}(期望 {expected_nq}), "
            f"nv={model.nv}(期望 {expected_nv})。本工程针对 2 自由度不倒翁。"
        )
    if model.nu != index.n_joints:
        raise ValueError(f"执行器数量 {model.nu} 与关节数 {index.n_joints} 不一致。")

    return index


def body_upright(data: mujoco.MjData, body_id: int) -> float:
    """基座 body 的 z 轴与世界 z 轴的点积, 1.0 表示完全竖直, 0 表示侧躺。"""
    rot = data.xmat[body_id].reshape(3, 3)  # body -> world
    return float(rot[2, 2])


def body_tilt_deg(data: mujoco.MjData, body_id: int) -> float:
    """基座相对竖直方向的倾角(度)。"""
    return float(np.degrees(np.arccos(np.clip(body_upright(data, body_id), -1.0, 1.0))))


def gravity_in_body_frame(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """重力方向在 body 坐标系下的分量, 站立时约为 (0, 0, -1)。

    这是机器人 RL 里最常用的姿态观测: 比四元数更平滑、无万向节问题,
    而且和真机上 IMU 能直接测到的量对应。
    """
    rot = data.xmat[body_id].reshape(3, 3)
    return rot.T @ np.array([0.0, 0.0, -1.0])
