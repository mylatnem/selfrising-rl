"""SelfRising RL 学习工程.

一个基于 MuJoCo 的 2 自由度"不倒翁"强化学习实验环境, 模型与参考动作取自
StackForce 的 RL_SelfRisingRobot_StackForce 工程(上游为 HomeMadeGarbage 的
SelfRisingRobot "poco")。本工程只保留可复用的模型与参考数据, 训练/评估/
可视化代码全部重写, 用于教学与实验。

注意: 导入本包不会加载 MuJoCo, 便于在无仿真环境时做纯逻辑测试。
"""

from .paths import ASSETS_DIR, MODEL_XML, PROJECT_ROOT, REFERENCE_JSON

__all__ = ["PROJECT_ROOT", "ASSETS_DIR", "MODEL_XML", "REFERENCE_JSON"]
__version__ = "0.1.0"
