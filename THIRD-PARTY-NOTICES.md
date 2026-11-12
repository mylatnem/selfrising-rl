# 第三方来源与许可 / Third-party notices

本仓库分为两部分：**我自己写的代码与文档**，以及**从别人项目里取的模型资源**。
两者的授权范围完全不同，请分别看待。

## 1. 本仓库新增的部分（可自由使用）

| 路径 | 内容 | 许可 |
|---|---|---|
| `selfrising/` | Gymnasium 环境、奖励、姿态、参考动作回放、评测 | MIT |
| `scripts/` | 自检、训练、评估、可视化入口 | MIT |
| `tests/` | 纯逻辑单元测试 | MIT |
| `firmware/` | ESP32 固件骨架、C 前向推理、权重导出脚本 | MIT |
| `configs/` | 三套训练超参 | MIT |
| `docs/学习手册.docx` | 学习手册 | MIT |

## 2. 取自上游的模型与数据（**不随本仓库重新授权**）

| 路径 | 内容 | 来源 |
|---|---|---|
| `assets/robo1.xml` | 2 自由度机器人的 MJCF 模型 | HomeMadeGarbage / SelfRisingRobot（"poco"） |
| `assets/meshes/*.stl` | 6 个零件网格 | 同上 |
| `reference/author_policy.onnx` | 原作者神经网络策略（10 进 10 出） | HomeMadeGarbage，经 StackForce 发布包适配为 ONNX |
| `reference/reference-learning.json` | 四个跌倒姿态的关键帧参考动作 | StackForce《开源强化学习不倒翁》发布包 |
| `reference/training-result-evaluation.json` | 上游训练结果的评估记录 | 同上 |

**上游在发布时没有声明可据以重新许可这些资源的统一许可证。** 因此：

- 这些文件在本仓库中**仅用于学习与研究**；
- 本仓库**不声称**对上述模型、网格、权重拥有著作权；
- 本仓库**不就上述文件授予**任何 MIT / GPL / CC 许可；
- 若要商用或再发布，请自行联系上游权利人取得授权。

## 3. 侵权处理

如果你是权利人，或你代表权利人，认为本仓库中的某些文件不应被公开托管，
请在本仓库开一个 Issue 说明具体文件，我会在确认后**立即删除相关内容**。

## 4. 上游链接

- HomeMadeGarbage / SelfRisingRobot：https://github.com/homemadegarbage/SelfRisingRobot
- StackForce《开源强化学习不倒翁》：https://github.com/Deng-ge-open-source/RL_SelfRisingRobot_StackForce
- StackForce 仿真平台：https://sim2real.stackforce.cc/
