# SelfRising RL — 2 自由度不倒翁的强化学习实验工程

把 StackForce 的「开源强化学习不倒翁」模型，改造成一个**可以直接训练、可以做消融、
可以逐步加深难度**的 RL 学习环境。模型是真的，物理是真的，训练代码是重新写的。

- 机器人：2 自由度（两只 MG90S 舵机）+ 自由基座，会倒、能自己站起来
- 仿真：MuJoCo，物理步长 1 ms，控制频率 50 Hz
- 算法：Stable-Baselines3 的 PPO（CPU 可训，不需要 GPU）
- 配套文档：[`docs/学习手册.docx`](docs/学习手册.docx)（43 页初学者版）· 在线读 [`docs/学习手册.md`](docs/学习手册.md)（同一份内容的 Markdown 排版版）

## 快速开始（Linux）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python -m scripts.selftest                       # 1. 自检，必须全绿
python -m scripts.run_scripted                   # 2. 看脚本基线成功率
python -m scripts.train_ppo --config configs/ppo_fast.json   # 3. 训练
tensorboard --logdir runs/ppo_fast/tb --bind_all            # 4. 看曲线
python -m scripts.evaluate --model runs/ppo_fast/best_model --episodes 10  # 5. 评测
python -m scripts.play --model runs/ppo_fast/best_model     # 6. 可视化
```

> 无图形界面的虚拟机：用 `python -m scripts.play --record out.gif` 导出 GIF 再看。

## 目录结构

```
selfrising-rl/
├── assets/                 # 机器人模型（来自 A 工程）
│   ├── robo1.xml           #   MJCF：2 关节 + 自由基座 + 2 个位置执行器
│   └── meshes/*.stl        #   6 个零件网格
├── reference/              # 参考数据（来自 A 工程）
│   ├── reference-learning.json   # 四个姿态的关键帧参考动作
│   └── author_policy.onnx        # 原作者策略（仅对照，不参与训练）
├── selfrising/
│   ├── env.py              # Gymnasium 环境（核心）
│   ├── reward.py           # 奖励项
│   ├── poses.py            # 四个跌倒姿态 + 扰动
│   ├── scripted.py         # 参考动作回放（对照组）
│   ├── model.py            # 模型加载与索引
│   ├── rollout.py          # 回合执行
│   ├── evaluate.py         # 按姿态评测
│   └── callbacks.py        # SB3 训练期评测回调
├── scripts/                # 命令行入口
├── configs/                # 三套训练配置
├── tests/test_logic.py     # 纯逻辑单测（不需要 MuJoCo）
├── firmware/               # ESP32 固件（PlatformIO，第 7.5 节）
│   ├── platformio.ini
│   ├── src/main.cpp        #   50 Hz 循环 + 观测拼装 + 舵机输出
│   ├── src/policy_runtime.h#   纯 C 前向推理
│   ├── include/policy.h    #   权重（占位版，用导出脚本覆盖）
│   └── tools/export_policy.py  # SB3 .zip -> policy.h
└── docs/                   # 学习手册（.docx 原版 + .md 排版版）
```

## 环境接口

| 项目 | 定义 |
|---|---|
| 观测（10 维） | 重力在机体系分量(3) + 两关节角(2) + 两关节角速度(2) + 上一目标角(2) + 基座高度(1) |
| 动作（2 维） | [-1, 1]，表示关节目标角的增量；`target += 0.08 * a`，限幅 ±1.55 rad |
| 控制频率 | 50 Hz（1 ms × 20 次物理步进） |
| 初始状态 | roll ±90° / pitch ±90° 四个姿态之一，附加 ±15° 扰动 |
| 成功判定 | 倾角 ≤ 14.32° + 关节回到 0±0.12 rad + 基座速度 ≤ 0.25 m/s，并保持 1 s；单回合上限 14 s |
| 失败判定 | 连续 0.5 s 不接触地面（被弹飞/穿模）、数值发散 |

### 奖励项（`selfrising/reward.py`）

| 项 | 权重 | 作用 |
|---|---|---|
| `upright` | +1.0 | 基座 z 轴与世界 z 轴的点积，鼓励站直 |
| `joint_goal` | +0.5 | 站直后把两关节收回 0（成功姿态）；用 upright² 门控，只在已经站直时给 |
| `progress` | +0.2 | 比上一步更直才给分 |
| `action_rate` | -0.02 | 抑制动作抖动（上真机最关键） |
| `joint_vel` | -0.001 | 抑制甩动 |
| `success` / `fail` | +10 / -5 | 终端奖励 |
| ~~`height`~~ | 0（已关闭） | 原 -0.5。它取的是 `qpos[2]`（foot 坐标系原点，在几何体之外），站直时反而变负，实测会压制起身，故停用 |

## 三套配置

| 配置 | 用途 | 关键差异 |
|---|---|---|
| `ppo_fast.json` | 先跑通 | 30 万步，8 环境，无扰动 |
| `ppo_full.json` | 认真训 | 200 万步，16 环境，128×128 |
| `ppo_robust.json` | 练 sim2real | 加动作延迟、观测噪声、质量/摩擦/增益随机化 |

## 和原工程 A 的差异（重要）

1. **动作空间**：A 的策略是 10 进 10 出，其中 8 路是"控制器状态回灌"（last target、
   get-up active、tick 计数……），属于手写状态机；本工程改成干净的 **2 维增量动作**。
2. **观测**：A 用 roll/pitch 两个欧拉角；本工程用**重力向量**（3 维），无万向节问题，
   且与真机 IMU 可测量直接对应。
3. **模型**：本工程用 A 包内 `Training/` 的 `robo1.xml`（2 个 actuator），而不是
   `Sim-to-Sim` 里那个 10 个 actuator 的 `model.xml`。
4. **碰撞**：原模型所有 `*_collision` 盒子是 `contype="0" conaffinity="0"`，
   实际碰撞由 STL 凸包承担。本工程保留原样；正式实验前建议按手册 6.3 节改成盒子碰撞。
5. **训练**：A 的实际训练引擎是云端的 `native_mujoco_bc_v1`（行为克隆），
   本工程用 SB3 的 PPO 本地训练，可完全离线复现。
6. **路径**：`meshdir` 从 `assets` 改为 `meshes`（目录整理），其余 XML 内容未改。

## 出处与许可

本仓库**分两部分授权**，请分别看待：

| 范围 | 路径 | 许可 |
|---|---|---|
| 我写的代码与文档 | `selfrising/`、`scripts/`、`tests/`、`firmware/`、`docs/` | MIT |
| 取自上游的模型与数据 | `assets/`、`reference/` | **不随本仓库重新授权** |

- 机器人模型与策略权重来自 HomeMadeGarbage / SelfRisingRobot（"poco"），上游未声明统一许可证。
- 参考动作与 `.onnx` 来自 StackForce `RL_SelfRisingRobot_StackForce` 发布物。
- 上述文件**仅用于学习研究**，本仓库不声称拥有其著作权、也不对其授予 MIT/GPL，
  商用或再发布前请自行联系权利人。

完整说明见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) 与 `docs/学习手册.docx` 附录 D。

## 公开托管的说明

本仓库是公开的，任何人可以克隆、下载、学习。如果你是这个机器人的权利人，
认为某些文件不应被公开托管，请开 Issue 说明文件，我会立即删除对应内容。


## 推送到 GitHub（本机说明）

本机的部署密钥放在 `.deploy/ssh/`（已被 .gitignore 忽略），不在 `~/.ssh` 里，
所以直接 `git push` 会失败。用自带的脚本推：

```cmd
push.cmd
```

它就是设好 `GIT_SSH` 再执行 `git push origin main`，换成你自己的环境时可以删掉。
