# 固件：把策略跑在 ESP32 上（VSCode + PlatformIO）

这是 `docs/学习手册.docx` 第 7.5 节的可运行版本。目标很具体：
**把 Python 里训出来的策略，变成板子上每 20 ms 跑一次的前向计算。**

> ⚠ 这是一份**教学骨架**，不是开箱即用的产品固件。它由四个量构成闭环：
> 50 Hz、0.08 rad/步、±1.55 限幅、10 维观测。任何一个与训练时不一致，
> 机器人就会立刻失效——这恰好是 sim2real 要你亲手体会的部分。

## 目录

```
firmware/
├── platformio.ini          # 板型、框架、依赖
├── include/policy.h        # 权重（已带 ppo_robust_ft2 的真实权重；重训后覆盖）
├── src/
│   ├── main.cpp            # 50 Hz 主循环、观测拼装、舵机输出
│   └── policy_runtime.h    # 纯 C 前向推理（不依赖任何推理框架）
└── tools/export_policy.py  # SB3 .zip -> policy.h
```

## 接线

| 器件 | ESP32 引脚 | 说明 |
|---|---|---|
| 下舵机（A0） | GPIO25 | 信号线，电源单独 5 V、共地 |
| 上舵机（A1） | GPIO26 | 同上 |
| BNO055 SDA | GPIO21（默认 I2C） | 模块地址 0x29 |
| BNO055 SCL | GPIO22 | 同上 |
| BOOT 键 | GPIO0 | 按住 = 回中位并暂停 |

舵机**不要**从板子的 3.3 V 取电，MG90S 堵转电流会把板子拉重启。

## 三步用法

**第 1 步：导出权重**

```bash
cd selfrising-rl/firmware
python tools/export_policy.py ../runs/ppo_fast/best_model.zip -o include/policy.h
```

成功时会打印网络形状、激活函数与 `max|w|`。如果提示“带 VecNormalize”，
说明训练脚本里加了观测归一化，需要先把归一化去掉再训（本工程默认不加）。

**第 2 步：先不动舵机，只看观测**

`src/main.cpp` 顶部把 `DRY_RUN` 保持为 `1`，编译烧录：

```bash
pio run -t upload
pio device monitor
```

串口每 5 Hz 打印一次。检查两件事：

1. 机器人**站直**时 `g=(0.00,0.00,-1.00)` 附近；
2. 向左倒 `gy` 变负、向右倒 `gy` 变正（符号不对就改 `GRAVITY_FLIP`）。

这两条不对，后面全白搭——因为观测的坐标系与训练时不同。

**第 3 步：改成 `DRY_RUN 0`，标定后再跑**

先把 `SERVO_DIR_LOW/HIGH` 与 `SERVO_TRIM_*_US` 调到两舵机同向、中位对称，
再上电。第一次建议把 `ACTION_DELTA` 临时改成 0.04 跑一遍，确认方向对了再恢复。

## 与仿真的对齐表

| 量 | 仿真（env.py） | 固件（main.cpp） |
|---|---|---|
| 控制周期 | `control_dt = 0.02` | `PERIOD_US = 20000` |
| 动作增量 | `action_delta = 0.08` | `ACTION_DELTA = 0.08` |
| 动作限幅 | `action_limit = 1.55` | `ACTION_LIMIT = 1.55` |
| 观测 | `_obs()` 的 10 个量 | `read_observation()` |
| 输出裁剪 | `clip(a, -1, 1)` | `policy_forward()` 内部 clamp |

## 三个必须知道的差距

这份骨架在“物理上”和仿真并不完全一样，这不是疏忽，是真实条件的限制：

1. **没有关节编码器**。MG90S 不带反馈，固件用“上次下发的目标角”当关节角，
   用目标角差分当角速度。仿真里用的是真实角度。这个差距只能靠“舵机跟得上”
   来近似，舵机软、负载重时会失真。
2. **没有高度传感器**。观测第 9 维直接用常量 0.08。
3. **IMU 有噪声与零漂**。仿真里默认没有噪声；想更接近实机，用
   `configs/ppo_robust.json` 训练，它会打开观测噪声与延迟。

这三条正是手册 7.3 节那张“对齐清单”存在的理由：先知道差在哪，再决定补不补。

## 排错

| 现象 | 先查 |
|---|---|
| 编译报 `POLICY_IN` 未定义 | `include/policy.h` 丢了：重新导出，或 `git checkout -- include/policy.h` |
| 串口全是 `g=(0,0,-1)` | BNO055 没认到：查 I2C 接线与 0x29 地址 |
| 舵机一直不动 | `DRY_RUN` 还是 1 |
| 舵机往一个方向顶死 | `SERVO_DIR_*` 与行程没标定 |
| 动作一上电就疯 | 观测符号错了，回第 2 步查 `g` |
| 串口刷屏 `policy shape mismatch` | 网络层数与 `POLICY_N_HIDDEN` 不一致，重新导出 |
