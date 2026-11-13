#!/usr/bin/env python
"""校验 firmware/include/policy.h 与训练用的 SB3 模型数值一致。

为什么需要:
    C 代码里权重是按行优先展平的二维数组, 靠下标算术寻址:
        s += W[i * in_dim + j] * in[j]
    如果导出时的展平顺序和这里的读法不一致(行列颠倒), 编译、烧录、串口都不会报错,
    但板子上算出来的动作是错的 —— 只会在真机上表现为"动了, 但不对"。

三段对照, 一次定位问题出在哪:
    [A] 头文件解析 → numpy 前向 (完全复现 policy_runtime.h 的下标算术)
    [B] 直接用训练模型的 torch 子模块算
    [C] model.predict() 的输出
    A vs B 不一致 => 头文件导出/展平顺序有问题
    B vs C 不一致 => predict 的裁剪差异(通常无害)

用法:
    python tools/check_policy_export.py runs/xxx/final_model.zip firmware/include/policy.h
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- 头文件解析
def parse_header(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")

    def define(name: str) -> int:
        m = re.search(rf"#define\s+{name}\s+(\d+)", txt)
        if not m:
            raise SystemExit(f"头文件里找不到 #define {name}")
        return int(m.group(1))

    def nums(name: str) -> str:
        m = re.search(rf"\b{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}\s*;", txt, re.S)
        if not m:
            raise SystemExit(f"头文件里找不到数组 {name}")
        return m.group(1)

    def floats(name: str) -> np.ndarray:
        return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", nums(name))])

    def ints(name: str) -> np.ndarray:
        return np.array([int(x) for x in re.findall(r"-?\d+", nums(name))])

    act = "elu"
    for cand in ("elu", "relu", "tanh"):
        if re.search(rf"#define\s+POLICY_ACT_{cand.upper()}\s+1", txt):
            act = cand

    return {
        "layers": [define("POLICY_IN")] + [int(v) for v in ints("POLICY_HIDDEN")] + [define("POLICY_OUT")],
        "act": act,
        "W": floats("POLICY_W"),
        "B": floats("POLICY_B"),
        "w_off": ints("POLICY_W_OFFSET"),
        "b_off": ints("POLICY_B_OFFSET"),
    }


def act_fn(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "tanh":
        return np.tanh(x)
    return np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1.0)   # ELU(alpha=1)


def forward_numpy(h: dict, obs: np.ndarray, col_major: bool = False) -> np.ndarray:
    """按 policy_runtime.h 的下标算术复现前向; col_major=True 时换成列优先读法。"""
    x = np.asarray(obs, dtype=np.float64)
    W, B, wo, bo = h["W"], h["B"], h["w_off"], h["b_off"]
    dims = h["layers"]
    for L in range(len(dims) - 1):
        in_dim, out_dim = dims[L], dims[L + 1]
        w0, b0 = int(wo[L]), int(bo[L])
        out = np.zeros(out_dim)
        for i in range(out_dim):
            s = B[b0 + i]
            for j in range(in_dim):
                s += (W[w0 + j * out_dim + i] if col_major else W[w0 + i * in_dim + j]) * x[j]
            out[i] = s
        x = act_fn(out, h["act"]) if L < len(dims) - 2 else out
    return x


# ----------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="校验 policy.h 与 SB3 模型一致")
    ap.add_argument("model", help="SB3 模型路径 (.zip)")
    ap.add_argument("header", nargs="?", default="firmware/include/policy.h")
    ap.add_argument("-n", "--samples", type=int, default=100)
    args = ap.parse_args()

    h = parse_header(Path(args.header))
    dims = h["layers"]
    print(f"头文件   : {args.header}")
    print(f"结构     : {' -> '.join(map(str, dims))}   激活: {h['act']}")
    expect = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    expect_b = sum(dims[1:])
    print(f"权重 {len(h['W'])} 个 / 偏置 {len(h['B'])} 个（应为 {expect} / {expect_b}）")
    if len(h["W"]) != expect or len(h["B"]) != expect_b:
        print("[FAIL] 权重个数和网络结构对不上, 说明导出时层顺序或维度错了。")
        return 1

    import torch
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    pol = model.policy

    def torch_forward(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            feat = pol.mlp_extractor.policy_net(t)
            return pol.action_net(feat).squeeze(0).numpy().astype(np.float64)

    obs_dim = dims[0]
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(args.samples, obs_dim)).astype(np.float32)
    obs[:, 0:3] /= np.linalg.norm(obs[:, 0:3], axis=1, keepdims=True)
    obs[0] = 0.0
    obs[1] = [0.0, 0.0, -1.0] + [0.0] * (obs_dim - 3)
    obs[2] = [1.0, 0.0, 0.0] + [0.0] * (obs_dim - 3)

    # predict 会把动作裁剪到动作空间的上下界; 所以对比裁剪后的值才有意义
    lo = float(np.min(model.action_space.low))
    hi = float(np.max(model.action_space.high))

    e_ab, e_bc, raw_max = [], [], 0.0
    for o in obs:
        a_np = forward_numpy(h, o)
        a_th = torch_forward(o)
        a_py = np.asarray(model.predict(o, deterministic=True)[0], dtype=np.float64)
        e_ab.append(np.max(np.abs(a_np - a_th)))
        e_bc.append(np.max(np.abs(np.clip(a_th, lo, hi) - a_py)))
        raw_max = max(raw_max, float(np.max(np.abs(a_th))))
    e_ab, e_bc = np.array(e_ab), np.array(e_bc)

    print(f"\n样本数   : {len(obs)}   动作空间: [{lo:g}, {hi:g}]")
    print(f"[A] 头文件→numpy vs [B] torch        : 最大误差 {e_ab.max():.3e}   ← 判定导出是否正确")
    print(f"[B] 裁剪到动作空间后 vs [C] predict  : 最大误差 {e_bc.max():.3e}   ← 应接近 0")
    if raw_max > hi + 1e-3:
        print(f"    (说明: 随机观测在网络训练分布之外, 原始输出最大到 {raw_max:.2f}, 会被 predict 裁到 {hi:g})")

    if e_ab.max() > 1e-4:
        e_alt = max(np.max(np.abs(forward_numpy(h, o, col_major=True) - torch_forward(o))) for o in obs[:20])
        print(f"\n[诊断] 列优先读法的误差: {e_alt:.3e}")
        if e_alt < 1e-4:
            print("       => 导出时按列优先展平, 但 policy_runtime.h 按行优先读。")
            print("          修法: export_policy.py 里 w.reshape(-1) 改成 w.T.reshape(-1)")
        else:
            print("       => 不是展平顺序问题。重点查: 头文件数值是否被截断精度,")
            print("          以及前向里最后一层是否漏/多了激活。")
        return 1

    print("\n[OK] 头文件的前向计算与训练模型一致，可以编译烧录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
