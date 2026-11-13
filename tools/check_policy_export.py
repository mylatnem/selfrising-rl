#!/usr/bin/env python
"""校验 firmware/include/policy.h 与训练用的 SB3 模型数值一致。

为什么需要:
    C 代码里权重是"行优先展平"的二维数组, 靠下标算术寻址:
        s += W[i * in_dim + j] * in[j]
    如果导出时的展平顺序和这里的读法不一致(行列颠倒), 编译、烧录、串口都不会报错,
    但板子上算出来的动作是错的 —— 只会在真机上表现为"动了, 但不对"。

用法:
    python tools/check_policy_export.py runs/ppo_robust_ft2/final_model.zip firmware/include/policy.h
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def parse_header(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")

    def define(name: str) -> int:
        m = re.search(rf"#define\s+{name}\s+(\d+)", txt)
        if not m:
            raise SystemExit(f"头文件里找不到 #define {name}")
        return int(m.group(1))

    def floats(name: str) -> np.ndarray:
        m = re.search(rf"\b{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}\s*;", txt, re.S)
        if not m:
            raise SystemExit(f"头文件里找不到数组 {name}")
        body = m.group(1)
        return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)])

    def ints(name: str) -> np.ndarray:
        m = re.search(rf"\b{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}\s*;", txt, re.S)
        if not m:
            raise SystemExit(f"头文件里找不到数组 {name}")
        body = m.group(1)
        return np.array([int(x) for x in re.findall(r"-?\d+", body)])

    act = "elu"
    for cand in ("elu", "relu", "tanh"):
        if re.search(rf"#define\s+POLICY_ACT_{cand.upper()}\s+1", txt):
            act = cand

    n_hidden = define("POLICY_N_HIDDEN")
    layers = [define("POLICY_IN")] + [int(v) for v in ints("POLICY_HIDDEN")] + [define("POLICY_OUT")]
    return {
        "layers": layers,
        "act": act,
        "W": floats("POLICY_W"),
        "B": floats("POLICY_B"),
        "w_off": ints("POLICY_W_OFFSET"),
        "b_off": ints("POLICY_B_OFFSET"),
        "n_hidden": n_hidden,
    }


def act_fn(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "tanh":
        return np.tanh(x)
    # ELU: x>0 -> x, x<=0 -> exp(x)-1
    return np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


def forward(h: dict, obs: np.ndarray) -> np.ndarray:
    """完全按 policy_runtime.h 的下标算术复现前向计算。"""
    x = np.asarray(obs, dtype=np.float64)
    W, B, wo, bo = h["W"], h["B"], h["w_off"], h["b_off"]
    dims = h["layers"]
    for L in range(len(dims) - 1):
        in_dim, out_dim = dims[L], dims[L + 1]
        w0, b0 = wo[L], bo[L]
        out = np.zeros(out_dim)
        for i in range(out_dim):
            s = B[b0 + i] if b0 + i < len(B) else 0.0
            for j in range(in_dim):
                s += W[w0 + i * in_dim + j] * x[j]
            out[i] = s
        x = act_fn(out, h["act"]) if L < len(dims) - 2 else out
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 policy.h 与 SB3 模型一致")
    ap.add_argument("model", help="SB3 模型路径 (.zip)")
    ap.add_argument("header", nargs="?", default="firmware/include/policy.h")
    ap.add_argument("-n", "--samples", type=int, default=200)
    args = ap.parse_args()

    h = parse_header(Path(args.header))
    print(f"头文件   : {args.header}")
    print(f"结构     : {' -> '.join(map(str, h['layers']))}  激活: {h['act']}")
    print(f"权重个数 : {len(h['W'])}  偏置个数: {len(h['B'])}")

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    obs_dim = h["layers"][0]

    rng = np.random.default_rng(0)
    # 用观测空间的边界生成合法样本, 再混入几个极端值
    lo, hi = -1.0, 1.0
    obs = rng.uniform(lo, hi, size=(args.samples, obs_dim)).astype(np.float32)
    obs[:, 0:3] /= np.linalg.norm(obs[:, 0:3], axis=1, keepdims=True)  # 重力向量是单位向量
    obs[0] = 0.0
    obs[1] = [0.0, 0.0, -1.0] + [0.0] * (obs_dim - 3)   # 站直姿态
    obs[2] = [1.0, 0.0, 0.0] + [0.0] * (obs_dim - 3)    # 侧倒姿态

    errs = []
    for o in obs:
        a_py = np.asarray(model.predict(o, deterministic=True)[0], dtype=np.float64)
        a_c = forward(h, o)
        errs.append(np.max(np.abs(a_py - a_c)))
    errs = np.array(errs)

    print(f"样本数   : {len(obs)}")
    print(f"最大误差 : {errs.max():.3e}")
    print(f"平均误差 : {errs.mean():.3e}")

    if errs.max() > 1e-4:
        print("\n[FAIL] C 代码与 Python 算出来的动作不一致。")
        print("       最常见原因: 权重展平顺序(行/列优先)与 policy_runtime.h 的读法不匹配。")
        return 1
    print("\n[OK] 固件里的前向计算与训练模型一致，可以编译烧录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
