#!/usr/bin/env python
"""把一个训练好的 SB3 PPO 模型导出成 C 头文件（policy.h）。

    python tools/export_policy.py runs/ppo_fast/best_model.zip -o include/policy.h

它只做一件事：把 mlp_extractor.policy_net + action_net 的权重按行展开成
一维数组，同时写出形状与每层偏移，供 src/policy_runtime.h 做前向。

注意：导出的观测顺序必须与元工程的 selfrising/env.py 里 _obs() 完全一致，
否则在板子上算出来的动作是“合法的垃圾”。
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import torch


def _fmt(values, per_line: int = 8) -> str:
    """把一维浮点数组格式化成 C 初始化列表（保证字面量带小数点）。"""
    chunks = []
    for i in range(0, len(values), per_line):
        parts = []
        for v in values[i : i + per_line]:
            s = f"{v:.8g}"
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            parts.append(s + "f")
        chunks.append("    " + ", ".join(parts) + ",")
    return "\n".join(chunks)


def _activation_name(module) -> str:
    if isinstance(module, torch.nn.ELU):
        return "elu"
    if isinstance(module, torch.nn.ReLU):
        return "relu"
    return "tanh"


def extract(model_path: str) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")

    # 如果训练时用了 VecNormalize，导出的权重就不再是“原始观测”的函数
    try:
        vecnorm = model.get_vec_normalize_env()
    except Exception:  # noqa: BLE001
        vecnorm = None
    if vecnorm is not None and getattr(vecnorm, "obs_rms", None) is not None:
        raise SystemExit(
            "这个模型带 VecNormalize（观测归一化）。请先在训练脚本里去掉，"
            "或把 mean/std 一起导出并在固件里做归一化，否则板子上会算错。"
        )

    policy = model.policy
    mlp = getattr(policy, "mlp_extractor", None)
    if mlp is None:
        raise SystemExit("这个策略不是 MLP（可能是 CNN/RNN），本脚本不支持。")

    seq = mlp.policy_net
    linears = [m for m in seq if isinstance(m, torch.nn.Linear)]
    acts = [m for m in seq if isinstance(m, (torch.nn.Tanh, torch.nn.ELU, torch.nn.ReLU))]
    if not linears:
        raise SystemExit("在 policy_net 里没找到 Linear 层，模型结构可能被改过。")
    linears.append(policy.action_net)  # 输出层

    act_name = _activation_name(acts[0]) if acts else "tanh"
    obs_dim = int(linears[0].in_features)
    out_dim = int(linears[-1].out_features)
    hidden = [int(l.out_features) for l in linears[:-1]]

    w_all: list[float] = []
    b_all: list[float] = []
    w_off: list[int] = []
    b_off: list[int] = []

    for layer in linears:
        w_off.append(len(w_all))
        b_off.append(len(b_all))
        w = layer.weight.detach().cpu().numpy()  # (out, in)
        b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
        if b is None:
            b = torch.zeros(w.shape[0]).numpy()
        w_all.extend(w.reshape(-1).tolist())
        b_all.extend(b.tolist())

    return {
        "act": act_name,
        "obs_dim": obs_dim,
        "out_dim": out_dim,
        "hidden": hidden or [4],
        "w": w_all,
        "b": b_all,
        "w_off": w_off,
        "b_off": b_off,
        "params": sum(len(l.weight.reshape(-1)) for l in linears) + len(b_all),
    }


def render(meta: dict, model_path: str) -> str:
    hidden = meta["hidden"]
    act_def = f"#define POLICY_ACT_{meta['act'].upper()} 1"
    lines = [
        "// ============================================================",
        "//  policy.h —— 由 tools/export_policy.py 自动生成，请勿手改",
        f"//  模型：{model_path}",
        f"//  生成时间：{_dt.datetime.now().isoformat(timespec='seconds')}",
        f"//  结构：{meta['obs_dim']} -> {' -> '.join(map(str, hidden))} -> {meta['out_dim']}"
        f"（激活 {meta['act']}，参数 {meta['params']} 个）",
        "// ============================================================",
        "#pragma once",
        "",
        f"#define POLICY_IN {meta['obs_dim']}",
        f"#define POLICY_OUT {meta['out_dim']}",
        f"#define POLICY_N_HIDDEN {len(hidden)}",
        act_def,
        "",
        f"static const int POLICY_HIDDEN[POLICY_N_HIDDEN] = {{{', '.join(map(str, hidden))}}};",
        "",
        "// 每层权重按行展开：第 i 行 = 第 i 个神经元的输入权重",
        "static const float POLICY_W[] = {",
        _fmt(meta["w"]),
        "};",
        "",
        "static const float POLICY_B[] = {",
        _fmt(meta["b"]),
        "};",
        "",
        "// 每层权重在 POLICY_W / POLICY_B 里的起始下标",
        f"static const int POLICY_W_OFFSET[POLICY_N_HIDDEN + 1] = {{{', '.join(map(str, meta['w_off']))}}};",
        f"static const int POLICY_B_OFFSET[POLICY_N_HIDDEN + 1] = {{{', '.join(map(str, meta['b_off']))}}};",
        "",
        "// 一致性检查：固件里的观测顺序与训练时不同，编译期无法发现，只能靠自检",
        "#if POLICY_IN != 10",
        '#error "观测维度不是 10：请确认你导出的是这个工程的策略"',
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="SB3 PPO -> policy.h")
    ap.add_argument("model", help="runs/xxx/best_model.zip 之类的路径")
    ap.add_argument("-o", "--out", default="include/policy.h")
    args = ap.parse_args()

    model_path = args.model[:-4] if args.model.endswith(".zip") else args.model
    if not Path(model_path + ".zip").exists() and not Path(model_path).exists():
        raise SystemExit(f"找不到模型：{model_path}(.zip)")

    meta = extract(model_path)
    text = render(meta, model_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig = 带 BOM。不加 BOM 时, Windows 上的编辑器可能按 GBK 解码,
    # 中文注释会变成乱码(不影响编译, 但没法读)。
    out.write_text(text, encoding="utf-8-sig")

    wmax = max(abs(v) for v in meta["w"]) if meta["w"] else 0.0
    print(f"已写入 {out}")
    print(f"  结构 : {meta['obs_dim']} -> {' -> '.join(map(str, meta['hidden']))} -> {meta['out_dim']}")
    print(f"  激活 : {meta['act']}")
    print(f"  参数 : {meta['params']} 个（约 {out.stat().st_size / 1024:.0f} KB 头文件）")
    print(f"  权重 : max|w| = {wmax:.4f}")
    if wmax > 50:
        print("  [WARN] 权重数值偏大，请确认没有漏掉归一化")
    print("下一步：pio run -t upload，并把 DRY_RUN 设为 1 先看串口输出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
