#!/usr/bin/env bash
# 压力阶梯评测: 同一个策略, 在"越来越像真机"的环境里表现如何。
#
# 用法:
#   bash tools/eval_ladder.sh                    # 默认评测 runs/ppo_fast/final_model
#   bash tools/eval_ladder.sh runs/xxx/final_model
#
# 为什么要有这个脚本: 在无桌面的虚拟机上, 长命令只能手敲(没有剪贴板),
# 所以把整条链路固化成脚本, git pull 之后一条命令跑完。

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-runs/ppo_fast/final_model}"
OUT="eval_ladder.txt"

if [ ! -f "${MODEL}.zip" ] && [ ! -f "${MODEL}" ]; then
    echo "[x] 找不到模型: ${MODEL}.zip"
    echo "    现有模型:"
    ls -1 runs/*/final_model.zip 2>/dev/null | sed 's/^/      /' || echo "      (没有)"
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[x] 没找到 .venv, 请先: python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

: > "$OUT"
run() {
    local name="$1" env="$2"
    echo "### ${name}    env=${env}" | tee -a "$OUT"
    python -m scripts.evaluate --model "$MODEL" --episodes 10 --env-json "$env" 2>&1 \
        | tail -n 8 | tee -a "$OUT"
    echo | tee -a "$OUT"
}

echo "模型: ${MODEL}"
echo

run "1 基线 (仅 +-15 度扰动)"                    '{}'
run "2 + 1 步动作延迟"                           '{"action_delay": 1}'
run "3 + 观测噪声"                               '{"obs_noise": 1.0}'
run "4 + 域随机化 (质量/摩擦/增益)"               '{"domain_randomize": true}'
run "5 全叠加 (最接近真机)"                       '{"action_delay": 1, "obs_noise": 1.0, "domain_randomize": true}'

echo "完整结果: ${OUT}"
echo "看全文:   less ${OUT}    (空格下翻, b 上翻, q 退出)"
