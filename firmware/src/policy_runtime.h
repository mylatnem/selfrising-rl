// ============================================================
//  策略前向推理：纯 C 实现，不依赖任何推理框架
//
//  网络形状：10 -> (若干隐藏层) -> 2，权重来自 policy.h
//  与 Python 侧对齐的关键点：
//    1. 观测顺序必须与 selfrising/env.py 的 _obs() 完全一致
//    2. 输入已经是 float32，且已按 observation_space 裁剪
//    3. 输出做 clamp 到 [-1, 1]，与 env.step() 一致
// ============================================================
#pragma once

#include <math.h>

#include "policy.h"

#define POLICY_MAX_WIDTH 256

static inline float policy_act(float x) {
#if defined(POLICY_ACT_ELU)
    return x > 0.0f ? x : (expf(x) - 1.0f);
#elif defined(POLICY_ACT_RELU)
    return x > 0.0f ? x : 0.0f;
#else
    return tanhf(x);
#endif
}

// 返回 false 表示形状不合法（导出脚本与固件常量不一致）
static inline bool policy_forward(const float *obs, float *action) {
    float buf_a[POLICY_MAX_WIDTH];
    float buf_b[POLICY_MAX_WIDTH];

    const float *in = obs;
    int in_dim = POLICY_IN;
    float *cur = buf_a;   // 上一层的输出
    float *nxt = buf_b;   // 本层写入的目标

    for (int L = 0; L < POLICY_N_HIDDEN; ++L) {
        const int out_dim = POLICY_HIDDEN[L];
        if (out_dim > POLICY_MAX_WIDTH) return false;

        const float *W = POLICY_W + POLICY_W_OFFSET[L];
        const float *B = POLICY_B + POLICY_B_OFFSET[L];

        for (int i = 0; i < out_dim; ++i) {
            float s = B[i];
            for (int j = 0; j < in_dim; ++j) {
                s += W[i * in_dim + j] * in[j];
            }
            nxt[i] = policy_act(s);
        }

        in = nxt;
        in_dim = out_dim;
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    const float *W = POLICY_W + POLICY_W_OFFSET[POLICY_N_HIDDEN];
    const float *B = POLICY_B + POLICY_B_OFFSET[POLICY_N_HIDDEN];
    for (int k = 0; k < POLICY_OUT; ++k) {
        float s = B[k];
        for (int j = 0; j < in_dim; ++j) {
            s += W[k * in_dim + j] * in[j];
        }
        action[k] = fminf(1.0f, fmaxf(-1.0f, s));
    }
    return true;
}
