// ============================================================
//  占位版策略文件（全零权重，能编译、不会动）
//
//  用真实策略覆盖它：
//      python tools/export_policy.py runs/ppo_fast/best_model.zip -o include/policy.h
//
//  生成后这个文件里的宏与数组会被替换掉，main.cpp 不用改。
// ============================================================
#pragma once

#define POLICY_IN 10
#define POLICY_OUT 2
#define POLICY_N_HIDDEN 1
#define POLICY_ACT_ELU 1     // 导出脚本会按模型实际激活函数改写

static const int POLICY_HIDDEN[POLICY_N_HIDDEN] = {4};

// 每层权重按行展开：第 i 行 = 第 i 个神经元的输入权重
static const float POLICY_W[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
};

static const float POLICY_B[] = {
    0, 0, 0, 0,
    0, 0,
};

// 每层权重在 POLICY_W / POLICY_B 里的起始下标
static const int POLICY_W_OFFSET[POLICY_N_HIDDEN + 1] = {0, 40};
static const int POLICY_B_OFFSET[POLICY_N_HIDDEN + 1] = {0, 4};
