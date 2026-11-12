// ============================================================
//  SelfRising · ESP32 固件骨架（PlatformIO / Arduino）
//
//  与仿真对齐的四个量：
//    控制频率 50 Hz · 动作增量 0.08 rad · 限幅 ±1.55 rad · 观测 10 维
//
//  ⚠ 第一次上电请把 DRY_RUN 设为 1：只打印观测与动作，不动舵机。
//    确认 obs[2] 在站直时接近 -1、推倒后变化合理，再改成 0。
// ============================================================
#include <Arduino.h>
#include <Adafruit_BNO055.h>
#include <ESP32Servo.h>
#include <Wire.h>

#include "policy.h"
#include "policy_runtime.h"

// ------------------------------------------------------------
//  1. 与仿真对齐的参数（改这里 = 改仿真里的同名参数）
// ------------------------------------------------------------
static const float CTRL_HZ = 50.0f;
static const uint32_t PERIOD_US = (uint32_t)(1000000.0f / CTRL_HZ);
static const float ACTION_DELTA = 0.08f;   // env.action_delta
static const float ACTION_LIMIT = 1.55f;   // env.action_limit
static const float OBS_HEIGHT = 0.08f;     // 没有高度传感器，用常量

// ------------------------------------------------------------
//  2. 硬件引脚与标定（必须按你的实物改）
// ------------------------------------------------------------
static const int PIN_SERVO_LOW = 25;    // 下方舵机 A0
static const int PIN_SERVO_HIGH = 26;   // 上方舵机 A1
static const int PIN_BOOT_BTN = 0;      // 开发板 BOOT 键：按住 = 回中位并暂停

static const int SERVO_MIN_US = 500;
static const int SERVO_MAX_US = 2500;
static const int SERVO_TRIM_LOW_US = 0;    // 中位微调
static const int SERVO_TRIM_HIGH_US = 0;
static const int SERVO_DIR_LOW = 1;        // 装反了改成 -1
static const int SERVO_DIR_HIGH = 1;

// BNO055 的安装方向：如果站直时 obs[2] 不是 -1 而是 +1，把它改成 1
#define GRAVITY_FLIP 0

// 1 = 只打印不动舵机；0 = 真正驱动
#define DRY_RUN 1

// 串口打印间隔（控制步为单位，50 Hz 下每 10 步 = 5 Hz）
static const int PRINT_EVERY = 10;

// ------------------------------------------------------------
//  3. 全局对象
// ------------------------------------------------------------
static Adafruit_BNO055 bno(-1, 0x29, &Wire);
static Servo servo_low, servo_high;
static bool imu_ok = false;

static float target[2] = {0.0f, 0.0f};        // 当前目标角（rad）
static float prev_action[2] = {0.0f, 0.0f};   // 上一帧动作（给观测用）
static float prev_target[2] = {0.0f, 0.0f};   // 上一帧目标（算角速度）
static bool paused = true;

// 观测的取值范围，与 env.py 的 observation_space 一致
static const float OBS_LOW[POLICY_IN] = {-1, -1, -1, -1.6f, -1.6f, -20, -20, -1.6f, -1.6f, -0.1f};
static const float OBS_HIGH[POLICY_IN] = {1, 1, 1, 1.6f, 1.6f, 20, 20, 1.6f, 1.6f, 0.5f};

// ------------------------------------------------------------
//  4. 观测拼装（顺序必须与 env.py 的 _obs() 完全一致）
//     0-2 重力在机体系 / 3-4 关节角 / 5-6 角速度
//     7-8 上一目标角 / 9 高度
// ------------------------------------------------------------
static void read_observation(float *obs) {
    // 0-2：BNO055 的重力向量，归一化成单位向量
    float gx = 0.0f, gy = 0.0f, gz = -1.0f;
    if (imu_ok) {
        imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
        gx = g.x();
        gy = g.y();
        gz = g.z();
        const float n = sqrtf(gx * gx + gy * gy + gz * gz);
        if (n > 1e-3f) {
            gx /= n;
            gy /= n;
            gz /= n;
        }
#if GRAVITY_FLIP
        gx = -gx;
        gy = -gy;
        gz = -gz;
#endif
    }
    obs[0] = gx;
    obs[1] = gy;
    obs[2] = gz;

    // 3-4：没有编码器，用“上次下发的目标角”当作关节角的近似
    obs[3] = target[0];
    obs[4] = target[1];

    // 5-6：目标角的变化率，当作角速度
    obs[5] = (target[0] - prev_target[0]) * CTRL_HZ;
    obs[6] = (target[1] - prev_target[1]) * CTRL_HZ;

    // 7-8：上一控制目标
    obs[7] = target[0];
    obs[8] = target[1];

    // 9：高度常量
    obs[9] = OBS_HEIGHT;

    for (int i = 0; i < POLICY_IN; ++i) {
        if (obs[i] < OBS_LOW[i]) obs[i] = OBS_LOW[i];
        if (obs[i] > OBS_HIGH[i]) obs[i] = OBS_HIGH[i];
    }
}

// ------------------------------------------------------------
//  5. 舵机输出：弧度 -> 脉宽
// ------------------------------------------------------------
static void servo_write(const float *t) {
    if (DRY_RUN) return;

    float a0 = t[0] * SERVO_DIR_LOW;
    float a1 = t[1] * SERVO_DIR_HIGH;

    // 把 ±ACTION_LIMIT 映射到 500~2500 us
    const float span = ACTION_LIMIT;
    int us0 = (int)(1500 + SERVO_TRIM_LOW_US + (a0 / span) * 1000.0f);
    int us1 = (int)(1500 + SERVO_TRIM_HIGH_US + (a1 / span) * 1000.0f);

    us0 = constrain(us0, SERVO_MIN_US, SERVO_MAX_US);
    us1 = constrain(us1, SERVO_MIN_US, SERVO_MAX_US);

    servo_low.writeMicroseconds(us0);
    servo_high.writeMicroseconds(us1);
}

static void go_neutral() {
    target[0] = 0.0f;
    target[1] = 0.0f;
    prev_target[0] = 0.0f;
    prev_target[1] = 0.0f;
    prev_action[0] = 0.0f;
    prev_action[1] = 0.0f;
    servo_write(target);
}

// ------------------------------------------------------------
//  6. setup
// ------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.println("SelfRising firmware booting...");
    Serial.printf("policy: in=%d out=%d hidden=%d act=%s\n", POLICY_IN, POLICY_OUT, POLICY_N_HIDDEN,
#if defined(POLICY_ACT_ELU)
                  "elu"
#elif defined(POLICY_ACT_RELU)
                  "relu"
#else
                  "tanh"
#endif
    );

    pinMode(PIN_BOOT_BTN, INPUT_PULLUP);

    Wire.begin();
    imu_ok = bno.begin();
    if (!imu_ok) {
        Serial.println("[WARN] BNO055 未找到，观测退化为 (0,0,-1)。请检查 SDA/SCL 与地址 0x29。");
    } else {
        bno.setExtCrystalUse(true);
        delay(500);
    }

    servo_low.setPeriodHertz(50);
    servo_high.setPeriodHertz(50);
    servo_low.attach(PIN_SERVO_LOW, SERVO_MIN_US, SERVO_MAX_US);
    servo_high.attach(PIN_SERVO_HIGH, SERVO_MIN_US, SERVO_MAX_US);

    go_neutral();
    Serial.println("[INFO] 保持中位 3 秒后进入控制循环；按住 BOOT 键可随时回中位暂停。");
    delay(3000);
    paused = false;
}

// ------------------------------------------------------------
//  7. 主循环：固定 50 Hz
// ------------------------------------------------------------
void loop() {
    static uint32_t last_us = 0;
    static uint32_t steps = 0;

    const uint32_t now = micros();
    if (now - last_us < PERIOD_US) return;
    last_us = now;

    // 安全：按住 BOOT 键 -> 回中位并暂停
    if (digitalRead(PIN_BOOT_BTN) == LOW) {
        if (!paused) {
            paused = true;
            go_neutral();
            Serial.println("[SAFE] paused");
        }
        return;
    } else if (paused) {
        paused = false;
        Serial.println("[RUN] resumed");
    }

    float obs[POLICY_IN];
    read_observation(obs);

    float action[POLICY_OUT];
    if (!policy_forward(obs, action)) {
        Serial.println("[ERR] policy shape mismatch");
        go_neutral();
        delay(1000);
        return;
    }

    prev_target[0] = target[0];
    prev_target[1] = target[1];

    // 增量位置控制，与 env.py 的 step() 一致
    target[0] = constrain(target[0] + ACTION_DELTA * action[0], -ACTION_LIMIT, ACTION_LIMIT);
    target[1] = constrain(target[1] + ACTION_DELTA * action[1], -ACTION_LIMIT, ACTION_LIMIT);

    prev_action[0] = action[0];
    prev_action[1] = action[1];

    servo_write(target);

    if (++steps % PRINT_EVERY == 0) {
        Serial.printf("obs g=(%.2f,%.2f,%.2f) q=(%.2f,%.2f) tgt=(%.2f,%.2f) act=(%.2f,%.2f)\n",
                      obs[0], obs[1], obs[2], obs[3], obs[4], target[0], target[1], action[0], action[1]);
    }
}
