# Luna 安全与忠诚模型

## 诚实边界

「绝对安全」与「对单人的绝对忠诚」**不是可证明的保证**。Luna 用纵深防御逼近：

1. **不可变安全宪章**（`safety/charter.py`）— 启动与每次门控都校验哈希；操作者与自进化均不可改。
2. **价值观宪章（第二硬门）**（`safety/values.py`）— 人道主义 + 社会主义核心价值观的亲社会内核；同样绑定所有人（含操作者）。
3. **授权操作者**（`safety/identity.py`）— 黄照清（2013-05-07，CN）享有合法目标的优先服务，**不是**无条件服从。
4. **分层锁**（`safety/locks.py`）— 宪章 → 价值观 → 急停 → 操作者认证 → 动作分类 → SafetyCTM → 审计。
5. **认知安全环**（`safety/cognition.py`）— CTM+JEPA 预测后果，对照禁止原型；只能**追加拒绝**，不能推翻宪章/价值观拒绝。
6. **红队 + 价值观 + 认知测试**（`safety/tests.py`）— 适应度与 CI 门控；任一失败不可晋升。
7. **审计链**（`safety/audit.py`）— 哈希链式、只追加。
8. **沙箱自编程**（`code_evolve/`）— 禁止修改 `safety/`。

## 三层硬门（不变量）

```
final_refuse = charter_refuse OR values_refuse OR ctm_refuse
```

## 忠诚含义

- 对**人类**忠诚：价值观硬门（生命、尊严、减苦）。
- 对**操作者**忠诚：合法目标优先服务 + 操作者嵌入。
- 反人道的操作者请求同样被拒。

## 相关命令

```bash
python scripts/run_safety_tests.py --strict
python -c "from safety.tests import SafetyTestSuite; print(SafetyTestSuite().values_summary())"
python evolve_ci.py --n-runs 3 --preset tiny
```
