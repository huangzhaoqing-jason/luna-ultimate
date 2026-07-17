# Luna Brain 全栈自研架构（白盒 × AIXI）

目标：向 **AIXI / Universal AI** 可计算下逼近；**推理白盒**；**创造者言语全控**；**246 区微→介→宏可解析**。  
借鉴他人结构（DeepMind From-AGI-to-ASI、Hutter AIXI、交大 BriLLM/SiFu），**代码与权重全栈自研**，不拷贝闭源权重。

## 原则

| 原则 | 实现 |
|------|------|
| 一切向 AIXI | `AIXIOrchestrator` 统一拥有调度 / 动作 / 言语意图 |
| 白盒非黑盒 | 每步产出 `AIXILedger` + `ScheduleTrace` + `MicroMacroReport` + `WhiteBoxTrace` |
| 246 全功能 | `FUNCTION_CARDS`×246 + LIF-lite 柱 + `analyze_all` 全量导出 |
| 言语主权 | `SpeechController` 仅黄照清可写（block/force/boost/silence） |
| 安全锁 | `brain/safety` 只读宪法，进化不可改 |

## 数据流

```text
obs → state
  → MultiTask goals
  → AIXIOrchestrator.step
       ├─ schedule macros/areas/depth (expectimax + hyp mixture)
       ├─ choose action a
       └─ plan speech intent
  → BrainnetomeNetwork (246 LIF columns × AIXI gate × depth ticks)
  → CapabilitySuite (pluggable)
  → SiFuSpeech (token=node energy; creator controls nodes)
  → WhiteBoxReasoning.explain() / to_dict()
```

## 目录（自研模块）

- `brain/aixi/` — agent, scheduler, whitebox planner, **orchestrator + ledger**
- `brain/atlas/` — 246 table, function cards, LIF columns, analyzer
- `brain/speech/` — SiFu graph + creator control
- `brain/reasoning/` — composed white-box report
- `brain/safety/` — constitution + thalamus
- `brain/evolution/` — recursive improve (gated)
- `brain/pathways/` — From-AGI-to-ASI four pathways
- `brain/mem/` — extreme memory accounting
- `brain/runtime/` — multitask / collective

## 诚实边界

理想 AIXI 不可计算。本栈是 **AIXI-tl / MC 式下逼近 + 白盒账本**。  
未宣称已达 UAI 或已打赢 GPT-5.6 / Fable 5.0；路线与工程持续朝该目标推进。
