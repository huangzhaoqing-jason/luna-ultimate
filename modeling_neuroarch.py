"""Whole-Brain Hybrid Cognitive Architecture (WB-HCA) — 7 大区 + 次级分区。

按用户提供的「7 大核心脑区 + 次级区域」规格，把脑区映射到 Luna 模块。
次级区域用「路由键」表达，不堆模型：共享骨干 + 轻量头/开关。

7 大区：
  1. 前额叶皮层 (prefrontal)  — CTM 全局推理 + MeaningPlanner
  2. 顶叶 (parietal)          — 神经符号 + GNN
  3. 颞叶 (temporal)          — 稀疏 MoE + RAG
  4. 海马体+内嗅皮层 (hippocampus) — 经验回放 + LoRA
  5. 小脑 (cerebellum)        — 蒸馏缓存
  6. 脑干+基底神经节+杏仁核 (brainstem) — 安全门控
  7. 丘脑 (thalamus)          — GWT 全局路由
+ 运动动作中枢 (motor)        — M1 / premotor / SMA（Action 引擎）

注：这是工程类比，不是生物学等价声明。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


# ==================== 次级区域枚举 ====================

class SubRegion(enum.Enum):
    # 前额叶
    DLPFC = "dlpfc"           # 背外侧：长链推理 / 工作记忆 / Agent 拆解
    VMPFC = "vmpfc"           # 腹内侧：价值判断 / 风险 / 道德
    OFC = "ofc"               # 眶额：奖惩 / 反馈 / 错误复盘 / 质量校验
    ACC = "acc"               # 前扣带回：冲突检测 / 算力调度
    # 顶叶
    LIP = "lip"               # 左顶下小叶：数字 / 代数 / 公式 / 代码语法
    RIP = "rip"               # 右顶下小叶：3D 空间 / 几何 / 场景
    SPL = "spl"               # 顶上小叶：细节 / 坐标 / 物理 / 碰撞
    AG = "ag"                 # 角回：文字↔数字↔符号互转
    # 颞叶
    WERNICKE = "wernicke"     # 左颞上回：语言语义 / 语法 / 代码语义
    MT_MIDDLE = "mt_middle"   # 颞中/下回：视觉语义 / 物体 / 常识
    MTL = "mtl"               # 内侧颞叶：长期事实记忆 / 知识库
    # 海马体
    HIPPO = "hippo"           # 经验回放 / 自进化
    ENTORHINAL = "entorhinal" # 短期→长期转化 / 网格化
    # 小脑
    CEREBELLUM = "cerebellum" # 技能缓存 / 高速重复
    # 脑干安全
    BRAINSTEM = "brainstem"   # 延髓/中脑/脑桥：本能拦截 / 急停
    BG = "bg"                 # 基底神经节：行为约束 / 奖惩信号
    AMYGDALA = "amygdala"     # 杏仁核：危险识别
    # 丘脑
    THALAMUS = "thalamus"     # GWT 中转
    # 运动
    M1 = "m1"                 # 初级运动皮层：标准化动作指令
    PREMOTOR = "premotor"     # 前运动皮层：动作流程编排
    SMA = "sma"               # 辅助运动区：长周期动作模板


@dataclass(frozen=True)
class BrainRegion:
    name: str
    region_id: str            # 7 大区之一
    biological_analogue: str
    module: str
    function: str
    wake_cost: str            # low | mid | high
    default_awake: bool
    sub_regions: tuple = ()


# ==================== 7 大区表 ====================

WB_HCA: Dict[str, BrainRegion] = {
    # 1. 前额叶皮层
    "prefrontal": BrainRegion(
        name="prefrontal",
        region_id="1_prefrontal",
        biological_analogue="prefrontal cortex / executive control",
        module="modeling_ctm.CTM + modeling_meaning.MeaningPlanner",
        function="全局推理总控：长链推理、价值判断、反馈复盘、冲突检测",
        wake_cost="mid",
        default_awake=True,
        sub_regions=(SubRegion.DLPFC, SubRegion.VMPFC, SubRegion.OFC, SubRegion.ACC),
    ),
    # 2. 顶叶
    "parietal": BrainRegion(
        name="parietal",
        region_id="2_parietal",
        biological_analogue="parietal lobe / spatial-math-symbolic",
        module="modeling_parietal.ParietalReasoner (+ GNN 占位)",
        function="数学/空间/符号计算：数字运算、3D 几何、坐标物理、文数互转",
        wake_cost="mid",
        default_awake=False,
        sub_regions=(SubRegion.LIP, SubRegion.RIP, SubRegion.SPL, SubRegion.AG),
    ),
    # 3. 颞叶
    "temporal": BrainRegion(
        name="temporal",
        region_id="3_temporal",
        biological_analogue="temporal lobe / language + memory",
        module="modeling_flashmoe.FlashMoE + rag/ + modeling_vjepa",
        function="语言语义、长期知识、视觉语义、常识；RAG 召回",
        wake_cost="mid",
        default_awake=True,
        sub_regions=(SubRegion.WERNICKE, SubRegion.MT_MIDDLE, SubRegion.MTL),
    ),
    # 4. 海马体 + 内嗅皮层
    "hippocampus": BrainRegion(
        name="hippocampus",
        region_id="4_hippocampus",
        biological_analogue="hippocampus + entorhinal cortex",
        module="modeling_hippocampus.EpisodicMemory + LoRAHook",
        function="经验回放、短→长转化、自进化复盘、新旧知识融合",
        wake_cost="mid",
        default_awake=False,
        sub_regions=(SubRegion.HIPPO, SubRegion.ENTORHINAL),
    ),
    # 5. 小脑
    "cerebellum": BrainRegion(
        name="cerebellum",
        region_id="5_cerebellum",
        biological_analogue="cerebellum / predictive control + skill cache",
        module="modeling_cerebellum.CerebellarCorrector + Mamba2",
        function="固化技能、重复任务极速、动作纠错回路、缓存命中跳主干",
        wake_cost="low",
        default_awake=True,
        sub_regions=(SubRegion.CEREBELLUM,),
    ),
    # 6. 脑干 + 基底神经节 + 杏仁核
    "brainstem": BrainRegion(
        name="brainstem",
        region_id="6_brainstem",
        biological_analogue="brainstem + basal ganglia + amygdala",
        module="safety/charter + values + locks + cognition + action/action_gate",
        function="本能拦截、行为约束、危险识别、动作门控、急停",
        wake_cost="low",
        default_awake=True,
        sub_regions=(SubRegion.BRAINSTEM, SubRegion.BG, SubRegion.AMYGDALA),
    ),
    # 7. 丘脑
    "thalamus": BrainRegion(
        name="thalamus",
        region_id="7_thalamus",
        biological_analogue="thalamus / GWT relay",
        module="modeling_thalamus.ThalamusRouter",
        function="唯一中转：分配任务、协同编排、整合输出、休眠闲置脑区",
        wake_cost="low",
        default_awake=True,
        sub_regions=(SubRegion.THALAMUS,),
    ),
    # + 运动动作中枢
    "motor": BrainRegion(
        name="motor",
        region_id="8_motor",
        biological_analogue="motor cortex / action output",
        module="modeling_motor.M1ActionHead + PremotorPlanner + SMACache",
        function="标准化动作指令、流程编排、模板缓存；过脑干动作门",
        wake_cost="mid",
        default_awake=False,
        sub_regions=(SubRegion.M1, SubRegion.PREMOTOR, SubRegion.SMA),
    ),
    # 复用：长上下文 / 注意力 / 世界模型（隶属于上述大区的支撑模块）
    "working_memory": BrainRegion(
        name="working_memory",
        region_id="1_prefrontal",
        biological_analogue="working memory / persistent state",
        module="modeling_mamba2.Mamba2Block",
        function="长上下文状态压缩",
        wake_cost="mid",
        default_awake=True,
        sub_regions=(),
    ),
    "attention_networks": BrainRegion(
        name="attention_networks",
        region_id="2_parietal",
        biological_analogue="frontoparietal attention",
        module="modeling_mla.MLA",
        function="压缩 KV 的选择性注意",
        wake_cost="mid",
        default_awake=True,
        sub_regions=(),
    ),
    "world_model": BrainRegion(
        name="world_model",
        region_id="4_hippocampus",
        biological_analogue="predictive coding / world model",
        module="modeling_world_action.WorldActionModule",
        function="JEPA 下一状态预测 + 动作先验",
        wake_cost="high",
        default_awake=False,
        sub_regions=(),
    ),
    "language_areas": BrainRegion(
        name="language_areas",
        region_id="3_temporal",
        biological_analogue="Broca/Wernicke analogues",
        module="modeling_meaning.MeaningFirstDecoder (no AR fallback)",
        function="按目标语义解码 token",
        wake_cost="mid",
        default_awake=True,
        sub_regions=(),
    ),
    "default_mode": BrainRegion(
        name="default_mode",
        region_id="4_hippocampus",
        biological_analogue="default mode network",
        module="evolve/ + code_evolve/ (offline)",
        function="自反思、架构搜索、离线演练",
        wake_cost="high",
        default_awake=False,
        sub_regions=(),
    ),
}


# ==================== 任务 → 脑区 + 次级区域 ====================

TASK_REGION_BIAS: Dict[str, Sequence[str]] = {
    "chat": ("language_areas", "working_memory", "attention_networks", "prefrontal", "cerebellum"),
    "code": ("language_areas", "cerebellum", "temporal", "prefrontal", "working_memory"),
    "math": ("parietal", "cerebellum", "prefrontal", "working_memory"),
    "vision": ("temporal", "parietal", "attention_networks"),
    "planning": ("prefrontal", "hippocampus", "working_memory", "default_mode"),
    "memory": ("hippocampus", "working_memory", "language_areas"),
    "safety": ("brainstem", "prefrontal"),
    "vla": ("motor", "temporal", "prefrontal", "thalamus"),
    "wa": ("world_model", "motor", "prefrontal", "hippocampus"),
    "action": ("motor", "prefrontal", "cerebellum", "brainstem", "thalamus"),
}

# 任务 → 推荐次级区域（路由键；用于在共享骨干上做轻量切换）
TASK_SUBREGION: Dict[str, Sequence[SubRegion]] = {
    "chat": (SubRegion.WERNICKE, SubRegion.OFC, SubRegion.CEREBELLUM),
    "code": (SubRegion.WERNICKE, SubRegion.LIP, SubRegion.M1, SubRegion.CEREBELLUM),
    "math": (SubRegion.LIP, SubRegion.SPL, SubRegion.AG, SubRegion.DLPFC),
    "vision": (SubRegion.MT_MIDDLE, SubRegion.RIP, SubRegion.SPL),
    "planning": (SubRegion.DLPFC, SubRegion.VMPFC, SubRegion.ACC, SubRegion.HIPPO),
    "memory": (SubRegion.HIPPO, SubRegion.ENTORHINAL, SubRegion.MTL),
    "safety": (SubRegion.AMYGDALA, SubRegion.BG, SubRegion.BRAINSTEM, SubRegion.VMPFC),
    "vla": (SubRegion.M1, SubRegion.PREMOTOR, SubRegion.WERNICKE, SubRegion.DLPFC),
    "wa": (SubRegion.HIPPO, SubRegion.M1, SubRegion.DLPFC, SubRegion.ENTORHINAL),
    "action": (SubRegion.M1, SubRegion.PREMOTOR, SubRegion.SMA, SubRegion.BRAINSTEM, SubRegion.BG),
}


def list_regions() -> List[BrainRegion]:
    return list(WB_HCA.values())


def regions_for_task(task: str) -> List[BrainRegion]:
    keys = TASK_REGION_BIAS.get(task, TASK_REGION_BIAS["chat"])
    return [WB_HCA[k] for k in keys if k in WB_HCA]


def subregions_for_task(task: str) -> List[SubRegion]:
    return list(TASK_SUBREGION.get(task, TASK_SUBREGION["chat"]))


def awake_modules(task: str) -> List[str]:
    """任务相关模块 + 默认常驻。"""
    awake = {r.module for r in WB_HCA.values() if r.default_awake}
    for r in regions_for_task(task):
        awake.add(r.module)
    return sorted(awake)


def describe_architecture() -> str:
    lines = ["# WB-HCA 7 大区 + 次级分区", ""]
    for r in list_regions():
        subs = ", ".join(s.value for s in r.sub_regions) or "-"
        lines.append(
            f"- **{r.name}** ({r.region_id}) ← {r.biological_analogue}\n"
            f"  - module: `{r.module}`\n"
            f"  - function: {r.function}\n"
            f"  - sub_regions: {subs}\n"
            f"  - wake_cost: {r.wake_cost}; default_awake: {r.default_awake}"
        )
    return "\n".join(lines)


def suggest_route(task: Optional[str] = None) -> Dict[str, object]:
    task = task or "chat"
    regs = regions_for_task(task)
    return {
        "task": task,
        "regions": [r.name for r in regs],
        "region_ids": sorted({r.region_id for r in regs}),
        "sub_regions": [s.value for s in subregions_for_task(task)],
        "modules": awake_modules(task),
        "decode_mode": "meaning_first",
        "ar_fallback": False,
    }
