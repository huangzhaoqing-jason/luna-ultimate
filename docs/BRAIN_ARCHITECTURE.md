# Luna Neuro-Architecture: Brain-Region → Structure Mapping

## Goal

Top-tier capability at the **lowest local hardware cost**. We analyze the human
brain by region, map each region's function to the Luna module that implements
it at the highest performance-per-FLOP, and run the whole stack on the
`tiny`/`1b` ladder for local CPU/low-end-GPU inference.

## Brain regions → Luna structures

| Brain region | Function | Luna module | Why this module (perf-per-FLOP) |
|---|---|---|---|
| Prefrontal cortex | Executive function, deliberation, planning | `CTM` + `MeaningPlanner` | Adaptive ticks + target-meaning plan before act. |
| Hippocampus | Episodic memory, recall | `modeling_hippocampus` + `MLA` | Episodes/LoRA + compressed KV for long recall. |
| Cerebellum | Fast habitual / sequential processing | `modeling_cerebellum` + `Mamba2-SSD` | LRU skill cache + O(1) state encode. |
| Parietal | Spatial / numeric / symbolic | `modeling_parietal` | Neural intuition + lightweight AST verify. |
| Thalamus | Sensory relay / routing (GWT) | `modeling_thalamus.ThalamusRouter` | Task-type wake + expert budget (~10–100%). |
| Cortex columns | Specialized capabilities | `FlashMoE` experts + `rag/` | Sparse MoE + retrieval for knowledge tasks. |
| Amygdala | Threat / safety detection | `SafetyCTM` | Cognitive safety loop: predicts consequence, refuses threats. |
| Basal ganglia | Action selection / go-no-go | `SafetyLock.gate` | Triple floor: charter OR values OR ctm refuse. |
| Language areas | Speech / token output | `MeaningFirstDecoder` (Q1B, no AR fallback) | Decode from JEPA target meaning; collapse marked only. |
| Corpus callosum | Inter-region integration | CTM residual injection | Global CTM state broadcast to every layer. |
| Sensory cortex | Perception | `V-JEPA` + `ActionEncoder` | JEPA world model + text encoder = sensory front. |
| Mirror neurons / empathy | Align with others' welfare | `ValuesCharter` + `ForbiddenPrototypeSet` | Values floor + anti-humanitarian prototypes → prosocial gating. |

## Data flow (cognitive brain)

```mermaid
flowchart TB
  Sense["Sensory cortex (V-JEPA / ActionEncoder)"]
  Thal["Thalamus (dynamic route + expert budget)"]
  Cereb["Cerebellum (cache + Mamba2 front)"]
  Hippo["Hippocampus (episodes + MLA back)"]
  Parietal["Parietal (neuro-symbolic)"]
  PFC["Prefrontal (CTM + MeaningPlanner)"]
  Col["Cortex columns (FlashMoE + RAG)"]
  Decode["MeaningFirstDecoder (no AR fallback)"]
  Amyg["Amygdala (SafetyCTM)"]
  BG["Basal ganglia (SafetyLock gate)"]
  Emp["Empathy (ValuesCharter + prototypes)"]
  Out["Motor / speech output"]

  Sense --> Thal --> Cereb --> Hippo --> Decode
  Thal --> Parietal
  Thal --> Col
  PFC -->|target meaning| Decode
  PFC -->|residual every N layers| Cereb
  Amyg --> BG
  Emp --> BG
  Decode --> BG
  BG -->|allow| Out
  BG -->|refuse| Block["Refuse + audit"]
```

## Why this is the highest-perf-per-FLOP structure

- **Adaptive compute** (CTM): pays full depth only on hard inputs.
- **O(1) memory encode** (Mamba2-SSD): no KV growth on long context.
- **Compressed memory** (MLA): 32× KV compression → long recall at low VRAM.
- **Sparse capacity** (FlashMoE): 550B total / ~77B active → big brain, small per-token cost.
- **Cognitive safety** (SafetyCTM): threat detection without a separate heavy model.
- **Values floor** (ValuesCharter): prosocial alignment as a cheap immutable check.

## Lowest-local-hardware operating point

| Preset | Hardware | Role |
|---|---|---|
| `tiny` | CPU / 1×8GB | dev, evolve smoke, safety+values tests |
| `1b` | 1×24GB | local e2e on a consumer GPU |
| `7b` | 1–8×80GB | capability sprint |
| `550b` | multi-node | challenge (only after genomes stabilize) |

For "top model at lowest local cost," run `1b` locally with INT4 KV + CTM
early-exit + MoE hot/cold expert placement. The cost model (`cost_model.py`)
quantifies $/MTok and peak VRAM per preset.

## Safety + loyalty model (brain analogue)

- **Amygdala + basal ganglia** = `SafetyLock` (charter hard floor + CTM soft judge).
- **Empathy / values** = `ValuesCharter` (humanitarian + socialist core values'
  prosocial substance) — an immutable second hard floor. The AI is loyal to
  **humanity** (values floor) and to the **operator** (黄照清) for priority
  service, but values bind the operator too: anti-humanitarian operator
  requests are refused.
- **Self-evolution** = `evolve/` + `code_evolve/`, gated by safety + values
  + cognitive tests, sandboxed, audited. The brain optimizes itself without
  ever weakening the amygdala, basal ganglia, or empathy layers (`safety/`
  is immutable to self-evolution).
