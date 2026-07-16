#!/usr/bin/env python3
"""蒸馏 Luna → Llama 结构学生，供 Ollama/GGUF 导出（Track A）。

完整 CTM×Mamba×MLA×MoE 不能进 Ollama；此脚本产出兼容学生权重 +
HuggingFace 布局，并留下 llama.cpp 转换钩子。

用法：
  python3 export/distill_ollama.py --preset tiny --steps 3 --out export/out/luna-distill
  # 有 llama.cpp 时：
  # python3 export/distill_ollama.py --convert-gguf --llama-cpp /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class StudentConfig:
    vocab_size: int = 4096
    hidden_size: int = 256
    num_layers: int = 4
    n_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5


class LlamaStudentBlock(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        d = cfg.hidden_size
        self.attn_norm = nn.LayerNorm(d, eps=cfg.rms_norm_eps)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ffn_norm = nn.LayerNorm(d, eps=cfg.rms_norm_eps)
        self.gate = nn.Linear(d, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(d, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, d, bias=False)
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.o(y)
        h = self.ffn_norm(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class LlamaStudent(nn.Module):
    """纯 Attention+MLP 学生，Ollama/llama.cpp 可识别的拓扑。"""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([LlamaStudentBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        # 可选视觉投影（VLM mmproj 占位）
        self.mmproj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


class VisionMMProj(nn.Module):
    """极简 mmproj：vision pooled → hidden（导出为独立权重）。"""

    def __init__(self, vision_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, hidden_size, bias=False)

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        return self.proj(vision)


def _student_from_preset(preset: str) -> StudentConfig:
    from config import LunaConfig
    c = LunaConfig.from_preset(preset)
    return StudentConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        num_layers=max(2, min(c.num_hidden_layers, 8)),
        n_heads=c.n_heads,
        intermediate_size=c.intermediate_size,
        max_position_embeddings=min(c.max_position_embeddings, 4096),
    )


def distill(
    preset: str = "tiny",
    steps: int = 5,
    batch_size: int = 2,
    seq_len: int = 32,
    lr: float = 1e-3,
    out_dir: str = "export/out/luna-distill",
    use_teacher: bool = True,
) -> Path:
    device = torch.device("cpu")
    scfg = _student_from_preset(preset)
    student = LlamaStudent(scfg).to(device)
    teacher = None
    if use_teacher:
        from config import LunaConfig
        from modeling_luna_ultimate import LunaUltimateFused
        tcfg = LunaConfig.from_preset(preset)
        tcfg.vjepa_enabled = False
        teacher = LunaUltimateFused(tcfg, decode_mode="meaning_first").to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    print(f"[distill] preset={preset} steps={steps} student_layers={scfg.num_layers}")

    for step in range(steps):
        ids = torch.randint(0, scfg.vocab_size, (batch_size, seq_len), device=device)
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        logits = student(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, scfg.vocab_size),
            labels[:, :-1].reshape(-1),
            ignore_index=-100,
        )
        if teacher is not None:
            with torch.no_grad():
                tout = teacher(ids, return_all_losses=True)
                tlogits = tout["lm_logits"]
            # KL on overlapping vocab dims
            v = min(tlogits.shape[-1], scfg.vocab_size)
            kl = F.kl_div(
                F.log_softmax(logits[:, :, :v], dim=-1),
                F.softmax(tlogits[:, :, :v], dim=-1),
                reduction="batchmean",
            )
            loss = loss + 0.1 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"  step {step}: loss={float(loss.detach()):.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # HF-ish layout
    torch.save(student.state_dict(), out / "pytorch_model.bin")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": scfg.vocab_size,
            "hidden_size": scfg.hidden_size,
            "intermediate_size": scfg.intermediate_size,
            "num_hidden_layers": scfg.num_layers,
            "num_attention_heads": scfg.n_heads,
            "max_position_embeddings": scfg.max_position_embeddings,
            "rms_norm_eps": scfg.rms_norm_eps,
            "torch_dtype": "float32",
            "luna_track": "ollama-distill",
            "luna_teacher_preset": preset,
            "note": "Distilled Llama-topology student for Ollama; not full Luna WB-HCA.",
        }, f, indent=2)
    # mmproj 占位
    mm = VisionMMProj(scfg.hidden_size, scfg.hidden_size)
    torch.save(mm.state_dict(), out / "mmproj.pt")
    # 复制 Modelfile
    mf_src = ROOT / "export" / "Modelfile.luna"
    mf_dst = out / "Modelfile"
    if mf_src.exists():
        text = mf_src.read_text(encoding="utf-8")
        text = text.replace("{{MODEL_GGUF}}", "./luna-distill-q4_k_m.gguf")
        text = text.replace("{{MMPROJ_GGUF}}", "./luna-mmproj-f16.gguf")
        mf_dst.write_text(text, encoding="utf-8")
    with open(out / "CONVERT.md", "w", encoding="utf-8") as f:
        f.write(
            "# GGUF 转换钩子\n\n"
            "需要本机安装 llama.cpp 后执行：\n\n"
            "```bash\n"
            "python <llama.cpp>/convert_hf_to_gguf.py ./ \\\n"
            "  --outfile luna-distill-f16.gguf\n"
            "<llama.cpp>/llama-quantize luna-distill-f16.gguf luna-distill-q4_k_m.gguf Q4_K_M\n"
            "ollama create luna -f Modelfile\n"
            "```\n\n"
            "`ollama run luna` = 蒸馏兼容版；完整 VLA/WA 请用 `python3 scripts/run_serve.py`。\n"
        )
    print(f"[distill] saved → {out}")
    return out


def try_convert_gguf(out_dir: Path, llama_cpp: str) -> bool:
    conv = Path(llama_cpp) / "convert_hf_to_gguf.py"
    if not conv.exists():
        # 新版路径可能不同
        alt = Path(llama_cpp) / "convert_hf_to_gguf.py"
        print(f"[distill] convert script not found under {llama_cpp}; skip GGUF")
        return False
    out_gguf = out_dir / "luna-distill-f16.gguf"
    try:
        subprocess.run(
            [sys.executable, str(conv), str(out_dir), "--outfile", str(out_gguf)],
            check=True,
            timeout=600,
        )
        print(f"[distill] wrote {out_gguf}")
        return True
    except Exception as e:
        print(f"[distill] GGUF convert failed (script ready): {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="蒸馏 Luna → Ollama 学生")
    p.add_argument("--preset", default="tiny")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--out", default="export/out/luna-distill")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--convert-gguf", action="store_true")
    p.add_argument("--llama-cpp", type=str, default="")
    args = p.parse_args()
    out = distill(
        preset=args.preset,
        steps=args.steps,
        out_dir=args.out,
        use_teacher=not args.no_teacher,
    )
    if args.convert_gguf and args.llama_cpp:
        try_convert_gguf(out, args.llama_cpp)


if __name__ == "__main__":
    main()
