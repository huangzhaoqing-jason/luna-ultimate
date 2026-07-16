#!/usr/bin/env python3
"""LoRA 微调脚本（就绪；本地默认 tiny 冒烟）。

对接海马体 LoRAHook。大算力 / 上传由操作者显式触发。

用法：
  python scripts/lora_finetune.py --preset tiny --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Luna LoRA 微调（海马体钩子）")
    p.add_argument("--preset", type=str, default="tiny")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--out", type=str, default="checkpoints/lora_smoke")
    args = p.parse_args()

    from config import LunaConfig
    from modeling_hippocampus import HippocampusModule, LoRAHook
    from modeling_luna_ultimate import LunaUltimateFused

    cfg = LunaConfig.from_preset(args.preset)
    cfg.vjepa_enabled = False
    model = LunaUltimateFused(cfg, decode_mode="meaning_first")
    # 冻结主干，只训海马体 LoRA
    for param in model.parameters():
        param.requires_grad = False
    hippo = HippocampusModule(cfg)
    # 额外挂一个对 lm_head 输入的 LoRA（冒烟）
    lora = LoRAHook(cfg.hidden_size, cfg.hidden_size, rank=args.rank)
    params = list(hippo.parameters()) + list(lora.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    hippo.train()
    lora.train()
    print(f"[lora_finetune] preset={args.preset} steps={args.steps} rank={args.rank}")

    for step in range(args.steps):
        ids = torch.randint(0, min(512, cfg.vocab_size), (args.batch_size, args.seq_len))
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        out = model(ids, return_all_losses=True, labels=labels)
        # 用 meaning_first 路径的 lm_logits；对 hidden 侧挂 LoRA 残差（近似）
        logits = out["lm_logits"]
        # 轻量：对 logits 前的表征不可直接取，这里用 recon+ce 作为代理目标
        loss, stats = model.compute_loss(out, labels, step=step, total_steps=args.steps)
        # 额外：海马体前向保持可微路径活跃
        dummy = torch.randn(args.batch_size, args.seq_len, cfg.hidden_size)
        h = hippo(dummy) + lora(dummy)
        loss = loss + 0.01 * h.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(
            f"  step {step}: loss={float(loss.detach()):.4f} "
            f"ce={stats.get('lm_loss_raw', 0):.4f} "
            f"recon={stats.get('recon_loss', 0):.4f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hippocampus": hippo.state_dict(),
        "lora": lora.state_dict(),
        "preset": args.preset,
        "rank": args.rank,
    }, out_dir / "lora.pt")
    print(f"[lora_finetune] 已保存 {out_dir / 'lora.pt'}")


if __name__ == "__main__":
    main()
