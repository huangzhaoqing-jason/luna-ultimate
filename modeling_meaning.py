"""意义优先解码器（Q1B 纯意义优先，无自回归 fallback）。

设计思路（对应你的世界模型直觉）：
  - 不再 token-by-token 贪心跳跃，而是先用 JEPA 目标编码器从 prompt
    预测「最终目标语义状态」(target meaning latent)；
  - 再由 MeaningFirstDecoder **纯按这个目标语义**解码 token 序列：
    每一步的 logits 由「当前已生成 hidden」与「目标语义」的相似度驱动，
    朝目标语义走，而不是单纯的局部贪心；
  - 训练目标 = LM next-token loss + λ · 语义重建 loss（让目标语义能重建
    出原 token 序列），端到端。

Q3B 约束：无 fallback。若 tiny 上发生解码坍塌（重复/低熵/固定序列），
CollapseDetector 只标记、不回退——这是 Q3B 的预期行为，不是 bug。

复用点：CTMJEPAPredictor（modeling_ctm.py）已是「current→future」的
JEPA 预测器，这里把它当目标语义编码器用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LunaConfig
from modeling_ctm import CTMJEPAPredictor


class MeaningPlanner(nn.Module):
    """prompt hidden → 目标语义 latent。

    用一个轻量 MLP 把 prompt 的 pooled hidden 投影到目标语义空间，
    再用 CTMJEPAPredictor 做「目标状态」编码（复用 JEPA 目标编码器）。
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_neurons = config.ctm_n_neurons
        # prompt hidden → neuron state space（与 CTM.synapse 同维）
        self.to_neurons = nn.Linear(self.d_model, self.n_neurons, bias=False)
        # 复用 JEPA 目标编码器：current neuron state → future/target neuron state
        pred_hidden = min(256, max(32, self.n_neurons // 2))
        self.target_encoder = CTMJEPAPredictor(
            n_neurons=self.n_neurons,
            hidden_dim=pred_hidden,
            predict_horizon=1,
        )
        # target neuron state → 目标语义 latent (d_model)
        self.to_meaning = nn.Linear(self.n_neurons, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_model)

    def plan(self, prompt_hidden: torch.Tensor) -> torch.Tensor:
        """prompt_hidden: [B, L, d] → target_meaning: [B, d]"""
        pooled = prompt_hidden.mean(dim=1)              # [B, d]
        neuron_state = self.to_neurons(pooled)          # [B, n_neurons]
        target_neurons = self.target_encoder(neuron_state)  # [B, n_neurons]
        meaning = self.to_meaning(target_neurons)       # [B, d]
        return self.norm(meaning)


class MeaningFirstDecoder(nn.Module):
    """纯按目标语义解码 token（Q1B，无 fallback）。

    训练：给定 target_meaning + labels，每步 logits =
      lm_head(hidden_t) ⊙ gate(target_meaning similarity)
    其中 gate 用「当前 hidden 与目标语义的余弦相似度」调制 logits，
    使生成朝目标语义走。

    推理 generate：自回归采样，但 logits 同样由目标语义调制。
    """

    def __init__(self, config: LunaConfig, lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.vocab_size = config.vocab_size
        # 复用主模型的 lm_head（weight tying 由调用方传入）
        self.lm_head = lm_head  # 可能是 None，forward 时由调用方提供
        # 目标语义 → 每步调制门
        self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        # 重建 loss：从目标语义直接预测 token 分布（语义重建项）
        self.reconstruct_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.sim_temperature = 0.2

    def _similarity_gate(
        self, hidden: torch.Tensor, target_meaning: torch.Tensor
    ) -> torch.Tensor:
        """hidden: [B, L, d], target_meaning: [B, d] → gate [B, L, d]"""
        tm = self.gate_proj(target_meaning).unsqueeze(1)  # [B, 1, d]
        sim = F.cosine_similarity(hidden, tm, dim=-1, eps=1e-6)  # [B, L]
        # 软门：把相似度映射到 (0, ~2) 区间，正向放大、负向抑制
        g = 1.0 + (sim / self.sim_temperature).clamp(-6, 6).unsqueeze(-1)
        return hidden * g  # [B, L, d]

    def forward(
        self,
        hidden: torch.Tensor,
        target_meaning: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lm_head: Optional[nn.Module] = None,
        recon_lambda: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """训练前向。

        Args:
          hidden: [B, L, d] 主模型输出的 hidden states
          target_meaning: [B, d] MeaningPlanner.plan 的输出
          labels: [B, L] next-token labels（可选，用于算 LM loss）
          lm_head: 主模型 lm_head（若构造时未传）
          recon_lambda: 语义重建 loss 权重

        Returns:
          dict: logits, lm_loss (if labels), recon_loss, total_loss
        """
        head = lm_head if lm_head is not None else self.lm_head
        assert head is not None, "MeaningFirstDecoder 需要 lm_head"

        gated = self._similarity_gate(hidden, target_meaning)
        logits = head(gated)  # [B, L, V]

        out: Dict[str, torch.Tensor] = {"logits": logits, "gated_hidden": gated}

        if labels is not None:
            B, L, V = logits.shape
            # 对齐长度
            if labels.shape[1] != L:
                m = min(labels.shape[1], L)
                logits_lm = logits[:, :m, :]
                labels_lm = labels[:, :m]
            else:
                logits_lm, labels_lm = logits, labels
            lm_loss = F.cross_entropy(
                logits_lm.reshape(-1, V),
                labels_lm.reshape(-1),
                ignore_index=-100,
            )
            out["lm_loss"] = lm_loss

            # 语义重建 loss：从目标语义直接预测 token 分布，逼迫 meaning 有信息
            recon_logits = self.reconstruct_proj(target_meaning)  # [B, V]
            # 用 labels 的「整体 bag-of-token」分布做目标
            B2 = labels_lm.shape[0]
            target_dist = torch.zeros(B2, V, device=logits.device, dtype=logits.dtype)
            for b in range(B2):
                valid = labels_lm[b][labels_lm[b] != -100]
                if valid.numel() > 0:
                    target_dist[b].scatter_add_(
                        0, valid, torch.ones_like(valid, dtype=target_dist.dtype)
                    )
                    target_dist[b] = target_dist[b] / valid.numel()
            recon_loss = F.kl_div(
                F.log_softmax(recon_logits, dim=-1),
                target_dist,
                reduction="batchmean",
            )
            out["recon_loss"] = recon_loss
            out["total_loss"] = lm_loss + recon_lambda * recon_loss

        return out

    @torch.no_grad()
    def generate(
        self,
        hidden_proj: nn.Module,
        embed: nn.Embedding,
        target_meaning: torch.Tensor,
        max_len: int = 32,
        lm_head: Optional[nn.Module] = None,
        bos_token_id: int = 0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """纯按目标语义自回归生成（无 fallback）。

        为保持 Q1B「纯意义优先」，每步 logits 仍由目标语义调制。
        这是一个最小可跑的生成路径：不依赖 KV cache，逐 token 走。
        """
        head = lm_head if lm_head is not None else self.lm_head
        assert head is not None and hidden_proj is not None
        device = target_meaning.device
        B = target_meaning.shape[0]
        ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            emb = embed(ids)  # [B, T, d]
            hidden = hidden_proj(emb)  # 主模型前向（由调用方提供）
            gated = self._similarity_gate(hidden, target_meaning)
            logits = head(gated[:, -1:, :]).squeeze(1)  # [B, V]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                nxt = logits.argmax(dim=-1)
            ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)

        return ids  # [B, max_len+1]


@dataclass
class CollapseReport:
    collapsed: bool
    reason: str
    metrics: Dict[str, float]


class CollapseDetector:
    """检测解码坍塌：重复 token、低熵、固定序列。

    Q3B：只标记、不回退。返回 CollapseReport 供 CI/审计记录。
    """

    def __init__(
        self,
        repeat_threshold: float = 0.6,
        entropy_threshold: float = 0.5,
        min_unique_ratio: float = 0.15,
    ):
        self.repeat_threshold = repeat_threshold
        self.entropy_threshold = entropy_threshold
        self.min_unique_ratio = min_unique_ratio

    def check(self, generated_ids: torch.Tensor) -> CollapseReport:
        """generated_ids: [B, T] → CollapseReport"""
        ids = generated_ids.detach().cpu()
        B, T = ids.shape
        metrics: Dict[str, float] = {}
        reasons = []

        # 1. 重复率：相邻 token 相同的比例
        same = (ids[:, 1:] == ids[:, :-1]).float().mean().item()
        metrics["adjacent_repeat_rate"] = same
        if same > self.repeat_threshold:
            reasons.append(f"adjacent_repeat={same:.2f}")

        # 2. unique token 比例
        unique_ratio = float(len(torch.unique(ids)) / max(1, T))
        metrics["unique_ratio"] = unique_ratio
        if unique_ratio < self.min_unique_ratio:
            reasons.append(f"unique_ratio={unique_ratio:.2f}")

        # 3. token 分布熵（归一化）
        vals, counts = torch.unique(ids, return_counts=True)
        p = counts.float() / counts.sum()
        entropy = float(-(p * p.log()).sum().item())
        norm_entropy = entropy / max(1e-6, float(len(vals)))
        metrics["norm_entropy"] = norm_entropy
        if norm_entropy < self.entropy_threshold:
            reasons.append(f"norm_entropy={norm_entropy:.2f}")

        collapsed = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "ok"
        return CollapseReport(collapsed=collapsed, reason=reason, metrics=metrics)
