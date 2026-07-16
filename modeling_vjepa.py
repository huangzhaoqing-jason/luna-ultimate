"""V-JEPA (Visual Joint Embedding Predictive Architecture) for Luna-Ultimate.

Implements a ViT-based visual encoder with spatiotemporal patch masking and a
predictor that reconstructs masked patch features from visible context.

Pipeline:
  Input: (B, C, T, H, W) video or (B, C, H, W) image
  → Patchify: split into spatiotemporal patches
  → Embed: project patches to vision_dim
  → Mask: randomly mask a subset of patches
  → Encoder: process visible patches through ViT blocks
  → Predictor: predict masked patch features from visible context
  → Loss: cosine similarity between predicted and target features

Reference: V-JEPA (Meta, 2024) — Revisited and Extracted for Luna-Ultimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video: (B, C, T, H, W) → (B, N, D).

    Splits input into spatiotemporal patches of size (t_patch, h_patch, w_patch).
    For images, set T=1 and t_patch=1.

    Args:
        img_size: (H, W) of frames.
        patch_size: (T, H, W) patch dimensions.
        in_channels: Input channels (3 for RGB).
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.t_patch, self.h_patch, self.w_patch = patch_size
        self.grid_t = 1  # Default for images; overridden for video
        self.grid_h = img_size[0] // self.h_patch
        self.grid_w = img_size[1] // self.w_patch

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Patchify input.

        Input:  [B, C, T, H, W]
        Output: [B, N, embed_dim], (grid_t, grid_h, grid_w)
        """
        B, C, T, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, grid_t, grid_h, grid_w]
        grid_t, grid_h, grid_w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x, (grid_t, grid_h, grid_w)


class ViTBlock(nn.Module):
    """Standard ViT transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B, N, D]"""
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based Vision Encoder for V-JEPA.

    Processes visible (unmasked) patches through transformer blocks.
    Accepts [CLS] token prepended patches.

    Args:
        embed_dim: Patch embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim: int = 1024, depth: int = 24, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # Will be created dynamically
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _get_pos_embed(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Get or create positional embeddings."""
        if self.pos_embed is None or self.pos_embed.shape[1] < num_patches + 1:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches + 1, self.embed_dim, device=device) * 0.02
            )
        return self.pos_embed[:, :num_patches + 1, :]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode visible patches.

        Input:  [B, N, embed_dim] — patch embeddings
        Output: [B, N, embed_dim] — encoded features
        """
        B, N, D = x.shape
        pos_embed = self._get_pos_embed(N, x.device)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, D]
        x = x + pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Remove CLS token, return patch features
        return x[:, 1:, :]  # [B, N, D]


class VisionPredictor(nn.Module):
    """Predictor for V-JEPA: predicts masked patch features from visible context.

    Takes encoder output (visible patches) and mask tokens, predicts features
    for all positions including masked ones.

    Args:
        embed_dim: Feature dimension.
        predictor_depth: Number of predictor transformer blocks (shallower than encoder).
    """

    def __init__(self, embed_dim: int = 1024, predictor_depth: int = 6, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(predictor_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.predictor_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Predict features for all patches including masked ones.

        Args:
            encoder_output: [B, N_visible, embed_dim] — visible patch features
            mask: [B, N_total] — boolean mask (True = masked)
            ids_restore: [B, N_total] — indices to restore original order

        Returns:
            predictions: [B, N_total, embed_dim] — predicted features for all patches
        """
        B, N_visible, D = encoder_output.shape
        N_total = mask.shape[1]
        N_masked = mask.sum(dim=1).max().item()

        # Create full sequence with mask tokens
        # Place encoder output at visible positions, mask tokens at masked positions
        full = torch.zeros(B, N_total, D, device=encoder_output.device, dtype=encoder_output.dtype)
        full[:, ~mask, :] = encoder_output
        full[:, mask, :] = self.mask_token.expand(B, N_masked, -1)[:, :N_masked, :]

        # Restore original order
        batch_indices = torch.arange(B, device=encoder_output.device).unsqueeze(1)
        full = full[batch_indices, ids_restore]

        # Apply predictor blocks
        for block in self.blocks:
            full = block(full)

        full = self.norm(full)
        return self.predictor_proj(full)


class VJEPA(nn.Module):
    """V-JEPA: Visual Joint Embedding Predictive Architecture.

    Full pipeline:
      1. Patchify input video/image
      2. Apply random block-wise masking
      3. Encode visible patches with ViT
      4. Predict masked patches via Predictor
      5. Compute cosine similarity loss against target encoder features

    Args:
        img_size: Frame dimensions (H, W).
        patch_size: Spatiotemporal patch size (T, H, W).
        in_channels: Input channels.
        embed_dim: Vision embedding dimension.
        encoder_depth: ViT encoder depth.
        predictor_depth: Predictor depth.
        num_heads: Attention heads.
        mask_ratio: Fraction of patches to mask (0.0-1.0).
        use_target_encoder: Whether to use EMA target encoder.
        ema_decay: EMA decay rate for target encoder.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 1024,
        encoder_depth: int = 24,
        predictor_depth: int = 6,
        num_heads: int = 16,
        mask_ratio: float = 0.75,
        use_target_encoder: bool = True,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.use_target_encoder = use_target_encoder
        self.ema_decay = ema_decay

        # Patch embedding
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_channels, embed_dim)

        # Context encoder (processes visible patches)
        self.context_encoder = VisionEncoder(embed_dim, encoder_depth, num_heads)

        # Target encoder (EMA-updated, processes all patches for target)
        if use_target_encoder:
            self.target_encoder = VisionEncoder(embed_dim, encoder_depth, num_heads)
            self._init_target_encoder()
        else:
            self.target_encoder = None

        # Predictor
        self.predictor = VisionPredictor(embed_dim, predictor_depth, num_heads)

        # Projection from vision_dim to model's hidden_size for fusion
        self.projector = nn.Linear(embed_dim, 8192)  # 8192 = Luna d_model

    def _init_target_encoder(self):
        """Initialize target encoder with same weights as context encoder."""
        if self.target_encoder is not None:
            for target_param, context_param in zip(
                self.target_encoder.parameters(), self.context_encoder.parameters()
            ):
                target_param.data.copy_(context_param.data)
                target_param.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        """EMA update of target encoder."""
        if self.target_encoder is not None:
            for target_param, context_param in zip(
                self.target_encoder.parameters(), self.context_encoder.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(
                    context_param.data, alpha=1.0 - self.ema_decay
                )

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random block-wise masking to patches.

        Uses block-wise masking for better spatial coherence
        (blocks of adjacent patches are masked together).

        Args:
            x: [B, N, D] — patch embeddings
            mask_ratio: Fraction to mask

        Returns:
            x_masked: [B, N_visible, D] — only visible patches
            mask: [B, N] — boolean mask (True = masked)
            ids_restore: [B, N] — indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1.0 - mask_ratio))

        # Generate random noise for sorting
        noise = torch.rand(B, N, device=x.device)  # [B, N]

        # Sort noise to get shuffling indices
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, N]
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, N]

        # Keep the first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, len_keep, D]

        # Generate boolean mask
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        # Unshuffle mask to match original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V-JEPA forward pass.

        Input:  [B, C, T, H, W] or [B, C, H, W]
        Returns:
            vision_features: [B, N, d_model=8192] — projected features for fusion
            vjepa_loss: scalar — cosine similarity prediction loss
            predictions: [B, N, embed_dim] — predicted features
        """
        # Handle 4D image input (B, C, H, W) → (B, C, 1, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)

        # Patchify: [B, C, T, H, W] → [B, N, embed_dim]
        patches, grid = self.patch_embed(x)
        B, N, D = patches.shape

        # Apply masking
        if self.training:
            visible_patches, mask, ids_restore = self._random_masking(
                patches, self.mask_ratio
            )
        else:
            # During inference, process all patches
            visible_patches = patches
            mask = torch.zeros(B, N, device=patches.device, dtype=torch.bool)
            ids_restore = torch.arange(N, device=patches.device).unsqueeze(0).expand(B, -1)

        # Context encoder: process visible patches
        # [B, N_visible, D] → [B, N_visible, D]
        context_features = self.context_encoder(visible_patches, mask)

        # Predictor: predict all patches including masked
        # [B, N_visible, D] → [B, N, D]
        predictions = self.predictor(context_features, mask, ids_restore)

        # Target: if using EMA target encoder, compute target features
        vjepa_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.target_encoder is not None:
            with torch.no_grad():
                # Target encoder processes all patches (no masking)
                target_features = self.target_encoder(patches)  # [B, N, D]

            # Compute prediction loss only on masked patches
            masked_pred = predictions[mask]  # [N_masked, D]
            masked_target = target_features[mask]  # [N_masked, D]

            if masked_pred.shape[0] > 0:
                # Cosine similarity loss
                pred_norm = F.normalize(masked_pred, dim=-1)
                target_norm = F.normalize(masked_target, dim=-1)
                vjepa_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

            # Update target encoder via EMA
            self._update_target_encoder()

        # Project to model dimension for fusion
        # [B, N, embed_dim] → [B, N, d_model=8192]
        vision_features = self.projector(predictions)

        return vision_features, vjepa_loss, predictions

    def get_num_patches(self, frames: int = 1) -> int:
        """Get number of patches for given number of frames."""
        grid_t = max(1, frames // self.patch_embed.t_patch)
        return grid_t * self.patch_embed.grid_h * self.patch_embed.grid_w