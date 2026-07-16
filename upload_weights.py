"""Upload Luna-Ultimate 550B weights to ModelScope.

Usage:
    python upload_weights.py \
        --model_path ./checkpoints/final \
        --repo_name luna-ultimate-550b \
        --token YOUR_MODELSCOPE_TOKEN

Requirements:
    pip install modelscope

The script:
    1. Loads model weights from a checkpoint directory
    2. Converts to safetensors format (sharded, ~5GB per shard)
    3. Saves model config alongside weights
    4. Uploads everything to ModelScope Hub
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn

# Check for safetensors (optional but recommended)
try:
    from safetensors.torch import save_file as safe_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not installed. Will use .bin format.")
    print("Install with: pip install safetensors")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Luna-Ultimate weights to ModelScope")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--repo_name", type=str, default="luna-ultimate-550b",
                        help="ModelScope repo name")
    parser.add_argument("--namespace", type=str, default="huang18928827157",
                        help="ModelScope namespace (default: your username)")
    parser.add_argument("--token", type=str, default=None,
                        help="ModelScope SDK token (or set MODELSCOPE_SDK_TOKEN env var)")
    parser.add_argument("--revision", type=str, default="main",
                        help="Git revision to push to")
    parser.add_argument("--shard_size_gb", type=float, default=5.0,
                        help="Target shard size in GB (default: 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Prepare files without uploading")
    parser.add_argument("--message", type=str, default=None,
                        help="Commit message for the upload")
    return parser.parse_args()


def load_model_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from checkpoint directory.

    Supports:
        - pytorch_model.bin (single file)
        - pytorch_model-00001-of-XXXXX.bin (sharded)
        - model.safetensors (single safetensors)
        - model-00001-of-XXXXX.safetensors (sharded safetensors)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    weights = {}

    # Try safetensors first
    safe_files = sorted(path.glob("*.safetensors"))
    if safe_files:
        from safetensors.torch import load_file as safe_load
        for f in safe_files:
            weights.update(safe_load(str(f)))
        return weights

    # Try pytorch .bin files
    bin_files = sorted(path.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            weights.update(torch.load(str(f), map_location="cpu"))
        return weights

    # Try .pt checkpoint (train.py / smoke_final.pt)
    pt_files = list(path.glob("*.pt")) + list(path.glob("*.pth"))
    if pt_files:
        checkpoint = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if "model" in checkpoint:
                return checkpoint["model"]
            # flat state_dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        if isinstance(checkpoint, dict):
            raise ValueError(
                f"Unrecognized checkpoint keys in {pt_files[0]}: {list(checkpoint.keys())[:12]}"
            )
        return checkpoint

    raise FileNotFoundError(f"No weight files found in {model_path}")


def save_sharded_safetensors(
    weights: Dict[str, torch.Tensor],
    output_dir: str,
    config: dict,
    shard_size_gb: float = 5.0,
) -> List[str]:
    """Save weights as sharded safetensors with index."""
    if not HAS_SAFETENSORS:
        return save_sharded_bin(weights, output_dir, config, shard_size_gb)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate total size
    total_params = sum(w.numel() for w in weights.values())
    total_bytes = sum(w.numel() * w.element_size() for w in weights.values())
    total_gb = total_bytes / (1024 ** 3)
    shard_bytes = int(shard_size_gb * (1024 ** 3))

    print(f"Total weights: {total_params:,} parameters, {total_gb:.2f} GB")
    print(f"Target shard size: {shard_size_gb:.1f} GB")

    # Assign weights to shards
    shards = []
    current_shard = {}
    current_size = 0
    shard_index = 0

    for name, tensor in sorted(weights.items()):
        t_size = tensor.numel() * tensor.element_size()

        # Large tensors get their own shard
        if t_size > shard_bytes:
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            shards.append({name: tensor})
            continue

        if current_size + t_size > shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[name] = tensor
        current_size += t_size

    if current_shard:
        shards.append(current_shard)

    # Save shards
    weight_map = {}
    saved_files = []

    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = output_path / shard_name
        safe_save(shard, str(shard_path))
        saved_files.append(shard_name)

        for name in shard:
            weight_map[name] = shard_name

        shard_size = sum(w.numel() * w.element_size() for w in shard.values())
        print(f"  [{i + 1}/{len(shards)}] {shard_name}: "
              f"{shard_size / (1024 ** 3):.2f} GB, "
              f"{len(shard)} tensors")

    # Save config.json
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save model index
    index = {
        "metadata": {
            "total_size": total_bytes,
            "shard_count": len(shards),
        },
        "weight_map": weight_map,
    }
    index_path = output_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved {len(shards)} shards to {output_dir}")
    return saved_files


def save_sharded_bin(
    weights: Dict[str, torch.Tensor],
    output_dir: str,
    config: dict,
    shard_size_gb: float = 5.0,
) -> List[str]:
    """Fallback: save as sharded .bin files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_bytes = int(shard_size_gb * (1024 ** 3))

    shards = []
    current_shard = {}
    current_size = 0

    for name, tensor in sorted(weights.items()):
        t_size = tensor.numel() * tensor.element_size()
        if current_size + t_size > shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += t_size

    if current_shard:
        shards.append(current_shard)

    saved_files = []
    for i, shard in enumerate(shards):
        shard_name = f"pytorch_model-{i + 1:05d}-of-{len(shards):05d}.bin"
        torch.save(shard, output_path / shard_name)
        saved_files.append(shard_name)
        print(f"  [{i + 1}/{len(shards)}] {shard_name}")

    # Save config.json
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return saved_files


def get_model_config(model_path: Optional[str] = None) -> dict:
    """Build model config dict for ModelScope.

    Prefer ``config.json`` next to weights (e.g. tiny champion); else LunaConfig defaults.
    """
    from config import LunaConfig

    if model_path:
        cfg_file = Path(model_path) / "config.json"
        if cfg_file.exists():
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            raw.setdefault("model_type", "luna_ultimate")
            raw.setdefault("architectures", ["LunaUltimateFused"])
            return raw

    config = LunaConfig()
    return {
        "model_type": "luna_ultimate",
        "architectures": ["LunaUltimateFused"],
        "preset_name": config.preset_name,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "mamba2_layers": config.mamba2_layers,
        "mla_layers": config.mla_layers,
        "n_heads": config.n_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": True,
        # MoE
        "num_routed_experts": config.num_routed_experts,
        "num_shared_experts": config.num_shared_experts,
        "num_expert_activated": config.num_expert_activated,
        # Mamba2
        "mamba_d_state": config.mamba_d_state,
        "mamba_d_conv": config.mamba_d_conv,
        "mamba_expand": config.mamba_expand,
        # MLA
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        # CTM / JEPA 总控
        "ctm_n_neurons": config.ctm_n_neurons,
        "ctm_nlm_hidden": config.ctm_nlm_hidden,
        "ctm_max_ticks": config.ctm_max_ticks,
        "ctm_jepa_enabled": config.ctm_jepa_enabled,
        "ctm_jepa_horizon": config.ctm_jepa_horizon,
        "ctm_jepa_ema_decay": config.ctm_jepa_ema_decay,
        "jepa_control_enabled": config.jepa_control_enabled,
        "route_mode": config.route_mode,
        "vjepa_config": config.vjepa_config,
    }


def upload_to_modelscope(
    local_dir: str,
    repo_name: str,
    namespace: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = "main",
    message: Optional[str] = None,
):
    """Upload files to ModelScope Hub."""
    if token is None:
        token = os.environ.get("MODELSCOPE_SDK_TOKEN")
    if token is None:
        print("ERROR: No ModelScope token provided.")
        print("Set MODELSCOPE_SDK_TOKEN env variable or use --token flag.")
        print("Get your token at: https://modelscope.cn/my/myaccesstoken")
        sys.exit(1)

    os.environ["MODELSCOPE_SDK_TOKEN"] = token

    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.repository import Repository
    except ImportError:
        print("ERROR: modelscope not installed.")
        print("Install with: pip install modelscope")
        sys.exit(1)

    api = HubApi()

    if namespace is None:
        user_info = api.get_user_info()
        namespace = user_info.get("username", user_info.get("login", "unknown"))
        print(f"Using namespace: {namespace}")

    full_repo = f"{namespace}/{repo_name}"
    print(f"\nUploading to ModelScope: {full_repo}")

    # Create repo if it doesn't exist
    try:
        api.create_model(
            model_id=full_repo,
            visibility="public",
            license="Apache 2.0",
        )
        print(f"Created repo: {full_repo}")
    except Exception as e:
        if "already exists" in str(e) or "409" in str(e):
            print(f"Repo already exists: {full_repo}")
        else:
            print(f"Warning: {e}")

    # Upload files
    local_path = Path(local_dir)
    files = sorted(local_path.glob("**/*"))
    files = [f for f in files if f.is_file()]

    total = sum(f.stat().st_size for f in files)
    uploaded = 0

    for f in files:
        rel_path = f.relative_to(local_path)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  Uploading {rel_path} ({size_mb:.1f} MB)...")

        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=str(rel_path),
            model_id=full_repo,
            revision=revision,
            message=message,
        )
        uploaded += f.stat().st_size
        print(f"    Progress: {uploaded / (1024 ** 3):.1f} / {total / (1024 ** 3):.1f} GB")

    print(f"\nUpload complete: {full_repo}")
    print(f"View at: https://modelscope.cn/models/{full_repo}")


def prepare_tokenizer(model_path: str, output_dir: str):
    """Copy tokenizer files if they exist."""
    src = Path(model_path)
    dst = Path(output_dir)

    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for fname in tokenizer_files:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)
            print(f"  Copied: {fname}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Luna-Ultimate Weight Uploader (ModelScope)")
    print("=" * 60)

    # Step 1: Load weights
    print(f"\n[1/4] Loading weights from: {args.model_path}")
    weights = load_model_weights(args.model_path)
    total_params = sum(w.numel() for w in weights.values())
    total_gb = sum(w.numel() * w.element_size() for w in weights.values()) / (1024 ** 3)
    print(f"  Loaded: {total_params:,} parameters, {total_gb:.2f} GB")

    # Step 2: Prepare config
    print("\n[2/4] Building model config...")
    config = get_model_config(args.model_path)
    print(f"  Model type: {config.get('model_type')}")
    print(f"  Hidden size: {config.get('hidden_size')}")
    print(f"  Layers: {config.get('num_hidden_layers')}")
    print(f"  Preset: {config.get('preset_name', 'unknown')}")

    # Step 3: Save sharded weights
    output_dir = "/tmp/luna_upload"
    print(f"\n[3/4] Saving sharded weights to: {output_dir}")
    shard_size = min(args.shard_size_gb, 4.9)  # ModelScope has ~5GB limit
    saved = save_sharded_safetensors(weights, output_dir, config, shard_size)

    # Copy tokenizer
    prepare_tokenizer(args.model_path, output_dir)

    # Write README for ModelScope
    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text("""# Luna-Ultimate 550B

The world's first CTM × Mamba2-SSD × MLA × FlashMoE hybrid architecture.

550B total / 80B active parameters.

## Architecture
- **Layers 1-12**: Mamba2-SSD (no KV Cache)
- **Layers 13-32**: MLA (KV compression to 1024-dim)
- **All 32**: FlashMoE (48 routed + 2 shared experts, Top-4)
- **Global**: CTM (4096-neuron recurrent, 1-4 adaptive ticks)

## Quick Start
```python
from modelscope import snapshot_download
from config import LunaConfig
from modeling_luna_ultimate import LunaUltimateFused

# Download weights
model_dir = snapshot_download("NAMESPACE/luna-ultimate-550b")

# Load model
config = LunaConfig()
model = LunaUltimateFused(config)
model.load_state_dict_from_safetensors(model_dir)
```

GitHub: https://github.com/huangzhaoqing-jason/luna-ultimate
""")

    # Step 4: Upload
    if args.dry_run:
        print(f"\n[4/4] DRY RUN — files prepared in: {output_dir}")
        print("Upload skipped. Use --no-dry_run to upload.")
    else:
        print(f"\n[4/4] Uploading to ModelScope...")
        upload_to_modelscope(
            local_dir=output_dir,
            repo_name=args.repo_name,
            namespace=args.namespace,
            token=args.token,
            revision=args.revision,
            message=args.message,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()