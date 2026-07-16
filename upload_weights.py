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
    parser.add_argument("--revision", type=str, default="master",
                        help="Git revision to push to (ModelScope default is often master)")
    parser.add_argument("--shard_size_gb", type=float, default=5.0,
                        help="Target shard size in GB (default: 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Prepare files without uploading")
    parser.add_argument("--message", type=str, default=None,
                        help="Commit message for the upload")
    return parser.parse_args()


def _prepare_weights_for_export(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop accidental nested CTM copies and clone shared storages for safetensors."""
    cleaned: Dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        if name.startswith("jepa_controller.ctm."):
            continue
        # meaning_decoder.lm_head 与 embed_tokens 权重绑定；clone 避免 safetensors 共享内存报错
        cleaned[name] = tensor.detach().cpu().contiguous().clone()
    return cleaned


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

    weights: Dict[str, torch.Tensor] = {}

    # Try safetensors first
    safe_files = sorted(path.glob("*.safetensors"))
    if safe_files:
        from safetensors.torch import load_file as safe_load
        for f in safe_files:
            weights.update(safe_load(str(f)))
        return _prepare_weights_for_export(weights)

    # Prefer .pt train checkpoints over intermediate .bin exports
    pt_files = list(path.glob("*.pt")) + list(path.glob("*.pth"))
    if pt_files:
        checkpoint = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return _prepare_weights_for_export(checkpoint["model_state_dict"])
            if "model" in checkpoint:
                return _prepare_weights_for_export(checkpoint["model"])
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return _prepare_weights_for_export(checkpoint)
            raise ValueError(
                f"Unrecognized checkpoint keys in {pt_files[0]}: {list(checkpoint.keys())[:12]}"
            )
        return _prepare_weights_for_export(checkpoint)

    # Try pytorch .bin files
    bin_files = sorted(path.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            loaded = torch.load(str(f), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                weights.update(loaded["model_state_dict"])
            else:
                weights.update(loaded)
        return _prepare_weights_for_export(weights)

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
        token = os.environ.get("MODELSCOPE_SDK_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if token is None:
        print("ERROR: No ModelScope token provided.")
        print("Set MODELSCOPE_TOKEN / MODELSCOPE_SDK_TOKEN or use --token flag.")
        print("Get your token at: https://modelscope.cn/my/myaccesstoken")
        sys.exit(1)

    os.environ["MODELSCOPE_SDK_TOKEN"] = token
    os.environ["MODELSCOPE_TOKEN"] = token

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("ERROR: modelscope not installed.")
        print("Install with: pip install modelscope")
        sys.exit(1)

    api = HubApi()
    try:
        api.login(token)
    except Exception as e:
        print(f"Warning: api.login failed ({e}); continuing with env token")

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
            license="Apache License 2.0",
        )
        print(f"Created repo: {full_repo}")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "409" in err or "exist" in err:
            print(f"Repo already exists: {full_repo}")
        else:
            print(f"Warning creating repo: {e}")

    # ModelScope 新建仓默认分支多为 master（不是 main）
    try:
        valid = api.get_valid_revision(full_repo)
        if valid:
            revision = valid
            print(f"  using revision: {revision}")
    except Exception:
        pass

    # Hub 兼容：同时提供 configuration.json
    local_path = Path(local_dir)
    cfg = local_path / "config.json"
    conf = local_path / "configuration.json"
    if cfg.exists() and not conf.exists():
        conf.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"  push_model from {local_dir} …")
    pushed = False
    try:
        api.push_model(
            model_id=full_repo,
            model_dir=local_dir,
            commit_message=message or f"Upload {repo_name} weights",
            revision=revision,
        )
        pushed = True
    except TypeError:
        api.push_model(
            model_id=full_repo,
            model_dir=local_dir,
            commit_message=message or f"Upload {repo_name} weights",
        )
        pushed = True
    except Exception as e:
        print(f"push_model failed ({e}); fallback upload_folder …")
        try:
            api.upload_folder(
                repo_id=full_repo,
                folder_path=local_dir,
                revision=revision,
                commit_message=message or f"Upload {repo_name} weights",
                token=token,
            )
            pushed = True
        except Exception as e2:
            print(f"ERROR: upload_folder also failed: {e2}")
            sys.exit(1)

    # 校验：远端应有权重文件
    try:
        try:
            remote_files = api.get_model_files(full_repo, revision=revision)
        except TypeError:
            remote_files = api.get_model_files(full_repo)
        names = {f.get("Path") or f.get("Name") for f in remote_files}
        has_weights = any(
            n.endswith(".safetensors") or n.endswith(".bin") for n in names if n
        )
        if not has_weights:
            print(f"ERROR: remote has no weight files: {sorted(names)[:20]}")
            sys.exit(1)
        print(f"  verified remote files: {len(names)} (weights present)")
    except Exception as e:
        print(f"WARNING: could not verify remote files: {e}")
        if not pushed:
            sys.exit(1)

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

    # Write README for ModelScope（与 GitHub 仓库名对齐）
    preset = config.get("preset_name", "unknown")
    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text(f"""---
license: Apache License 2.0
---
# Luna-Ultimate

GitHub: https://github.com/huangzhaoqing-jason/luna-ultimate

本仓库权重对应代码仓 `luna-ultimate`。当前上传 preset：`{preset}`（研究原型/烟测权重，不等于 550B 全量）。

## Quick Start
```python
from modelscope import snapshot_download
model_dir = snapshot_download("huang18928827157/luna-ultimate")
```
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