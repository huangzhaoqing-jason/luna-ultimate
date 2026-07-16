"""Download Luna-Ultimate 550B weights from ModelScope.

Usage:
    python download_weights.py \
        --repo huangzhaoqing-jason/luna-ultimate-550b \
        --output ./checkpoints

    # Or with specific revision
    python download_weights.py \
        --repo huangzhaoqing-jason/luna-ultimate-550b \
        --revision v1.0 \
        --output ./checkpoints

Requirements:
    pip install modelscope
"""

import os
import sys
import argparse
from pathlib import Path
import json
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Luna-Ultimate weights from ModelScope"
    )
    parser.add_argument(
        "--repo", type=str,
        default="huang18928827157/luna-ultimate-550b",
        help="ModelScope repo name (namespace/model_name)"
    )
    parser.add_argument(
        "--output", type=str, default="./checkpoints",
        help="Output directory for downloaded weights"
    )
    parser.add_argument(
        "--revision", type=str, default="master",
        help="Model revision to download"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="ModelScope cache directory"
    )
    return parser.parse_args()


def download_from_modelscope(
    repo: str,
    output_dir: str,
    revision: str = "master",
    cache_dir: str = None,
):
    """Download model weights from ModelScope Hub."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope not installed.")
        print("Install with: pip install modelscope")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo} (revision: {revision})...")
    print(f"Output: {output_path.resolve()}")

    local_path = snapshot_download(
        repo,
        revision=revision,
        cache_dir=cache_dir or str(output_path / ".cache"),
        local_dir=str(output_path),
    )

    print(f"\nDownloaded to: {local_path}")

    # List downloaded files
    files = sorted(Path(local_path).glob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\nFiles ({total / (1024**3):.2f} GB total):")
    for f in files:
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")

    return local_path


def load_model_from_checkpoint(checkpoint_dir: str):
    """Load Luna-Ultimate model from downloaded checkpoint."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config import LunaConfig
    from modeling_luna_ultimate import LunaUltimateFused

    ckpt_path = Path(checkpoint_dir)

    # Load config
    config_path = ckpt_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = LunaConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        config = LunaConfig()

    print(f"\nInitializing model ({config.hidden_size}d, {config.num_hidden_layers} layers)...")
    model = LunaUltimateFused(config)

    # Load weights
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        from safetensors.torch import load_file as safe_load

        shard_files = set(index["weight_map"].values())
        for shard_file in sorted(shard_files):
            shard_path = ckpt_path / shard_file
            print(f"  Loading {shard_file}...")
            weights = safe_load(str(shard_path))
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
    else:
        # Try single safetensors
        safe_path = ckpt_path / "model.safetensors"
        if safe_path.exists():
            from safetensors.torch import load_file as safe_load
            weights = safe_load(str(safe_path))
            model.load_state_dict(weights, strict=False)
        else:
            # Try pytorch .bin
            bin_files = sorted(ckpt_path.glob("pytorch_model*.bin"))
            if bin_files:
                for f in bin_files:
                    weights = torch.load(str(f), map_location="cpu")
                    model.load_state_dict(weights, strict=False)
            else:
                print("WARNING: No weight files found in checkpoint directory.")

    print("Model loaded successfully.")
    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("Luna-Ultimate Weight Downloader (ModelScope)")
    print("=" * 60)

    local_path = download_from_modelscope(
        repo=args.repo,
        output_dir=args.output,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    print(f"\nTo load the model:")
    print(f"  from config import LunaConfig")
    print(f"  from modeling_luna_ultimate import LunaUltimateFused")
    print(f"  model = LunaUltimateFused(LunaConfig())")
    print(f"  # Load weights from: {local_path}")


if __name__ == "__main__":
    main()