"""ModelScope dataset loader for Luna training.

Default public dataset: AI-ModelScope/alpaca-gpt4-data-zh (from modelscope.cn/datasets).

ponytail: uses a deterministic hash tokenizer (mod vocab_size), not a real BPE.
Upgrade path: swap in a HuggingFace/ModelScope tokenizer matching the 550b vocab.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Curated defaults from https://modelscope.cn/datasets (verified via API).
DEFAULT_DATASET = "AI-ModelScope/alpaca-gpt4-data-zh"
FALLBACK_DATASETS = (
    "AI-ModelScope/alpaca-gpt4-data-zh",
    "AI-ModelScope/HC3-Chinese",
)
API = "https://www.modelscope.cn/api/v1"

TEXT_KEYS = (
    "text",
    "content",
    "instruction",
    "input",
    "output",
    "response",
    "answer",
    "question",
    "conversations",
)


def _row_to_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return str(row)

    parts: List[str] = []
    for key in TEXT_KEYS:
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if key == "conversations" and isinstance(val, list):
            for turn in val:
                if isinstance(turn, dict):
                    parts.append(str(turn.get("value") or turn.get("content") or ""))
                else:
                    parts.append(str(turn))
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())
    if parts:
        return "\n".join(parts)
    # last resort: join string fields
    return "\n".join(str(v) for v in row.values() if isinstance(v, str) and v.strip())


def hash_tokenize(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """Map UTF-8 text → token ids in [0, vocab_size) of length seq_len."""
    data = text.encode("utf-8", errors="ignore") or b" "
    ids: List[int] = []
    # 3-byte rolling windows hashed into vocab
    for i in range(0, max(1, len(data)), 2):
        chunk = data[i : i + 3]
        h = hashlib.blake2b(chunk, digest_size=4).digest()
        ids.append(int.from_bytes(h, "little") % vocab_size)
        if len(ids) >= seq_len:
            break
    if not ids:
        ids = [0]
    while len(ids) < seq_len:
        ids.extend(ids[: max(1, seq_len - len(ids))])
    return torch.tensor(ids[:seq_len], dtype=torch.long)


def _http_get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    """Fetch URL bytes; prefer curl for large ModelScope blobs (more reliable resume)."""
    curl = shutil.which("curl")
    if curl:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                curl, "-fsSL", "--retry", str(retries), "--retry-all-errors",
                "-C", "-", "-o", str(tmp_path), url,
            ]
            subprocess.run(cmd, check=True, timeout=max(timeout, 600))
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "luna-ultimate/0.2"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                chunks: List[bytes] = []
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                return b"".join(chunks)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("HTTP GET attempt %d failed: %s", attempt + 1, e)
    raise RuntimeError(f"HTTP GET failed after {retries} tries: {url}") from last_err


def _list_dataset_files(dataset_id: str) -> List[dict]:
    url = f"{API}/datasets/{dataset_id}/repo/tree?Revision=master&Recursive=true"
    payload = json.loads(_http_get(url).decode("utf-8"))
    return (payload.get("Data") or {}).get("Files") or []


def _parse_table_bytes(name: str, raw: bytes, max_samples: int) -> List[Any]:
    text = raw.decode("utf-8", errors="ignore")
    lower = name.lower()
    rows: List[Any] = []
    if lower.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in ("data", "train", "examples", "rows"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
            if not rows:
                rows = [data]
    elif lower.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    elif lower.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    else:
        return []
    if max_samples:
        rows = rows[:max_samples]
    return rows


def _cache_dir() -> Path:
    root = Path(os.environ.get("LUNA_DATASET_CACHE", ".cache/modelscope_datasets"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_via_http(dataset_id: str, max_samples: int) -> List[Any]:
    """Lightweight fallback: pull train.csv / json from ModelScope repo API."""
    files = _list_dataset_files(dataset_id)
    preferred = sorted(
        [f for f in files if f.get("Type") == "blob"],
        key=lambda f: (
            0 if str(f.get("Path", "")).endswith((".csv", ".json", ".jsonl")) else 1,
            -(f.get("Size") or 0),
        ),
    )
    cache_root = _cache_dir()
    for meta in preferred:
        path = meta["Path"]
        if not path.endswith((".csv", ".json", ".jsonl")):
            continue
        # skip tiny pointer stubs
        if (meta.get("Size") or 0) < 1024 and path.endswith(".json"):
            continue
        cache_file = cache_root / dataset_id.replace("/", "__") / Path(path).name
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists() and cache_file.stat().st_size > 1024:
            logger.info("Using cached %s", cache_file)
            raw = cache_file.read_bytes()
        else:
            url = f"{API}/datasets/{dataset_id}/repo?Revision=master&FilePath={path}"
            logger.info("HTTP fetch %s (%s bytes)", path, meta.get("Size"))
            raw = _http_get(url, timeout=300)
            cache_file.write_bytes(raw)
        rows = _parse_table_bytes(path, raw, max_samples)
        if rows:
            logger.info("Loaded %d rows from %s via HTTP", len(rows), path)
            return rows
    raise RuntimeError(f"No usable table files in {dataset_id}")


def _load_modelscope_rows(dataset_id: str, max_samples: int, split: str) -> List[Any]:
    # Prefer HTTP (few deps); fall back to MsDataset SDK when available.
    try:
        return _load_via_http(dataset_id, max_samples)
    except Exception as http_err:  # noqa: BLE001
        logger.warning("HTTP load failed for %s: %s; trying MsDataset", dataset_id, http_err)

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as e:
        raise ImportError(
            "Could not load dataset via HTTP and modelscope SDK is unavailable. "
            "pip install modelscope addict"
        ) from e

    logger.info("Loading ModelScope dataset %s (split=%s)", dataset_id, split)
    try:
        ds = MsDataset.load(dataset_id, split=split)
    except Exception:
        ds = MsDataset.load(dataset_id, split="train")

    rows: List[Any] = []
    for i, row in enumerate(ds):
        rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
        if i > 0 and i % 1000 == 0:
            logger.info("  loaded %d rows...", i)
    if not rows:
        raise RuntimeError(f"Dataset {dataset_id} produced 0 rows")
    logger.info("Loaded %d rows from %s", len(rows), dataset_id)
    return rows


class ModelScopeTextDataset(Dataset):
    """LM dataset built from a ModelScope hub dataset id."""

    def __init__(
        self,
        dataset_id: str,
        vocab_size: int,
        seq_len: int,
        max_samples: int = 10000,
        split: str = "train",
        rows: Optional[Sequence[Any]] = None,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_id = dataset_id
        if rows is None:
            rows = _load_modelscope_rows(dataset_id, max_samples, split)
        texts = [_row_to_text(r) for r in rows]
        self.texts = [t for t in texts if t and t.strip()]
        if not self.texts:
            raise RuntimeError(f"No text extracted from {dataset_id}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = hash_tokenize(self.texts[idx % len(self.texts)], self.vocab_size, self.seq_len)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


def build_dataset(
    dataset: Optional[str],
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    dummy_cls,
) -> Dataset:
    """Build ModelScope dataset, or fall back to DummyDataset when dataset is None/'dummy'."""
    if not dataset or dataset.lower() in {"dummy", "none", "random"}:
        return dummy_cls(vocab_size, seq_len, num_samples)

    tried: List[str] = []
    candidates: Iterable[str] = (dataset,)
    if dataset == "auto":
        candidates = FALLBACK_DATASETS

    last_err: Optional[Exception] = None
    for ds_id in candidates:
        tried.append(ds_id)
        try:
            return ModelScopeTextDataset(
                dataset_id=ds_id,
                vocab_size=vocab_size,
                seq_len=seq_len,
                max_samples=num_samples,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Failed to load %s: %s", ds_id, e)

    raise RuntimeError(f"Could not load any dataset from {tried}: {last_err}")
