#!/usr/bin/env python3
"""Assert-based self-check for ModelScope dataset loader."""

from __future__ import annotations

from data.modelscope_dataset import ModelScopeTextDataset, _row_to_text, hash_tokenize


def main() -> None:
    ids = hash_tokenize("你好世界 hello", vocab_size=4096, seq_len=32)
    assert ids.shape == (32,)
    assert int(ids.min()) >= 0 and int(ids.max()) < 4096

    text = _row_to_text({"instruction": "问", "input": "", "output": "答"})
    assert "问" in text and "答" in text

    ds = ModelScopeTextDataset(
        "AI-ModelScope/alpaca-gpt4-data-zh",
        vocab_size=4096,
        seq_len=64,
        max_samples=4,
    )
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["input_ids"].shape[0] == 63
    assert sample["labels"].shape[0] == 63
    print("OK", len(ds), int(sample["input_ids"][0]))


if __name__ == "__main__":
    main()
