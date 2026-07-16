"""RAG 向量库接口（内存 dict 占位，预留 FAISS/Chroma）。

颞叶知识路径：丘脑在 knowledge/code 任务上设置 use_rag=True。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RAGDoc:
    doc_id: str
    text: str
    meta: Dict[str, object]


def _hash_embed(text: str, dim: int = 64) -> List[float]:
    """确定性伪嵌入：无外部依赖，便于本地冒烟。"""
    vec = [0.0] * dim
    tokens = (text or "").lower().split()
    if not tokens:
        return vec
    for i, tok in enumerate(tokens):
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        for j in range(dim):
            vec[j] += (h[j % len(h)] - 128) / 128.0
        # 位置扰动
        vec[i % dim] += 0.1
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class InMemoryRAG:
    """内存向量库占位。"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._docs: Dict[str, RAGDoc] = {}
        self._vecs: Dict[str, List[float]] = {}

    def add(self, text: str, doc_id: Optional[str] = None, **meta) -> str:
        did = doc_id or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self._docs[did] = RAGDoc(doc_id=did, text=text, meta=meta)
        self._vecs[did] = _hash_embed(text, self.dim)
        return did

    def search(self, query: str, k: int = 5) -> List[Tuple[RAGDoc, float]]:
        qv = _hash_embed(query, self.dim)
        scored = [(_cosine(qv, self._vecs[i]), self._docs[i]) for i in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(doc, score) for score, doc in scored[:k]]

    def __len__(self) -> int:
        return len(self._docs)


# 预留后端名；真正 FAISS/Chroma 接入时替换工厂
def build_rag(backend: str = "memory", **kwargs) -> InMemoryRAG:
    if backend != "memory":
        raise NotImplementedError(
            f"RAG backend {backend!r} 尚未接入；当前仅支持 memory 占位。"
        )
    return InMemoryRAG(**kwargs)
