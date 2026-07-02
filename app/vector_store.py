import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from app.catalog import CatalogRecord

Predicate = Callable[[CatalogRecord], bool]


class VectorStore:
    def __init__(self, records: list[CatalogRecord], vectors: np.ndarray) -> None:
        if len(records) != vectors.shape[0]:
            raise ValueError("records and vectors length mismatch")
        self._records = records
        self._by_id = {record.id: record for record in records}
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._name_token_index = _build_rare_token_index(records)

    def get(self, record_id: str) -> CatalogRecord | None:
        return self._by_id.get(record_id)

    def match_name(self, text: str, limit: int = 5) -> list[CatalogRecord]:
        hit_counts: dict[str, int] = defaultdict(int)
        for token in _tokens(text):
            for record_id in self._name_token_index.get(token, ()):
                hit_counts[record_id] += 1
        ranked = sorted(hit_counts.items(), key=lambda pair: pair[1], reverse=True)
        return [self._by_id[record_id] for record_id, _ in ranked[:limit]]

    @property
    def dimension(self) -> int:
        return self._index.d

    def __len__(self) -> int:
        return len(self._records)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        predicate: Predicate | None = None,
    ) -> list[tuple[CatalogRecord, float]]:
        query = query_vector.reshape(1, -1).astype(np.float32)
        depth = len(self._records) if predicate else min(k, len(self._records))
        scores, indices = self._index.search(query, depth)

        results: list[tuple[CatalogRecord, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx == -1:
                continue
            record = self._records[idx]
            if predicate and not predicate(record):
                continue
            results.append((record, float(score)))
            if len(results) >= k:
                break
        return results


_RARE_TOKEN_MAX_RECORDS = 5
_STOPWORDS = {"the", "and", "for", "with", "new", "out", "what"}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _build_rare_token_index(records: list[CatalogRecord]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for token in _tokens(record.name):
            index[token].append(record.id)
    return {
        token: record_ids
        for token, record_ids in index.items()
        if len(record_ids) <= _RARE_TOKEN_MAX_RECORDS
    }


def save_vectors(path: str | Path, ids: list[str], vectors: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids, dtype=object), vectors=vectors)


def load_vectors(path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["vectors"].astype(np.float32)
