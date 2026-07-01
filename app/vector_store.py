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

    def get(self, record_id: str) -> CatalogRecord | None:
        return self._by_id.get(record_id)

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


def save_vectors(path: str | Path, ids: list[str], vectors: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids, dtype=object), vectors=vectors)


def load_vectors(path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["vectors"].astype(np.float32)
