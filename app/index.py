from pathlib import Path

import numpy as np
from openai import OpenAI

from app.catalog import CatalogRecord
from app.config import Settings
from app.embeddings import embed_texts
from app.vector_store import VectorStore, load_vectors, save_vectors

EMBEDDINGS_PATH = "data/catalog_embeddings.npz"


def _align_to_records(records: list[CatalogRecord], ids: list[str], vectors: np.ndarray) -> np.ndarray | None:
    if set(ids) != {r.id for r in records}:
        return None
    row_by_id = {record_id: row for row, record_id in enumerate(ids)}
    order = [row_by_id[r.id] for r in records]
    return vectors[order]


def build_vector_store(settings: Settings, records: list[CatalogRecord]) -> VectorStore:
    path = Path(EMBEDDINGS_PATH)
    if path.exists():
        ids, vectors = load_vectors(path)
        aligned = _align_to_records(records, ids, vectors)
        if aligned is not None:
            return VectorStore(records, aligned)

    client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)
    vectors = embed_texts(
        client,
        settings.embedding_model,
        settings.embedding_dimensions,
        [r.embedding_text() for r in records],
    )
    save_vectors(path, [r.id for r in records], vectors)
    return VectorStore(records, vectors)
