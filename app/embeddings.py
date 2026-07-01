import time

import numpy as np
from openai import AsyncOpenAI, OpenAI, RateLimitError


def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _embed_batch_with_retry(
    client: OpenAI,
    model: str,
    dimensions: int,
    batch: list[str],
    max_retries: int = 6,
) -> list[list[float]]:
    delay = 10.0
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=batch, dimensions=dimensions)
            return [item.embedding for item in response.data]
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 65.0)
    raise RuntimeError("unreachable")


def embed_texts(
    client: OpenAI,
    model: str,
    dimensions: int,
    texts: list[str],
    batch_size: int = 90,
    pause_seconds: float = 61.0,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_batch_with_retry(client, model, dimensions, batch))
        if start + batch_size < len(texts):
            time.sleep(pause_seconds)
    return _to_matrix(vectors)


async def embed_query(
    client: AsyncOpenAI,
    model: str,
    dimensions: int,
    text: str,
) -> np.ndarray:
    response = await client.embeddings.create(model=model, input=[text], dimensions=dimensions)
    return _to_matrix([response.data[0].embedding])[0]
