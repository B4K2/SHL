from dotenv import load_dotenv
from openai import OpenAI

from app.catalog import load_records
from app.config import get_settings
from app.embeddings import embed_texts
from app.vector_store import save_vectors

EMBEDDINGS_PATH = "data/catalog_embeddings.npz"


def main() -> None:
    load_dotenv()
    settings = get_settings()
    records = load_records(settings.catalog_path)
    print(f"Embedding {len(records)} records at {settings.embedding_dimensions} dims...")

    client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)
    vectors = embed_texts(
        client,
        settings.embedding_model,
        settings.embedding_dimensions,
        [r.embedding_text() for r in records],
    )

    save_vectors(EMBEDDINGS_PATH, [r.id for r in records], vectors)
    print(f"Saved {vectors.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
