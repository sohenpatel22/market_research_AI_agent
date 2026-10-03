"""Local (non-API) text embedding via a Hugging Face sentence-transformer"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from market_research_agent.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text chunks, returning one vector per input string"""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]
