"""Embedding helpers for semantic search.

Uses sentence-transformers all-MiniLM-L6-v2 when available.
Falls back gracefully — is_available() returns False if the library
is not installed, and all public functions become no-ops or raise clear errors.
"""

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None
_available: bool | None = None

# all-MiniLM-L6-v2 produces 384-dimensional float32 embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def is_available() -> bool:
    """Check whether sentence-transformers is importable."""
    global _available
    if _available is None:
        try:
            import sentence_transformers  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _get_model() -> "SentenceTransformer":
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        if not is_available():
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install 'corc[search]'"
            )
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts into embedding vectors.

    Returns a list of float lists, one per input text.
    Each vector has EMBEDDING_DIM dimensions.
    """
    model = _get_model()
    # model.encode returns numpy array of shape (n, dim)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def encode_single(text: str) -> list[float]:
    """Encode a single text into an embedding vector."""
    return encode([text])[0]


# --- Serialization: float32 list <-> blob ---

def embedding_to_blob(embedding: list[float]) -> bytes:
    """Serialize a float32 embedding vector to a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def blob_to_embedding(blob: bytes) -> list[float]:
    """Deserialize a binary blob back to a float32 embedding vector."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses pure Python to avoid requiring numpy at query time
    when embeddings are pre-computed.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def reset() -> None:
    """Reset the module state. Useful for testing."""
    global _model, _available
    _model = None
    _available = None
