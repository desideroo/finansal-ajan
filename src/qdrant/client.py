"""Qdrant vektör veritabanı bağlantısı ve koleksiyon yönetimi.

Koleksiyon yoksa otomatik oluşturur: BGE-M3 dense (1024 dim) + sparse hybrid.
Bağlantı bilgileri .env'den alınır.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Singleton Qdrant istemcisi döndürür."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _client = QdrantClient(url=url)
        logger.info("Qdrant istemcisi oluşturuldu: %s", url)
    return _client


def ensure_collection() -> None:
    """Koleksiyon yoksa BGE-M3 uyumlu şemada oluşturur."""
    client = get_client()
    collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.info("Koleksiyon zaten mevcut: %s", collection)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    logger.info("Koleksiyon oluşturuldu: %s", collection)


def test_connection() -> bool:
    """Qdrant bağlantısını doğrular ve koleksiyonu hazırlar."""
    try:
        client = get_client()
        info = client.get_collections()
        logger.info("Qdrant bağlantısı başarılı — %d koleksiyon mevcut", len(info.collections))
        ensure_collection()
        print("Qdrant bağlantısı başarılı")
        return True
    except Exception as exc:
        logger.error("Qdrant bağlantısı başarısız: %s", exc)
        print(f"Qdrant bağlantısı başarısız: {exc}")
        return False
