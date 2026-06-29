"""Qdrant vektör veritabanı bağlantısı ve koleksiyon yönetimi.

Dense (BGE-M3, 1024 dim) + sparse (BM25 hybrid) koleksiyon şeması.
Bağlantı bilgileri .env'den alınır.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Singleton Qdrant istemcisi döndürür.

    Raises:
        ConnectionError: Qdrant'a bağlanılamazsa.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=url)
        client.get_collections()  # bağlantıyı doğrula
        _client = client
        logger.info("Qdrant istemcisi oluşturuldu: %s", url)
        return _client
    except Exception as exc:
        logger.error("Qdrant bağlantısı kurulamadı (%s): %s", url, exc)
        raise ConnectionError(f"Qdrant bağlantısı başarısız: {exc}") from exc


def get_or_create_collection() -> None:
    """Koleksiyon yoksa dense + sparse hybrid şemada oluşturur."""
    client = get_client()
    collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.info("Koleksiyon zaten mevcut: %s", collection)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            ),
        },
    )
    logger.info("Koleksiyon oluşturuldu: %s", collection)


def test_connection() -> None:
    """Qdrant bağlantısını doğrular ve koleksiyonu hazırlar."""
    client = get_client()
    get_or_create_collection()
    collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
    info = client.get_collection(collection)
    point_count = info.points_count or 0
    logger.info(
        "Qdrant bağlantısı başarılı, koleksiyon hazır: %s (%d kayıt)",
        collection, point_count,
    )
    print("Qdrant bağlantısı başarılı, koleksiyon hazır")
