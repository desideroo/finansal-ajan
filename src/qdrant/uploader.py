"""Qdrant'a finansal sinyal verisi yükleme modülü.

BGE-M3 ile dense + sparse embedding üretir, hybrid upsert yapar.
"""

import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.http.models import PointStruct, SparseVector

from src.qdrant.client import get_client
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_model: BGEM3FlagModel | None = None


def get_embedding_model() -> BGEM3FlagModel:
    """Singleton BGE-M3 embedding modeli döndürür."""
    global _model
    if _model is None:
        logger.info("BGE-M3 modeli yükleniyor: BAAI/bge-m3")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        logger.info("BGE-M3 modeli yüklendi")
    return _model


def embed_text(text: str) -> dict:
    """Metni dense + sparse vektöre dönüştürür.

    Args:
        text: Embed edilecek metin.

    Returns:
        {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
    """
    model = get_embedding_model()
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec = output["dense_vecs"][0].tolist()
    lexical = output["lexical_weights"][0]
    return {
        "dense": dense_vec,
        "sparse": {
            "indices": [int(k) for k in lexical.keys()],
            "values": [float(v) for v in lexical.values()],
        },
    }


def upload_signal(signal: dict, chunk: dict, video_title: str) -> bool:
    """Tek bir sinyali Qdrant'a yükler.

    Args:
        signal: analyst.py çıktısındaki tek sinyal dict'i.
        chunk: Sinyalin geldiği chunk (chunk_id, start_sec için).
        video_title: Kaynak video/ses dosyası başlığı.

    Returns:
        True yüklendi, False atlandı/hata.
    """
    hisse = signal.get("hisse", "belirsiz")
    if hisse == "belirsiz":
        logger.info("Belirsiz hisse atlandı (chunk %s)", chunk.get("chunk_id"))
        return False

    embed_query = f"{hisse} {signal.get('sinyal_tipi', '')} {signal.get('gerekce', '')}"
    try:
        embed = embed_text(embed_query)
    except Exception as exc:
        logger.error("Embedding hatası (chunk %s): %s", chunk.get("chunk_id"), exc)
        return False

    payload = {
        "hisse": hisse,
        "sinyal_tipi": signal.get("sinyal_tipi"),
        "fiyat": signal.get("fiyat"),  # None ise null olarak saklanır
        "para_birimi": signal.get("para_birimi", "TL"),
        "guven": signal.get("guven"),
        "kaynak_cumle": signal.get("kaynak_cumle", ""),
        "gerekce": signal.get("gerekce", ""),
        "chunk_id": chunk.get("chunk_id"),
        "zaman_sn": chunk.get("start_sec", 0.0),
        "video_title": video_title,
        "created_at": datetime.utcnow().isoformat(),
    }

    point = PointStruct(
        id=str(uuid4()),
        vector={
            "dense": embed["dense"],
            "sparse": SparseVector(
                indices=embed["sparse"]["indices"],
                values=embed["sparse"]["values"],
            ),
        },
        payload=payload,
    )

    try:
        collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
        get_client().upsert(collection_name=collection, points=[point])
        logger.info("Sinyal yüklendi: %s %s (chunk %s)", hisse, signal.get("sinyal_tipi"), chunk.get("chunk_id"))
        return True
    except Exception as exc:
        logger.error("Qdrant upsert hatası (chunk %s): %s", chunk.get("chunk_id"), exc)
        return False


def upload_chunk_results(chunk: dict, analysis: dict, video_title: str) -> None:
    """Bir chunk'ın tüm sinyallerini Qdrant'a yükler.

    Args:
        chunk: Kaynak chunk dict'i.
        analysis: analyze_chunk() çıktısı.
        video_title: Kaynak video/ses dosyası başlığı.
    """
    sinyaller = analysis.get("sinyaller", [])
    if not sinyaller:
        logger.info("Chunk %s'de sinyal yok, atlanıyor", chunk.get("chunk_id"))
        return

    success, skipped = 0, 0
    for signal in sinyaller:
        try:
            uploaded = upload_signal(signal, chunk, video_title)
            if uploaded:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Sinyal yükleme hatası: %s", exc)
            skipped += 1

    logger.info(
        "Chunk %s yükleme tamamlandı: %d başarılı, %d atlandı",
        chunk.get("chunk_id"), success, skipped,
    )
