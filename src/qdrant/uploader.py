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
    import re as _re
    from src.transcription.prompts import load_bist_aliases, load_bist_ticker_names, load_bist_tickers
    _valid_tickers = load_bist_tickers()
    _ticker_names = load_bist_ticker_names()
    _aliases = load_bist_aliases()
    # Resmi şirket adı → ticker ters eşleştirme (normalize edilmiş)
    _name_to_ticker = {
        _re.sub(r"[^a-z0-9]", "", v.lower()): k
        for k, v in _ticker_names.items()
    }

    hisse = signal.get("hisse", "belirsiz")
    if hisse == "belirsiz":
        logger.info("Belirsiz hisse atlandı (chunk %s)", chunk.get("chunk_id"))
        return False
    if hisse not in _valid_tickers:
        normalized = _re.sub(r"[^a-z0-9]", "", hisse.lower())
        matched = None

        # 1) Alias listesinde tam eşleşme (paholi, vakfa, bim...)
        matched = _aliases.get(hisse.lower()) or _aliases.get(normalized)

        # 2) Resmi şirket adında tam eşleşme
        if not matched:
            matched = _name_to_ticker.get(normalized)

        # 3) Kısmi eşleşme (şirket adı içinde geçiyor mu?)
        if not matched:
            for norm_name, ticker in _name_to_ticker.items():
                if len(normalized) >= 4 and (normalized in norm_name or norm_name in normalized):
                    matched = ticker
                    break

        if matched:
            logger.info("Hisse eşleştirildi: '%s' → %s", hisse, matched)
            hisse = matched
            signal["hisse"] = matched
        else:
            logger.info("Geçersiz hisse atlandı: '%s' (chunk %s)", hisse, chunk.get("chunk_id"))
            return False

    embed_query = f"{hisse} {signal.get('sinyal_tipi', '')} {signal.get('gerekce', '')}"
    try:
        embed = embed_text(embed_query)
    except Exception as exc:
        logger.error("Embedding hatası (chunk %s): %s", chunk.get("chunk_id"), exc)
        return False

    raw_fiyat = signal.get("fiyat")
    normalized_fiyat = round(float(raw_fiyat), 2) if raw_fiyat is not None else None

    payload = {
        "hisse": hisse,
        "sinyal_tipi": signal.get("sinyal_tipi"),
        "fiyat": normalized_fiyat,  # None ise null olarak saklanır
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


def _is_duplicate(hisse: str, sinyal_tipi: str, fiyat, video_title: str) -> bool:
    """Aynı hisse+tip+fiyat+kaynak kombinasyonu Qdrant'ta zaten var mı kontrol eder."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
    try:
        collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
        conditions = [
            FieldCondition(key="hisse", match=MatchValue(value=hisse)),
            FieldCondition(key="sinyal_tipi", match=MatchValue(value=sinyal_tipi)),
            FieldCondition(key="video_title", match=MatchValue(value=video_title)),
        ]
        if fiyat is not None:
            f = round(float(fiyat), 2)
            # Range filtresi: MatchValue float tip uyumsuzluğunu atlatır
            conditions.append(FieldCondition(key="fiyat", range=Range(gte=f - 0.005, lte=f + 0.005)))
        results, _ = get_client().scroll(
            collection_name=collection,
            scroll_filter=Filter(must=conditions),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0
    except Exception:
        return False


def delete_by_video_title(video_title: str) -> int:
    """Verilen video_title'a ait tüm Qdrant noktalarını siler.

    Args:
        video_title: Silinecek kaynak başlığı.

    Returns:
        Silinen nokta sayısı (tahmin), hata durumunda -1.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    try:
        collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
        client = get_client()

        # Önce kaç nokta var say
        results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="video_title", match=MatchValue(value=video_title))
            ]),
            limit=1000,
            with_payload=False,
            with_vectors=False,
        )
        count = len(results)
        if count == 0:
            logger.info("Qdrant'ta '%s' için kayıt yok, silme atlandı", video_title)
            return 0

        client.delete(
            collection_name=collection,
            points_selector=Filter(must=[
                FieldCondition(key="video_title", match=MatchValue(value=video_title))
            ]),
        )
        logger.info("Qdrant temizlendi: '%s' → %d nokta silindi", video_title, count)
        return count
    except Exception as exc:
        logger.error("Qdrant silme hatası ('%s'): %s", video_title, exc)
        return -1


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

    # Chunk içi tekilleştirme: aynı hisse+tip+fiyat kombinasyonunu bir kez al
    seen_in_chunk: set = set()
    unique_sinyaller = []
    for s in sinyaller:
        key = (s.get("hisse", "belirsiz"), s.get("sinyal_tipi", ""), s.get("fiyat"))
        if key not in seen_in_chunk:
            seen_in_chunk.add(key)
            unique_sinyaller.append(s)
    if len(unique_sinyaller) < len(sinyaller):
        logger.info("Chunk %s içi duplikat: %d → %d sinyal",
                    chunk.get("chunk_id"), len(sinyaller), len(unique_sinyaller))
    sinyaller = unique_sinyaller

    success, skipped, dupes = 0, 0, 0
    for signal in sinyaller:
        try:
            hisse = signal.get("hisse", "belirsiz")
            tip   = signal.get("sinyal_tipi", "")
            fiyat = signal.get("fiyat")
            if hisse != "belirsiz" and _is_duplicate(hisse, tip, fiyat, video_title):
                logger.info("Tekrar sinyal atlandı: %s %s %.2f (chunk %s)",
                            hisse, tip, fiyat or 0, chunk.get("chunk_id"))
                dupes += 1
                continue
            uploaded = upload_signal(signal, chunk, video_title)
            if uploaded:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Sinyal yükleme hatası: %s", exc)
            skipped += 1

    logger.info(
        "Chunk %s yükleme tamamlandı: %d başarılı, %d tekrar, %d atlandı",
        chunk.get("chunk_id"), success, dupes, skipped,
    )
