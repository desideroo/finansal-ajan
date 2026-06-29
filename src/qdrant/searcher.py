"""Qdrant'ta finansal sinyal arama modülü.

Hisse kodu, sinyal tipi ve serbest metin ile hybrid (dense+sparse) arama.
"""

import os

from dotenv import load_dotenv
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Prefetch, SparseVector
from qdrant_client.models import FusionQuery

from src.qdrant.client import get_client
from src.qdrant.uploader import embed_text
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _collection() -> str:
    return os.getenv("QDRANT_COLLECTION", "finansal_analiz")


def _hits_to_payloads(hits) -> list[dict]:
    return [h.payload for h in hits if h.payload]


def search_by_stock(hisse: str, limit: int = 10) -> list[dict]:
    """Belirli bir hisse koduna ait tüm sinyalleri döndürür.

    Args:
        hisse: BIST ticker kodu (ör. "THYAO").
        limit: Maksimum sonuç sayısı.

    Returns:
        Payload dict listesi.
    """
    try:
        flt = Filter(must=[FieldCondition(key="hisse", match=MatchValue(value=hisse))])
        hits = get_client().scroll(
            collection_name=_collection(),
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
        )[0]
        logger.info("search_by_stock '%s': %d sonuç", hisse, len(hits))
        return _hits_to_payloads(hits)
    except Exception as exc:
        logger.error("search_by_stock hatası: %s", exc)
        return []


def search_by_signal_type(sinyal_tipi: str, limit: int = 10) -> list[dict]:
    """Belirli bir sinyal tipine ait kayıtları döndürür.

    Args:
        sinyal_tipi: alım | satım | stop_loss | destek | direnc | genel_yorum
        limit: Maksimum sonuç sayısı.

    Returns:
        Payload dict listesi.
    """
    try:
        flt = Filter(must=[FieldCondition(key="sinyal_tipi", match=MatchValue(value=sinyal_tipi))])
        hits = get_client().scroll(
            collection_name=_collection(),
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
        )[0]
        logger.info("search_by_signal_type '%s': %d sonuç", sinyal_tipi, len(hits))
        return _hits_to_payloads(hits)
    except Exception as exc:
        logger.error("search_by_signal_type hatası: %s", exc)
        return []


def search_semantic(query: str, limit: int = 10) -> list[dict]:
    """Serbest metin ile dense+sparse hybrid arama yapar (RRF füzyon).

    Args:
        query: Arama sorgusu.
        limit: Maksimum sonuç sayısı.

    Returns:
        Payload dict listesi.
    """
    try:
        embed = embed_text(query)
        hits = get_client().query_points(
            collection_name=_collection(),
            prefetch=[
                Prefetch(query=embed["dense"], using="dense", limit=limit * 2),
                Prefetch(
                    query=SparseVector(
                        indices=embed["sparse"]["indices"],
                        values=embed["sparse"]["values"],
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=limit,
            with_payload=True,
        ).points
        logger.info("search_semantic '%s': %d sonuç", query[:50], len(hits))
        return _hits_to_payloads(hits)
    except Exception as exc:
        logger.error("search_semantic hatası: %s", exc)
        return []


def search_filtered(
    query: str,
    hisse: str | None = None,
    sinyal_tipi: str | None = None,
    guven: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Opsiyonel metadata filtresiyle hybrid arama yapar.

    Args:
        query: Serbest metin sorgusu.
        hisse: Hisse kodu filtresi (None = filtre yok).
        sinyal_tipi: Sinyal tipi filtresi (None = filtre yok).
        guven: Güven seviyesi filtresi (None = filtre yok).
        limit: Maksimum sonuç sayısı.

    Returns:
        Payload dict listesi.
    """
    try:
        conditions = []
        if hisse:
            conditions.append(FieldCondition(key="hisse", match=MatchValue(value=hisse)))
        if sinyal_tipi:
            conditions.append(FieldCondition(key="sinyal_tipi", match=MatchValue(value=sinyal_tipi)))
        if guven:
            conditions.append(FieldCondition(key="guven", match=MatchValue(value=guven)))

        flt = Filter(must=conditions) if conditions else None

        embed = embed_text(query)
        hits = get_client().query_points(
            collection_name=_collection(),
            prefetch=[
                Prefetch(query=embed["dense"], using="dense", limit=limit * 2),
                Prefetch(
                    query=SparseVector(
                        indices=embed["sparse"]["indices"],
                        values=embed["sparse"]["values"],
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
        logger.info(
            "search_filtered '%s' (hisse=%s, tip=%s, güven=%s): %d sonuç",
            query[:40], hisse, sinyal_tipi, guven, len(hits),
        )
        return _hits_to_payloads(hits)
    except Exception as exc:
        logger.error("search_filtered hatası: %s", exc)
        return []
