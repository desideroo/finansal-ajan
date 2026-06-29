"""Fiyat doğrulama ajanı — analist sinyallerini anlık fiyatla karşılaştırır.

borsapy kullanır (yfinance DEĞİL). Ticker'larda .IS suffix yok.
5 dakikalık in-memory cache ile gereksiz API çağrıları önlenir.
"""

from datetime import datetime, timedelta

import borsapy as bp

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL = timedelta(minutes=5)
_price_cache: dict[str, dict] = {}


def get_current_price(hisse: str) -> dict:
    """Anlık fiyatı borsapy ile çeker, 5 dk cache'ler.

    Args:
        hisse: BIST ticker kodu (ör. "THYAO").

    Returns:
        {"hisse": str, "anlik_fiyat": float|None, "timestamp": str}
    """
    now = datetime.utcnow()
    cached = _price_cache.get(hisse)
    if cached and (now - datetime.fromisoformat(cached["timestamp"])) < _CACHE_TTL:
        logger.info("Cache hit: %s = %s TL", hisse, cached["anlik_fiyat"])
        return cached

    try:
        price = float(bp.Ticker(hisse).fast_info["last_price"])
        logger.info("Anlık fiyat alındı: %s = %.2f TL", hisse, price)
    except Exception as exc:
        logger.warning("Fiyat alınamadı (%s): %s", hisse, exc)
        price = None

    result = {
        "hisse": hisse,
        "anlik_fiyat": price,
        "timestamp": now.isoformat(),
    }
    _price_cache[hisse] = result
    return result


def verify_signal(signal: dict) -> dict:
    """Analist sinyalini anlık fiyatla karşılaştırır.

    Args:
        signal: Qdrant payload veya analyst.py çıktısındaki sinyal dict.

    Returns:
        Orijinal sinyal + anlik_fiyat, fark_yuzde, yorum alanları.
    """
    hisse = signal.get("hisse", "belirsiz")
    analist_fiyat = signal.get("fiyat")

    if analist_fiyat is None:
        return {
            **signal,
            "anlik_fiyat": None,
            "fark_yuzde": None,
            "yorum": "Analist fiyat vermedi, doğrulama yapılamadı.",
        }

    price_data = get_current_price(hisse)
    anlik = price_data["anlik_fiyat"]

    if anlik is None:
        return {
            **signal,
            "anlik_fiyat": None,
            "fark_yuzde": None,
            "yorum": f"Anlık fiyat alınamadı ({hisse}).",
        }

    fark_yuzde = round((anlik - analist_fiyat) / analist_fiyat * 100, 2)
    isaret = "+" if fark_yuzde >= 0 else ""
    yorum = (
        f"Analist {analist_fiyat:.2f} TL dedi, "
        f"şu an {anlik:.2f} TL ({isaret}{fark_yuzde}%)"
    )

    return {
        **signal,
        "anlik_fiyat": anlik,
        "fark_yuzde": fark_yuzde,
        "yorum": yorum,
    }
