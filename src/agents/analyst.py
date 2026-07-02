"""Ajan 2 — Analyst: chunk'tan finansal sinyalleri çıkarır.

safe_llm_call() üzerinden Gemini 2.5 Flash (yedek: GPT-4o-mini) kullanır.
Her chunk için JSON şemasına uygun sinyal dict'i döndürür.
"""

import json

from src.transcription.prompts import build_analyst_system_prompt, load_bist_ticker_names, load_bist_tickers
from src.utils.api_helpers import safe_llm_call
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BIST_TICKERS = load_bist_tickers()
_BIST_TICKER_NAMES = load_bist_ticker_names()

_REQUIRED_KEYS = {"chunk_id", "bahsedilen_hisseler", "sinyaller", "teknik_terimler", "genel_yorum"}


def analyze_chunk(chunk: dict, seen_stocks: list[str]) -> dict:
    """Tek bir chunk'ı LLM ile analiz eder ve sinyal dict'i döndürür.

    Args:
        chunk: process_chunks() çıktısından tek bir chunk.
        seen_stocks: O ana kadar tüm chunk'larda görülen hisse kodları.

    Returns:
        JSON şemasına uygun dict. Hata durumunda boş şema döndürür.
    """
    chunk_id = chunk.get("chunk_id", "??")

    user_msg = (
        f"[ÖNCEKİ BAĞLAM]:\n{chunk.get('context_prefix', '')}\n\n"
        f"[ŞİMDİYE KADAR BAHSEDİLEN HİSSELER]:\n"
        f"{', '.join(seen_stocks) if seen_stocks else 'Henüz yok'}\n\n"
        f"[ANALİZ METNİ]:\n{chunk.get('text', '')}\n\n"
        f"Chunk ID: {chunk_id}"
    )

    system = build_analyst_system_prompt(_BIST_TICKERS, _BIST_TICKER_NAMES)

    try:
        raw = safe_llm_call(prompt=user_msg, system=system)
        cleaned = _strip_markdown(raw)
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Chunk %s JSON parse hatası: %s | Yanıt: %.200s", chunk_id, exc, raw if 'raw' in dir() else "")
        return _empty_schema(chunk_id)
    except Exception as exc:
        logger.error("Chunk %s analiz hatası: %s", chunk_id, exc)
        return _empty_schema(chunk_id)

    if not validate_result(result):
        logger.warning("Chunk %s geçersiz şema, boş döndürülüyor", chunk_id)
        return _empty_schema(chunk_id)

    # chunk_id'yi her zaman kaynaktan al (model yanlış yazabilir)
    result["chunk_id"] = chunk_id

    # Post-processing: fiyatsız satım → genel_yorum (prompt kuralı garantisi)
    for s in result.get("sinyaller", []):
        if s.get("sinyal_tipi") == "satım" and s.get("fiyat") is None:
            s["sinyal_tipi"] = "genel_yorum"
            logger.info("Fiyatsız satım → genel_yorum dönüştürüldü (chunk %s, hisse %s)", chunk_id, s.get("hisse"))

    new_stocks = extract_stocks_from_result(result)
    logger.info(
        "Chunk %s analiz edildi: %d sinyal, %d yeni hisse",
        chunk_id, len(result.get("sinyaller", [])), len(new_stocks),
    )
    return result


def extract_stocks_from_result(result: dict) -> list[str]:
    """Analiz sonucundan geçerli BIST ticker'larını çıkarır.

    Args:
        result: analyze_chunk() çıktısı.

    Returns:
        "belirsiz" olmayanlar ve _BIST_TICKERS içinde bulunanlar.
    """
    stocks: list[str] = []
    for ticker in result.get("bahsedilen_hisseler", []):
        if isinstance(ticker, str) and ticker != "belirsiz" and ticker in _BIST_TICKERS:
            stocks.append(ticker)
    return stocks


def validate_result(result: dict) -> bool:
    """Sonucun gerekli alanları içerip içermediğini kontrol eder.

    Args:
        result: json.loads() ile parse edilmiş LLM yanıtı.

    Returns:
        True ise şema geçerli, False ise eksik alan var.
    """
    if not isinstance(result, dict):
        return False
    missing = _REQUIRED_KEYS - result.keys()
    if missing:
        logger.warning("Şemada eksik alanlar: %s", missing)
        return False
    return True


def _strip_markdown(text: str) -> str:
    """```json ... ``` veya ``` ... ``` bloklarını soyar."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # ilk satır ``` veya ```json, son satır ```
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()
    return text


def _empty_schema(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "bahsedilen_hisseler": [],
        "sinyaller": [],
        "teknik_terimler": [],
        "genel_yorum": "analiz başarısız",
    }
