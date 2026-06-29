"""Ajan 1 — Chunker: transcriber çıktısını Ajan 2'ye hazırlar.

LLM çağrısı YAPILMAZ. Yalnızca Python ile chunk doğrulama,
ID atama, kısa chunk filtreleme ve seen_stocks taşıma işlemleri yapılır.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_WORDS = 10


def process_chunks(raw_chunks: list[dict]) -> list[dict]:
    """Tam Ajan 1 pipeline'ı: ID ata → filtrele → seen_stocks taşı.

    Args:
        raw_chunks: transcribe_audio() + add_overlap() çıktısı.

    Returns:
        Ajan 2'ye gönderilmeye hazır, word_count eklenmiş chunk listesi.
    """
    validated = _validate_and_normalize(raw_chunks)
    with_ids = assign_chunk_ids(validated)
    filtered = filter_empty_chunks(with_ids)
    result = _carry_seen_stocks(filtered)
    logger.info("Chunker tamamlandı: %d → %d chunk (filtre sonrası)", len(raw_chunks), len(result))
    return result


def assign_chunk_ids(chunks: list[dict]) -> list[dict]:
    """Chunk listesine sıfırdan başlayan zero-padded ID atar.

    Args:
        chunks: Normalleştirilmiş chunk listesi.

    Returns:
        chunk_id alanı güncellenmiş liste (orijinal liste değiştirilmez).
    """
    result = []
    for i, chunk in enumerate(chunks):
        result.append({**chunk, "chunk_id": f"{i:02d}"})
    return result


def filter_empty_chunks(chunks: list[dict]) -> list[dict]:
    """10 kelimeden kısa chunk'ları filtreler ve loglar.

    Args:
        chunks: ID atanmış chunk listesi.

    Returns:
        Yeterli içeriğe sahip chunk'lar.
    """
    result = []
    for chunk in chunks:
        wc = len(chunk.get("text", "").split())
        if wc < _MIN_WORDS:
            logger.warning(
                "Chunk %s atlandı: çok kısa (%d kelime < %d eşik)",
                chunk.get("chunk_id", "?"), wc, _MIN_WORDS,
            )
            continue
        result.append({**chunk, "word_count": wc})
    return result


# ── yardımcı ─────────────────────────────────────────────────────────────────

def _validate_and_normalize(chunks: list[dict]) -> list[dict]:
    """Zorunlu alanları kontrol eder; eksikse varsayılan koyar, hatalıysa atlar."""
    result = []
    for i, chunk in enumerate(chunks):
        try:
            normalized = {
                "chunk_id": chunk.get("chunk_id", f"{i:02d}"),
                "start_sec": float(chunk.get("start_sec", 0.0)),
                "end_sec": float(chunk.get("end_sec", 0.0)),
                "text": chunk.get("text", "").strip(),
                "context_prefix": chunk.get("context_prefix", ""),
                "seen_stocks": list(chunk.get("seen_stocks", [])),
            }
            result.append(normalized)
        except Exception as exc:
            logger.warning("Chunk %d normalleştirilemedi, atlanıyor: %s", i, exc)
    return result


def _carry_seen_stocks(chunks: list[dict]) -> list[dict]:
    """seen_stocks listesini chunk'lar arasında birikimli olarak taşır.

    Ajan 2, her chunk'ta o ana kadar bahsedilen tüm hisseleri görmeli.
    seen_stocks bu aşamada boştur; analyst.py her chunk'tan sonra güncelleyecektir.
    Bu fonksiyon yalnızca listeyi başlatır ve sonraki chunk'a kopyalar.
    """
    accumulated: list[str] = []
    result = []
    for chunk in chunks:
        # Chunk kendi seen_stocks'u varsa (önceki çalışmadan) birleştir
        accumulated = list(dict.fromkeys(accumulated + chunk["seen_stocks"]))
        result.append({**chunk, "seen_stocks": list(accumulated)})
    return result
