"""MLX Whisper tabanlı ses transkripsiyon ve chunk oluşturma modülü.

Uzun Türkçe ses dosyalarını segment bazlı ~10 dakikalık chunk'lara böler.
3 katmanlı bağlam koruması: segment sınırları, 300 kelime overlap, seen_stocks.
"""

from pathlib import Path

import mlx_whisper

from src.transcription.prompts import WHISPER_INITIAL_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHECKPOINT = "mlx-community/whisper-large-v3-mlx"


def transcribe_audio(audio_path: str) -> list[dict]:
    """Ses dosyasını MLX Whisper ile metne çevirir ve segment listesi döndürür.

    Args:
        audio_path: Ses dosyasının yolu (.m4a, .mp3, .wav vb.).

    Returns:
        Whisper segment listesi — her eleman {start, end, text} içerir.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        RuntimeError: Whisper transkripsiyon başarısız olursa.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.error("Ses dosyası bulunamadı: %s", path.resolve())
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    logger.info("Transkripsiyon başlıyor: %s", path.name)
    try:
        result = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=_CHECKPOINT,
            language="tr",
            temperature=0.0,
            word_timestamps=True,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            no_speech_threshold=0.6,
        )
    except Exception as exc:
        logger.error("Whisper transkripsiyon hatası: %s", exc)
        raise RuntimeError(f"Whisper transkripsiyon başarısız: {exc}") from exc

    segments = result.get("segments", [])
    logger.info("Transkripsiyon tamamlandı: %d segment", len(segments))
    return segments


def build_chunks(segments: list, chunk_minutes: int = 10) -> list[dict]:
    """Whisper segmentlerini ~chunk_minutes dakikalık chunk'lara gruplar.

    Hiçbir zaman bir segmentin ortasında bölme yapmaz.

    Args:
        segments: transcribe_audio() çıktısı — Whisper segment listesi.
        chunk_minutes: Hedef chunk süresi dakika cinsinden (varsayılan 10).

    Returns:
        Chunk sözlükleri listesi. seen_stocks bu aşamada boş listedir;
        add_overlap() sonrasında doldurulur.
    """
    if not segments:
        logger.warning("Segment listesi boş, chunk oluşturulamadı")
        return []

    chunk_seconds = chunk_minutes * 60
    chunks: list[dict] = []
    current_segs: list[dict] = []
    chunk_start: float = segments[0]["start"]

    for seg in segments:
        current_segs.append(seg)
        elapsed = seg["end"] - chunk_start

        if elapsed >= chunk_seconds:
            chunks.append(_make_chunk(len(chunks), chunk_start, current_segs))
            chunk_start = seg["end"]
            current_segs = []

    # Kalan segmentler son chunk'ı oluşturur
    if current_segs:
        chunks.append(_make_chunk(len(chunks), chunk_start, current_segs))

    logger.info("%d chunk oluşturuldu (%d dk hedef)", len(chunks), chunk_minutes)
    return chunks


def add_overlap(chunks: list[dict], overlap_words: int = 300) -> list[dict]:
    """Her chunk'a önceki chunk'ın son overlap_words kelimesini context_prefix olarak ekler.

    Bu prefix Qdrant'a kaydedilmez; yalnızca LLM prompt'unda kullanılır.
    seen_stocks başlangıçta boş listedir; analyst.py tarafından doldurulur.

    Args:
        chunks: build_chunks() çıktısı.
        overlap_words: Önceki chunk'tan alınacak kelime sayısı (varsayılan 300).

    Returns:
        context_prefix ve seen_stocks alanları eklenmiş chunk listesi.
    """
    result: list[dict] = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if i > 0:
            prev_words = chunks[i - 1]["text"].split()
            prefix = " ".join(prev_words[-overlap_words:])

        result.append({
            **chunk,
            "context_prefix": prefix,
            "seen_stocks": [],
        })

    logger.info("Overlap eklendi: %d chunk, her biri max %d kelime prefix", len(result), overlap_words)
    return result


# ── yardımcı ─────────────────────────────────────────────────────────────────

def _make_chunk(index: int, start: float, segs: list[dict]) -> dict:
    """Segment listesinden ham chunk sözlüğü üretir (prefix/seen_stocks hariç)."""
    text = " ".join(s["text"].strip() for s in segs)
    return {
        "chunk_id": f"{index:02d}",
        "start_sec": round(start, 2),
        "end_sec": round(segs[-1]["end"], 2),
        "text": text,
    }
