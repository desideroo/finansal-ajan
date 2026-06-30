"""MLX Whisper tabanlı ses transkripsiyon ve chunk oluşturma modülü.

Uzun Türkçe ses dosyalarını segment bazlı ~10 dakikalık chunk'lara böler.
3 katmanlı bağlam koruması: segment sınırları, 300 kelime overlap, seen_stocks.
"""

import json
import math
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path

import mlx_whisper

from src.transcription.prompts import WHISPER_INITIAL_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHECKPOINT = "mlx-community/whisper-large-v3-mlx"
_HALLUCINATION_REPEAT_THRESHOLD = 4


def _cache_path(audio_path: str) -> Path:
    return Path(audio_path).with_suffix(".segments.json")


# ── Ses araçları ──────────────────────────────────────────────────────────────

def get_audio_duration(audio_path: str) -> float:
    """ffprobe ile ses dosyasının süresini saniye cinsinden döndürür."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe ile süre alınamadı: %s", exc)
        return 0.0


def _extract_segment_wav(audio_path: str, start: float, duration: float, out_path: str) -> bool:
    """ffmpeg ile sesten bir dilimi WAV olarak çıkarır."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(start), "-t", str(duration),
             "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            capture_output=True, timeout=120,
        )
        return Path(out_path).exists() and Path(out_path).stat().st_size > 0
    except Exception as exc:
        logger.error("ffmpeg segment çıkarma hatası: %s", exc)
        return False


# ── Tek seferlik tam transkripsiyon (cache destekli) ─────────────────────────

def transcribe_audio(audio_path: str, use_cache: bool = True) -> list[dict]:
    """Ses dosyasını MLX Whisper ile metne çevirir ve segment listesi döndürür."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    cache = _cache_path(audio_path)
    if use_cache and cache.exists():
        logger.info("Cache'den yükleniyor: %s", cache.name)
        with cache.open(encoding="utf-8") as f:
            return json.load(f)

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
            condition_on_previous_text=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Whisper transkripsiyon başarısız: {exc}") from exc

    segments = result.get("segments", [])
    logger.info("Transkripsiyon tamamlandı: %d segment", len(segments))

    if use_cache:
        with cache.open("w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
        logger.info("Segment cache'e yazıldı: %s", cache.name)
    return segments


# ── Gerçek zamanlı streaming transkripsiyon ───────────────────────────────────

def transcribe_streaming(
    audio_path: str,
    chunk_minutes: int = 10,
    cancelled: threading.Event | None = None,
) -> Generator[dict, None, None]:
    """Ses dosyasını parça parça transkribe eder, her chunk hazır olunca yield eder.

    Cache varsa segmentleri yükler ve chunk'ları simüle eder (çok hızlı).
    Cache yoksa ffmpeg ile ses parçalara bölünür, her parça ayrı Whisper çağrısına gider.

    Yields:
        {
            "chunk_idx": int,
            "total_chunks": int,
            "chunk_id": str,
            "start_sec": float,
            "end_sec": float,
            "text": str,
            "word_count": int,
            "from_cache": bool,
        }
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    cache = _cache_path(audio_path)
    chunk_secs = chunk_minutes * 60

    # ── Cache varsa: segmentlerden chunk'ları direkt oluştur ─────────────────
    if cache.exists():
        logger.info("Cache mevcut, chunk'lar direkt oluşturuluyor")
        with cache.open(encoding="utf-8") as f:
            all_segments = json.load(f)

        clean = [s for s in all_segments if not _is_hallucination(s.get("text", ""))]
        raw_chunks = build_chunks(clean, chunk_minutes)
        total = len(raw_chunks)

        for i, chunk in enumerate(raw_chunks):
            if cancelled and cancelled.is_set():
                logger.info("Transkripsiyon iptal edildi (chunk %d/%d)", i, total)
                return
            yield {
                "chunk_idx": i,
                "total_chunks": total,
                "chunk_id": chunk["chunk_id"],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "text": chunk["text"],
                "word_count": len(chunk["text"].split()),
                "from_cache": True,
            }
        return

    # ── Cache yok: ffmpeg ile böl, her parçayı Whisper'a ver ─────────────────
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        logger.warning("Ses süresi alınamadı, tam transkripsiyon yapılıyor")
        segments = transcribe_audio(audio_path, use_cache=True)
        raw_chunks = build_chunks(segments, chunk_minutes)
        total = len(raw_chunks)
        for i, chunk in enumerate(raw_chunks):
            yield {**chunk, "chunk_idx": i, "total_chunks": total,
                   "word_count": len(chunk["text"].split()), "from_cache": False}
        return

    total_chunks = math.ceil(duration / chunk_secs)
    logger.info("Ses süresi: %.0f sn → %d chunk planlandı", duration, total_chunks)

    tmp_dir = tempfile.mkdtemp(prefix="borsa_whisper_")
    all_segments: list[dict] = []
    chunk_idx = 0

    try:
        for i in range(total_chunks):
            if cancelled and cancelled.is_set():
                logger.info("Transkripsiyon iptal edildi (%d/%d)", i, total_chunks)
                break

            start = i * chunk_secs
            actual_end = min(start + chunk_secs, duration)
            seg_duration = actual_end - start

            piece_path = str(Path(tmp_dir) / f"piece_{i:02d}.wav")
            logger.info("Parça %d/%d çıkarılıyor (%.0f-%.0f sn)...", i + 1, total_chunks, start, actual_end)

            if not _extract_segment_wav(audio_path, start, seg_duration, piece_path):
                logger.error("Parça %d çıkarılamadı, atlanıyor", i)
                continue

            # Whisper bu parçaya uygula
            try:
                result = mlx_whisper.transcribe(
                    piece_path,
                    path_or_hf_repo=_CHECKPOINT,
                    language="tr",
                    temperature=0.0,
                    word_timestamps=True,
                    initial_prompt=WHISPER_INITIAL_PROMPT,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
            except Exception as exc:
                logger.error("Parça %d Whisper hatası: %s", i, exc)
                continue

            # Timestamp'leri gerçek ses konumuna kaydır
            piece_segs = result.get("segments", [])
            for seg in piece_segs:
                seg["start"] = round(seg["start"] + start, 2)
                seg["end"] = round(seg["end"] + start, 2)

            # Halüsinasyon filtrele
            clean_segs = [s for s in piece_segs if not _is_hallucination(s.get("text", ""))]
            all_segments.extend(clean_segs)

            text = " ".join(s["text"].strip() for s in clean_segs)
            if not text.strip():
                logger.info("Parça %d: metin boş (sessizlik/müzik?), atlanıyor", i)
                continue

            yield {
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "chunk_id": f"{chunk_idx:02d}",
                "start_sec": round(start, 2),
                "end_sec": round(actual_end, 2),
                "text": text,
                "word_count": len(text.split()),
                "from_cache": False,
            }
            chunk_idx += 1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Tüm segmentleri cache'e kaydet
    if all_segments and not (cancelled and cancelled.is_set()):
        with cache.open("w", encoding="utf-8") as f:
            json.dump(all_segments, f, ensure_ascii=False)
        logger.info("Tüm segmentler cache'e kaydedildi: %s", cache.name)


# ── Halüsinasyon tespiti ──────────────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    words = text.strip().split()
    if len(words) < 6:
        return False
    trigram = " ".join(words[:3])
    if text.count(trigram) >= _HALLUCINATION_REPEAT_THRESHOLD:
        return True
    bigrams: dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    max_repeat = max(bigrams.values()) if bigrams else 0
    if max_repeat >= max(8, len(words) * 0.30):
        return True
    return False


# ── Chunk oluşturma yardımcıları ──────────────────────────────────────────────

def build_chunks(segments: list, chunk_minutes: int = 10) -> list[dict]:
    if not segments:
        return []
    clean_segments = [s for s in segments if not _is_hallucination(s.get("text", ""))]
    skipped = len(segments) - len(clean_segments)
    if skipped:
        logger.warning("%d halüsinasyon segmenti filtrelendi", skipped)
    if not clean_segments:
        return []

    chunk_seconds = chunk_minutes * 60
    chunks: list[dict] = []
    current_segs: list[dict] = []
    chunk_start: float = clean_segments[0]["start"]

    for seg in clean_segments:
        current_segs.append(seg)
        if seg["end"] - chunk_start >= chunk_seconds:
            chunks.append(_make_chunk(len(chunks), chunk_start, current_segs))
            chunk_start = seg["end"]
            current_segs = []

    if current_segs:
        chunks.append(_make_chunk(len(chunks), chunk_start, current_segs))

    logger.info("%d chunk oluşturuldu", len(chunks))
    return chunks


def add_overlap(chunks: list[dict], overlap_words: int = 300) -> list[dict]:
    result: list[dict] = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if i > 0:
            prev_words = chunks[i - 1]["text"].split()
            prefix = " ".join(prev_words[-overlap_words:])
        result.append({**chunk, "context_prefix": prefix, "seen_stocks": []})
    return result


def _make_chunk(index: int, start: float, segs: list[dict]) -> dict:
    text = " ".join(s["text"].strip() for s in segs)
    return {
        "chunk_id": f"{index:02d}",
        "start_sec": round(start, 2),
        "end_sec": round(segs[-1]["end"], 2),
        "text": text,
    }
