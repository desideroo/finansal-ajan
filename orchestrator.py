"""Ana orkestratör — transkripsiyon, chunking, analiz ve Qdrant yükleme pipeline'ı.

Tüm modülleri sırayla koordine eder. Her chunk bağımsız olarak işlenir;
tek chunk hatası pipeline'ı durdurmaz.
"""

import time

from src.agents.analyst import analyze_chunk, extract_stocks_from_result
from src.agents.chunker import process_chunks
from src.qdrant.client import get_or_create_collection
from src.qdrant.uploader import upload_chunk_results
from src.transcription.transcriber import add_overlap, build_chunks, transcribe_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    audio_path: str,
    video_title: str = "bilinmiyor",
) -> dict:
    """Ses dosyasından Qdrant'a tam pipeline'ı çalıştırır.

    Args:
        audio_path: İşlenecek ses dosyasının yolu.
        video_title: Qdrant metadata'sında saklanacak kaynak başlığı.

    Returns:
        Pipeline özet istatistikleri.

    Raises:
        FileNotFoundError: Ses dosyası bulunamazsa.
        RuntimeError: Transkripsiyon veya chunking başarısız olursa.
    """
    start = time.time()
    logger.info("Pipeline başladı: %s | %s", audio_path, video_title)

    # 1 — Koleksiyonu hazırla
    get_or_create_collection()

    # 2 — Transkripsiyon
    logger.info("Adım 1/3: Transkripsiyon başlıyor")
    segments = transcribe_audio(audio_path)

    # 3 — Chunking: segment → ~10dk gruplar → overlap ekle → doğrula/filtrele
    logger.info("Adım 2/3: Chunk işleme başlıyor (%d segment)", len(segments))
    raw_chunks = build_chunks(segments, chunk_minutes=10)
    overlapped = add_overlap(raw_chunks, overlap_words=300)
    chunks = process_chunks(overlapped)
    toplam = len(chunks)
    logger.info("%d chunk oluşturuldu", toplam)

    if toplam == 0:
        logger.warning("Hiç chunk oluşturulamadı, pipeline sonlanıyor")
        return _summary(audio_path, video_title, 0, 0, 0, 0, [], time.time() - start)

    # 4 — Her chunk: analiz → yükle → seen_stocks güncelle
    logger.info("Adım 3/3: Chunk analizi başlıyor")
    seen_stocks: list[str] = []
    basarili, basarisiz, toplam_sinyal = 0, 0, 0
    tum_hisseler: set[str] = set()

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"{i:02d}")
        logger.info("Chunk %s işleniyor (%d/%d)", chunk_id, i + 1, toplam)

        # Analiz
        try:
            analysis = analyze_chunk(chunk, seen_stocks)
        except Exception as exc:
            logger.error("Chunk %s analiz hatası, atlanıyor: %s", chunk_id, exc)
            basarisiz += 1
            continue

        # Yükleme
        try:
            upload_chunk_results(chunk, analysis, video_title)
        except Exception as exc:
            logger.error("Chunk %s yükleme hatası, atlanıyor: %s", chunk_id, exc)
            basarisiz += 1
            continue

        # seen_stocks güncelle
        yeni = extract_stocks_from_result(analysis)
        seen_stocks = list(dict.fromkeys(seen_stocks + yeni))
        tum_hisseler.update(yeni)
        toplam_sinyal += len(analysis.get("sinyaller", []))
        basarili += 1

    sure = round(time.time() - start, 2)
    ozet = _summary(
        audio_path, video_title, toplam, basarili, basarisiz,
        toplam_sinyal, sorted(tum_hisseler), sure,
    )
    logger.info("Pipeline tamamlandı: %s", ozet)
    return ozet


def _summary(
    audio_path: str,
    video_title: str,
    toplam_chunk: int,
    basarili_chunk: int,
    basarisiz_chunk: int,
    toplam_sinyal: int,
    islenen_hisseler: list[str],
    sure_saniye: float,
) -> dict:
    return {
        "audio_path": audio_path,
        "video_title": video_title,
        "toplam_chunk": toplam_chunk,
        "basarili_chunk": basarili_chunk,
        "basarisiz_chunk": basarisiz_chunk,
        "toplam_sinyal": toplam_sinyal,
        "islenen_hisseler": islenen_hisseler,
        "sure_saniye": sure_saniye,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python orchestrator.py <audio_path> [video_title]")
        sys.exit(1)

    audio = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "bilinmiyor"
    result = run_pipeline(audio, title)
    print(result)
