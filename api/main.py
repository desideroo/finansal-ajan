"""FastAPI backend — ses yükleme, analiz ve sinyal sorgulama endpoint'leri."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.agents.analyst import analyze_chunk, extract_stocks_from_result
from src.agents.chunker import process_chunks
from src.qdrant.client import get_client, get_or_create_collection
from src.qdrant.searcher import search_by_stock, search_filtered
from src.qdrant.uploader import upload_chunk_results
from src.transcription.transcriber import add_overlap, build_chunks, transcribe_audio
from src.utils.logger import get_logger
from src.verification.agent import verify_signal

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="Borsa Analizi API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    get_or_create_collection()
    logger.info("API hazır")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── SSE yardımcısı ────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Transkripsiyon + chunking stream ─────────────────────────────────────────

@app.post("/stream/transcribe")
async def stream_transcribe(
    file: UploadFile = File(...),
    title: str = "bilinmiyor",
):
    """Ses dosyasını yükle, transkripsiyon yap, chunk'ları SSE ile aktar."""
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_path = None

    async def generate():
        nonlocal tmp_path
        try:
            # 1 — Dosyayı kaydet
            yield _sse("progress", {"adim": "kayit", "mesaj": "Dosya kaydediliyor...", "yuzde": 5})
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # 2 — Cache kontrolü
            cache_file = Path(tmp_path).with_suffix(".segments.json")
            if cache_file.exists():
                yield _sse("progress", {"adim": "transkripsiyon", "mesaj": "Cache'den yükleniyor...", "yuzde": 40})
            else:
                yield _sse("progress", {"adim": "transkripsiyon", "mesaj": "Whisper transkripsiyon başlıyor (bu ~25 dk sürebilir)...", "yuzde": 10})

            await asyncio.get_event_loop().run_in_executor(None, lambda: None)
            segments = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, tmp_path
            )
            yield _sse("progress", {"adim": "transkripsiyon", "mesaj": f"Transkripsiyon tamamlandı: {len(segments)} segment", "yuzde": 60})

            # 3 — Chunk oluşturma
            yield _sse("progress", {"adim": "chunking", "mesaj": "Chunk'lar oluşturuluyor...", "yuzde": 70})
            raw_chunks = build_chunks(segments, chunk_minutes=10)
            overlapped = add_overlap(raw_chunks, overlap_words=300)
            chunks = process_chunks(overlapped)
            yield _sse("progress", {"adim": "chunking", "mesaj": f"{len(chunks)} chunk oluşturuldu", "yuzde": 90})

            # 4 — Chunk verilerini gönder
            chunk_data = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "start_sec": c.get("start_sec"),
                    "end_sec": c.get("end_sec"),
                    "word_count": c.get("word_count"),
                    "text": c.get("text", ""),
                }
                for c in chunks
            ]

            # tmp_path'i payload'da sakla (analiz aşaması için)
            yield _sse("tamamlandi", {
                "adim": "transkripsiyon",
                "mesaj": "Transkripsiyon tamamlandı",
                "yuzde": 100,
                "chunks": chunk_data,
                "tmp_path": tmp_path,
                "title": title,
                "segment_sayisi": len(segments),
            })

        except Exception as exc:
            logger.error("Transkripsiyon stream hatası: %s", exc)
            yield _sse("hata", {"mesaj": str(exc)})
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Analiz stream ─────────────────────────────────────────────────────────────

@app.post("/stream/analyze")
async def stream_analyze(
    file: UploadFile = File(...),
    title: str = "bilinmiyor",
):
    """Transkripsiyon + analiz + Qdrant yükleme — her adım SSE ile iletilir."""
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_path = None

    async def generate():
        nonlocal tmp_path
        try:
            # 1 — Dosya kaydet
            yield _sse("progress", {"adim": "kayit", "mesaj": "Dosya kaydediliyor...", "yuzde": 2})
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # 2 — Transkripsiyon
            cache_file = Path(tmp_path).with_suffix(".segments.json")
            if cache_file.exists():
                mesaj = "Cache'den yükleniyor..."
            else:
                mesaj = "Whisper transkripsiyon (~25 dk)..."
            yield _sse("progress", {"adim": "transkripsiyon", "mesaj": mesaj, "yuzde": 5})

            segments = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, tmp_path
            )
            yield _sse("progress", {"adim": "transkripsiyon", "mesaj": f"{len(segments)} segment alındı", "yuzde": 35})

            # 3 — Chunking
            yield _sse("progress", {"adim": "chunking", "mesaj": "Chunk'lar hazırlanıyor...", "yuzde": 38})
            raw_chunks = build_chunks(segments, chunk_minutes=10)
            overlapped = add_overlap(raw_chunks, overlap_words=300)
            chunks = process_chunks(overlapped)
            toplam = len(chunks)
            yield _sse("progress", {"adim": "chunking", "mesaj": f"{toplam} chunk oluşturuldu", "yuzde": 40})

            if toplam == 0:
                yield _sse("hata", {"mesaj": "Hiç chunk oluşturulamadı"})
                return

            # 4 — Koleksiyon
            get_or_create_collection()

            # 5 — Her chunk analiz + yükle
            seen_stocks: list[str] = []
            tum_hisseler: set[str] = set()
            tum_sinyaller: list[dict] = []
            basarili = 0

            analiz_baslangic = 40
            analiz_aralik = 55  # 40% → 95%

            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get("chunk_id", f"{i:02d}")
                yuzde = int(analiz_baslangic + (i / toplam) * analiz_aralik)

                yield _sse("progress", {
                    "adim": "analiz",
                    "mesaj": f"Chunk {chunk_id} analiz ediliyor... ({i+1}/{toplam})",
                    "yuzde": yuzde,
                    "chunk_idx": i,
                    "toplam_chunk": toplam,
                })

                try:
                    analysis = await asyncio.get_event_loop().run_in_executor(
                        None, analyze_chunk, chunk, seen_stocks
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, upload_chunk_results, chunk, analysis, title
                    )

                    yeni = extract_stocks_from_result(analysis)
                    seen_stocks = list(dict.fromkeys(seen_stocks + yeni))
                    tum_hisseler.update(yeni)

                    sinyaller_bu_chunk = analysis.get("sinyaller", [])
                    for s in sinyaller_bu_chunk:
                        s["chunk_id"] = chunk_id
                    tum_sinyaller.extend(sinyaller_bu_chunk)
                    basarili += 1

                    yield _sse("chunk_tamamlandi", {
                        "chunk_id": chunk_id,
                        "hisseler": yeni,
                        "sinyal_sayisi": len(sinyaller_bu_chunk),
                        "sinyaller": sinyaller_bu_chunk,
                    })

                except Exception as exc:
                    logger.error("Chunk %s hatası: %s", chunk_id, exc)
                    yield _sse("chunk_hata", {"chunk_id": chunk_id, "mesaj": str(exc)})

            yield _sse("tamamlandi", {
                "adim": "analiz",
                "mesaj": "Analiz tamamlandı",
                "yuzde": 100,
                "ozet": {
                    "toplam_chunk": toplam,
                    "basarili_chunk": basarili,
                    "toplam_sinyal": len(tum_sinyaller),
                    "islenen_hisseler": sorted(tum_hisseler),
                },
                "sinyaller": tum_sinyaller,
            })

        except Exception as exc:
            logger.error("Analiz stream hatası: %s", exc)
            yield _sse("hata", {"mesaj": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Mevcut endpoint'ler ───────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    title: str = "bilinmiyor",
):
    from orchestrator import run_pipeline
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = run_pipeline(tmp_path, title)
        return result
    except Exception as exc:
        logger.error("Analiz hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/search")
async def search(
    q: str = Query(...),
    hisse: str | None = Query(None),
    sinyal_tipi: str | None = Query(None),
    guven: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    try:
        results = search_filtered(query=q, hisse=hisse, sinyal_tipi=sinyal_tipi, guven=guven, limit=limit)
        return {"results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stocks")
async def stocks():
    try:
        collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
        client = get_client()
        hisseler: set[str] = set()
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection, offset=offset, limit=100, with_payload=["hisse"],
            )
            for r in records:
                h = (r.payload or {}).get("hisse")
                if h and h != "belirsiz":
                    hisseler.add(h)
            if next_offset is None:
                break
            offset = next_offset
        return {"hisseler": sorted(hisseler)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/verify")
async def verify(hisse: str = Query(...)):
    try:
        sinyaller = search_by_stock(hisse, limit=50)
        ilgili = [s for s in sinyaller if s.get("sinyal_tipi") in ("alım", "satım")]
        results = [verify_signal(s) for s in ilgili]
        return {"hisse": hisse, "count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/progress")
async def progress(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Bağlantı kuruldu")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"pong: {data}")
    except Exception:
        pass
