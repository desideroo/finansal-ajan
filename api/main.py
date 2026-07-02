"""FastAPI backend — ses yükleme, analiz ve sinyal sorgulama endpoint'leri."""

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from src.agents.analyst import analyze_chunk, extract_stocks_from_result
from src.agents.chunker import process_chunks
from src.qdrant.client import get_client, get_or_create_collection
from src.qdrant.searcher import search_by_stock, search_filtered
from src.qdrant.uploader import upload_chunk_results
from src.transcription.transcriber import add_overlap, transcribe_streaming
from src.utils.logger import get_logger
from src.verification.agent import verify_signal

load_dotenv()
logger = get_logger(__name__)

# ── Session kalıcı depolama ───────────────────────────────────────────────────

_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:80] or "session"


def _session_dir(title: str) -> Path:
    return _SESSIONS_DIR / _slugify(title)


def _save_chunks(title: str, chunks: list[dict]) -> None:
    d = _session_dir(title)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _update_meta(title, {"chunk_count": len(chunks)})
    logger.info("Session chunks kaydedildi: %s (%d chunk)", title, len(chunks))


def _save_signals(title: str, signals: list[dict], done_chunks: list[str]) -> None:
    d = _session_dir(title)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"signals": signals, "done_chunks": done_chunks}
    (d / "signals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _update_meta(title, {"signal_count": len(signals), "done_chunks": len(done_chunks)})


def _update_meta(title: str, extra: dict) -> None:
    d = _session_dir(title)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"title": title, "updated_at": datetime.now().isoformat()})
    meta.update(extra)
    if "created_at" not in meta:
        meta["created_at"] = meta["updated_at"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_sessions() -> list[dict]:
    sessions = []
    for d in sorted(_SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["slug"] = d.name
        meta["has_chunks"] = (d / "chunks.json").exists()
        meta["has_signals"] = (d / "signals.json").exists()
        sessions.append(meta)
    return sessions

app = FastAPI(title="Borsa Analizi API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Job yönetimi ──────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.events: list[dict] = []          # tüm geçmiş olaylar
        self.cancelled = threading.Event()
        self.done = False
        self.error: str | None = None
        self._lock = threading.Lock()

    def add(self, event_type: str, data: dict):
        with self._lock:
            self.events.append({"type": event_type, "data": data})

    def snapshot(self, cursor: int) -> dict:
        with self._lock:
            new_events = self.events[cursor:]
            return {
                "done": self.done,
                "error": self.error,
                "cancelled": self.cancelled.is_set(),
                "total_events": len(self.events),
                "events": new_events,
            }


_jobs: dict[str, Job] = {}


def _new_job() -> Job:
    jid = str(uuid.uuid4())[:8]
    job = Job(jid)
    _jobs[jid] = job
    return job


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    get_or_create_collection()
    logger.info("API hazır")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Transkripsiyon job ────────────────────────────────────────────────────────

def _run_transcription(job: Job, audio_path: str, title: str, resume_from: int = 0):
    try:
        job.add("progress", {"mesaj": "Ses dosyası hazırlanıyor...", "yuzde": 1})
        all_chunks: list[dict] = []

        def on_prog(step: str, idx: int, total: int, dakika: int):
            if step == "ffmpeg":
                yuzde = int(2 + (idx / max(total, 1)) * 90)
                job.add("progress", {
                    "mesaj": f"🎬 Parça {idx+1}/{total} ses kesiliyor ({dakika}. dakika)...",
                    "yuzde": yuzde,
                })
            elif step == "whisper":
                yuzde = int(3 + (idx / max(total, 1)) * 90)
                job.add("progress", {
                    "mesaj": f"🎙 Parça {idx+1}/{total} Whisper'a gönderildi, bekleniyor (~2 dk)...",
                    "yuzde": yuzde,
                })

        for ev in transcribe_streaming(
            audio_path, chunk_minutes=10,
            cancelled=job.cancelled, on_progress=on_prog,
            resume_from=resume_from,
        ):
            if job.cancelled.is_set():
                break

            chunk_idx = ev["chunk_idx"]
            total = ev["total_chunks"]
            yuzde = int(5 + (chunk_idx / max(total, 1)) * 90)
            dakika = int(ev["start_sec"] // 60)

            job.add("progress", {
                "mesaj": f"Chunk {chunk_idx + 1}/{total} transkribe edildi ({dakika}. dakika)",
                "yuzde": yuzde,
                "chunk_idx": chunk_idx,
                "total_chunks": total,
            })
            job.add("chunk", {
                "chunk_id": ev["chunk_id"],
                "chunk_idx": chunk_idx,
                "total_chunks": total,
                "start_sec": ev["start_sec"],
                "end_sec": ev["end_sec"],
                "text": ev["text"],
                "word_count": ev["word_count"],
                "from_cache": ev.get("from_cache", False),
                "dakika": dakika,
            })
            all_chunks.append(ev)

        if job.cancelled.is_set():
            job.add("cancelled", {"mesaj": "Transkripsiyon durduruldu"})
        else:
            # Chunk'ları diske kaydet
            save_list = [
                {"chunk_id": c["chunk_id"], "start_sec": c["start_sec"],
                 "end_sec": c["end_sec"], "text": c["text"],
                 "word_count": c.get("word_count", 0)}
                for c in all_chunks
            ]
            _save_chunks(title, save_list)
            job.add("done", {
                "mesaj": "Transkripsiyon tamamlandı",
                "yuzde": 100,
                "toplam_chunk": len(all_chunks),
                "audio_path": audio_path,
                "title": title,
            })

    except Exception as exc:
        logger.error("Transkripsiyon job hatası: %s", exc)
        job.error = str(exc)
        job.add("error", {"mesaj": str(exc)})
    finally:
        job.done = True


@app.post("/jobs/transcribe")
async def start_transcription(
    file: UploadFile = File(...),
    title: str = Form("bilinmiyor"),
    resume_from: int = Form(0),
):
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job = _new_job()
    t = threading.Thread(target=_run_transcription, args=(job, tmp_path, title, resume_from), daemon=True)
    t.start()
    return {"job_id": job.job_id}


# ── Analiz job ────────────────────────────────────────────────────────────────

def _run_analysis(job: Job, raw_chunks: list[dict], title: str, skip_chunks: set[str] | None = None):
    try:
        job.add("progress", {"mesaj": f"{len(raw_chunks)} chunk analiz için hazırlanıyor...", "yuzde": 1})

        overlapped = add_overlap(raw_chunks, overlap_words=300)
        chunks = process_chunks(overlapped)
        toplam = len(chunks)

        get_or_create_collection()

        seen_stocks: list[str] = []
        tum_hisseler: set[str] = set()
        tum_sinyaller: list[dict] = []
        basarili = 0

        for i, chunk in enumerate(chunks):
            if job.cancelled.is_set():
                break

            chunk_id = chunk.get("chunk_id", f"{i:02d}")
            yuzde = int(5 + (i / max(toplam, 1)) * 90)
            dakika = int(chunk.get("start_sec", 0) // 60)

            # Daha önce yapılmış chunk'ı atla
            if skip_chunks and chunk_id in skip_chunks:
                job.add("chunk_skipped", {"chunk_id": chunk_id, "chunk_idx": i, "total_chunks": toplam})
                continue

            job.add("progress", {
                "mesaj": f"Chunk {chunk_id} analiz ediliyor ({i+1}/{toplam}, {dakika}. dakika)...",
                "yuzde": yuzde,
                "chunk_idx": i,
                "total_chunks": toplam,
            })

            try:
                analysis = analyze_chunk(chunk, seen_stocks)
                upload_chunk_results(chunk, analysis, title)

                yeni = extract_stocks_from_result(analysis)
                seen_stocks = list(dict.fromkeys(seen_stocks + yeni))
                tum_hisseler.update(yeni)

                sinyaller = analysis.get("sinyaller", [])
                for s in sinyaller:
                    s["chunk_id"] = chunk_id
                tum_sinyaller.extend(sinyaller)
                basarili += 1

                job.add("chunk_done", {
                    "chunk_id": chunk_id,
                    "chunk_idx": i,
                    "total_chunks": toplam,
                    "dakika": dakika,
                    "hisseler": yeni,
                    "sinyal_sayisi": len(sinyaller),
                    "sinyaller": sinyaller,
                    "genel_yorum": analysis.get("genel_yorum", ""),
                })
                # Anlık diske kaydet — durdurulsa bile sinyal kaybolmasın
                _save_signals(title, tum_sinyaller,
                              [c.get("chunk_id", f"{j:02d}") for j, c in enumerate(chunks) if j <= i
                               and c.get("chunk_id") not in (skip_chunks or set())])

            except Exception as exc:
                logger.error("Chunk %s analiz hatası: %s", chunk_id, exc)
                job.add("chunk_error", {"chunk_id": chunk_id, "mesaj": str(exc)})

        if job.cancelled.is_set():
            job.add("cancelled", {"mesaj": "Analiz durduruldu"})
        else:
            job.add("done", {
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
        logger.error("Analiz job hatası: %s", exc)
        job.error = str(exc)
        job.add("error", {"mesaj": str(exc)})
    finally:
        job.done = True


@app.post("/jobs/analyze")
async def start_analysis(
    chunks_json: str = Form(...),   # JSON string: [{chunk_id, start_sec, end_sec, text}, ...]
    title: str = Form("bilinmiyor"),
    skip_chunks: str = Form(""),
):
    try:
        raw_chunks: list[dict] = json.loads(chunks_json)
    except Exception:
        raise HTTPException(status_code=400, detail="chunks_json geçersiz JSON")

    skip_set = {c.strip() for c in skip_chunks.split(",") if c.strip()}
    job = _new_job()
    t = threading.Thread(target=_run_analysis, args=(job, raw_chunks, title, skip_set), daemon=True)
    t.start()
    return {"job_id": job.job_id}


# ── Job poll + iptal ──────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/poll")
async def poll_job(job_id: str, cursor: int = Query(0)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    return job.snapshot(cursor)


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    job.cancelled.set()
    return {"cancelled": True}


# ── Session endpoint'leri ─────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    return {"sessions": _list_sessions()}


@app.get("/sessions/{slug}")
async def get_session(slug: str):
    d = _SESSIONS_DIR / slug
    if not d.exists():
        raise HTTPException(status_code=404, detail="Session bulunamadı")
    result: dict = {}
    chunks_path = d / "chunks.json"
    if chunks_path.exists():
        result["chunks"] = json.loads(chunks_path.read_text(encoding="utf-8"))
    signals_path = d / "signals.json"
    if signals_path.exists():
        data = json.loads(signals_path.read_text(encoding="utf-8"))
        result["signals"] = data.get("signals", [])
        result["done_chunks"] = data.get("done_chunks", [])
    meta_path = d / "meta.json"
    if meta_path.exists():
        result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


@app.delete("/sessions/{slug}")
async def delete_session(slug: str):
    import shutil
    d = _SESSIONS_DIR / slug
    if not d.exists():
        raise HTTPException(status_code=404, detail="Session bulunamadı")
    shutil.rmtree(d)
    return {"deleted": slug}


# ── Mevcut endpoint'ler ───────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), title: str = "bilinmiyor"):
    from orchestrator import run_pipeline
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        return run_pipeline(tmp_path, title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/search")
async def search(
    q: str | None = Query(None),
    hisse: str | None = Query(None),
    sinyal_tipi: str | None = Query(None),
    guven: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    try:
        if q and q.strip():
            results = search_filtered(query=q, hisse=hisse, sinyal_tipi=sinyal_tipi, guven=guven, limit=limit)
        else:
            from src.qdrant.searcher import scroll_all
            results = scroll_all(hisse=hisse, sinyal_tipi=sinyal_tipi, guven=guven, limit=limit)
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


@app.delete("/signals/all")
async def clear_all_signals():
    """Qdrant koleksiyonundaki tüm sinyalleri siler ve koleksiyonu yeniden oluşturur."""
    from src.qdrant.client import get_client
    collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
    client = get_client()
    client.delete_collection(collection)
    get_or_create_collection()
    logger.info("Tüm sinyaller silindi, koleksiyon yeniden oluşturuldu")
    return {"cleared": True}


@app.get("/verify")
async def verify(hisse: str = Query(...)):
    try:
        sinyaller = search_by_stock(hisse, limit=50)
        ilgili = [s for s in sinyaller if s.get("sinyal_tipi") in ("alım", "satım")]
        results = [verify_signal(s) for s in ilgili]
        return {"hisse": hisse, "count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/verify/bulk")
async def verify_bulk(body: dict):
    """Birden fazla hisse kodunun anlık fiyatını paralel olarak çeker."""
    import asyncio
    from src.qdrant.searcher import scroll_all

    hisseler = body.get("hisseler", [])
    if not hisseler:
        return {"count": 0, "results": []}

    # Tüm hisselerin sinyallerini önce topla
    tum_sinyaller = []
    for h in hisseler:
        sinyaller = scroll_all(hisse=h, limit=50)
        tum_sinyaller.extend([s for s in sinyaller if s.get("fiyat") is not None])

    if not tum_sinyaller:
        return {"count": 0, "results": []}

    # Her hisse için fiyat çekimini paralel yap (thread pool)
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()

    try:
        with ThreadPoolExecutor(max_workers=min(len(hisseler), 10)) as pool:
            futures = [loop.run_in_executor(pool, verify_signal, s) for s in tum_sinyaller]
            results = await asyncio.gather(*futures)
        return {"count": len(results), "results": list(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/progress")
async def progress(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"pong: {data}")
    except Exception:
        pass
