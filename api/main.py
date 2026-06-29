"""FastAPI backend — ses yükleme, analiz ve sinyal sorgulama endpoint'leri."""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import run_pipeline
from src.qdrant.client import get_client, get_or_create_collection
from src.qdrant.searcher import search_by_stock, search_filtered
from src.utils.logger import get_logger
from src.verification.agent import verify_signal

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="Borsa Analizi API", version="1.0.0")

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


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    title: str = "bilinmiyor",
):
    """Ses dosyasını yükle, pipeline'ı çalıştır, özet döndür."""
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
    q: str = Query(..., description="Semantik arama sorgusu"),
    hisse: str | None = Query(None),
    sinyal_tipi: str | None = Query(None),
    guven: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    """Hybrid semantic + metadata filtreli arama."""
    try:
        results = search_filtered(
            query=q,
            hisse=hisse,
            sinyal_tipi=sinyal_tipi,
            guven=guven,
            limit=limit,
        )
        return {"results": results, "count": len(results)}
    except Exception as exc:
        logger.error("Arama hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stocks")
async def stocks():
    """Qdrant'taki tüm benzersiz hisse kodlarını alfabetik döndürür."""
    try:
        collection = os.getenv("QDRANT_COLLECTION", "finansal_analiz")
        client = get_client()
        hisseler: set[str] = set()
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                offset=offset,
                limit=100,
                with_payload=["hisse"],
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
        logger.error("Hisse listesi hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/verify")
async def verify(hisse: str = Query(..., description="BIST ticker kodu")):
    """Hissenin Qdrant'taki alım/satım sinyallerini anlık fiyatla karşılaştırır."""
    try:
        sinyaller = search_by_stock(hisse, limit=50)
        ilgili = [s for s in sinyaller if s.get("sinyal_tipi") in ("alım", "satım")]
        results = [verify_signal(s) for s in ilgili]
        return {"hisse": hisse, "count": len(results), "results": results}
    except Exception as exc:
        logger.error("Doğrulama hatası (%s): %s", hisse, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/progress")
async def progress(websocket: WebSocket):
    """İlerleme bildirimi için WebSocket iskeleti (ileriki geliştirme)."""
    await websocket.accept()
    await websocket.send_text("Bağlantı kuruldu")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"pong: {data}")
    except Exception:
        pass
