#!/bin/bash
# Borsa Analizi — Tek tıkla başlatıcı
# Terminal açık kaldığı sürece API logları burada görünür.
# Kapatmak için terminali kapat.

set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"
ST_LOG="$PROJECT_DIR/logs/streamlit.log"
mkdir -p "$PROJECT_DIR/logs"

echo "========================================"
echo "  Borsa Analizi — Başlatılıyor"
echo "========================================"

# ── 1. Docker Desktop ────────────────────────────────────────────────────────
if ! docker info &>/dev/null; then
    echo "[1/4] Docker açılıyor..."
    open -a "Docker Desktop" 2>/dev/null || open -a "Docker" 2>/dev/null || true
    echo "      Docker başlaması bekleniyor (max 60 sn)..."
    for i in $(seq 1 30); do
        sleep 2
        if docker info &>/dev/null; then
            echo "      ✓ Docker hazır"
            break
        fi
        [ $i -eq 30 ] && echo "      ⚠️  Docker 60 sn'de başlamadı"
    done
else
    echo "[1/4] ✓ Docker zaten çalışıyor"
fi

# ── 2. Qdrant ────────────────────────────────────────────────────────────────
echo "[2/4] Qdrant başlatılıyor..."
if docker ps --filter "name=qdrant" --format "{{.Names}}" | grep -q "qdrant"; then
    echo "      ✓ Qdrant zaten çalışıyor"
else
    if docker ps -a --filter "name=qdrant" --format "{{.Names}}" | grep -q "qdrant"; then
        docker start qdrant > /dev/null 2>&1
    else
        docker compose up -d qdrant > /dev/null 2>&1
    fi
    sleep 3
    echo "      ✓ Qdrant başlatıldı"
fi

# ── 3. Streamlit (arka planda, sessiz) ───────────────────────────────────────
echo "[3/4] Streamlit başlatılıyor (arka plan)..."
source "$VENV"

EXISTING_ST=$(lsof -ti:8501 2>/dev/null || true)
[ -n "$EXISTING_ST" ] && kill "$EXISTING_ST" 2>/dev/null; sleep 1

streamlit run "$PROJECT_DIR/ui/app.py" \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$ST_LOG" 2>&1 &
ST_PID=$!

# Tarayıcıyı arka planda aç (5 sn sonra)
(sleep 5 && open "http://localhost:8501") &

echo "      ✓ Streamlit başlatıldı (PID: $ST_PID)"
echo ""

# ── 4. FastAPI — FOREGROUND (loglar burada görünür) ──────────────────────────
echo "[4/4] API başlatılıyor — loglar aşağıda:"
echo "========================================"
echo "  Uygulama: http://localhost:8501"
echo "  API docs: http://localhost:8000/docs"
echo "  Kapatmak için: Ctrl+C"
echo "========================================"
echo ""

EXISTING_API=$(lsof -ti:8000 2>/dev/null || true)
[ -n "$EXISTING_API" ] && kill "$EXISTING_API" 2>/dev/null; sleep 1

exec python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
