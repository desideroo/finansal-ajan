#!/bin/bash
# Borsa Analizi — Tek tıkla başlatıcı
# Bu dosyayı Dock'a veya masaüstüne kısayol olarak ekleyin.

set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"
LOG="$PROJECT_DIR/logs/launcher.log"
mkdir -p "$PROJECT_DIR/logs"

echo "========================================" | tee -a "$LOG"
echo "$(date) — Başlatılıyor..." | tee -a "$LOG"

# ── 1. Docker Desktop'ı aç ────────────────────────────────────────────────────
echo "Docker açılıyor..." | tee -a "$LOG"
if ! docker info &>/dev/null; then
    open -a "Docker Desktop" 2>/dev/null || open -a "Docker" 2>/dev/null || true
    echo "Docker başlaması bekleniyor..." | tee -a "$LOG"
    for i in $(seq 1 30); do
        sleep 2
        if docker info &>/dev/null; then
            echo "Docker hazır ($((i*2)) sn)" | tee -a "$LOG"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "UYARI: Docker 60 saniyede başlamadı, devam ediliyor..." | tee -a "$LOG"
        fi
    done
else
    echo "Docker zaten çalışıyor." | tee -a "$LOG"
fi

# ── 2. Qdrant başlat ─────────────────────────────────────────────────────────
echo "Qdrant başlatılıyor..." | tee -a "$LOG"
if docker ps --filter "name=qdrant" --format "{{.Names}}" | grep -q "qdrant"; then
    echo "Qdrant zaten çalışıyor." | tee -a "$LOG"
else
    if docker ps -a --filter "name=qdrant" --format "{{.Names}}" | grep -q "qdrant"; then
        docker start qdrant >> "$LOG" 2>&1
        echo "Qdrant yeniden başlatıldı." | tee -a "$LOG"
    else
        docker compose up -d qdrant >> "$LOG" 2>&1
        echo "Qdrant docker-compose ile başlatıldı." | tee -a "$LOG"
    fi
    sleep 3
fi

# ── 3. FastAPI backend başlat ─────────────────────────────────────────────────
echo "FastAPI backend başlatılıyor..." | tee -a "$LOG"
source "$VENV"

# Zaten çalışıyorsa öldür
EXISTING_PID=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Port 8000'deki mevcut process kapatılıyor (PID: $EXISTING_PID)..." | tee -a "$LOG"
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

cd "$PROJECT_DIR"
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> "$LOG" 2>&1 &
API_PID=$!
echo "FastAPI başlatıldı (PID: $API_PID)" | tee -a "$LOG"

# API hazır olana kadar bekle
echo "API hazırlanıyor..." | tee -a "$LOG"
for i in $(seq 1 20); do
    sleep 1
    if curl -s http://localhost:8000/health | grep -q "ok"; then
        echo "API hazır ($i sn)" | tee -a "$LOG"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "UYARI: API 20 saniyede yanıt vermedi." | tee -a "$LOG"
    fi
done

# ── 4. Streamlit UI başlat ───────────────────────────────────────────────────
echo "Streamlit başlatılıyor..." | tee -a "$LOG"

EXISTING_ST=$(lsof -ti:8501 2>/dev/null || true)
if [ -n "$EXISTING_ST" ]; then
    kill "$EXISTING_ST" 2>/dev/null || true
    sleep 1
fi

nohup streamlit run "$PROJECT_DIR/ui/app.py" \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    >> "$LOG" 2>&1 &
ST_PID=$!
echo "Streamlit başlatıldı (PID: $ST_PID)" | tee -a "$LOG"

# Streamlit hazır olana kadar bekle
sleep 3
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:8501 | grep -q "streamlit"; then
        echo "Streamlit hazır ($i sn)" | tee -a "$LOG"
        break
    fi
done

# ── 5. Tarayıcıda aç ─────────────────────────────────────────────────────────
echo "Tarayıcı açılıyor..." | tee -a "$LOG"
sleep 1
open "http://localhost:8501"

echo "$(date) — Başlatma tamamlandı. PID'ler: API=$API_PID, Streamlit=$ST_PID" | tee -a "$LOG"
echo ""
echo "✅ Uygulama hazır: http://localhost:8501"
echo "   Kapatmak için bu terminali kapatın veya:"
echo "   kill $API_PID $ST_PID"
