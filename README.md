# Türkçe Borsa Analizi Multi-Agent Sistemi

Uzun Türkçe borsa analiz ses dosyalarını (95+ dakika) otonom olarak işleyen sistem. MLX Whisper ile sesi metne çevirir, LangGraph ajanları aracılığıyla finansal sinyalleri (hisse adı, alım/satım seviyeleri, teknik terimler) çıkarır ve Qdrant vektör veritabanında saklar. FastAPI backend ve Streamlit UI ile sunulur.

---

## Kurulum

### Gereksinimler

- Python 3.12
- Docker (Qdrant için)
- Apple Silicon Mac (MLX Whisper için)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/desideroo/finansal-ajan.git
cd finansal-ajan

# 2. Sanal ortam oluştur ve aktif et
python -m venv .venv
source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. .env dosyasını oluştur
cp .env.example .env
# .env içine API key'lerini gir:
#   GOOGLE_API_KEY=...
#   OPENAI_API_KEY=...

# 5. Qdrant'ı başlat
docker-compose up -d
```

---

## Kullanım

### CLI — Tek Komutla Pipeline

```bash
python orchestrator.py audio.m4a "Video Başlığı"
```

Çıktı:
```json
{
  "audio_path": "audio.m4a",
  "video_title": "Video Başlığı",
  "toplam_chunk": 10,
  "basarili_chunk": 10,
  "basarisiz_chunk": 0,
  "toplam_sinyal": 47,
  "islenen_hisseler": ["THYAO", "GARAN", "AKBNK"],
  "sure_saniye": 312.4
}
```

### API Sunucusu

```bash
uvicorn api.main:app --reload --port 8000
```

### Streamlit UI

```bash
streamlit run ui/app.py
```

---

## API Endpoint'leri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Sağlık kontrolü |
| POST | `/analyze` | Ses dosyası yükle ve analiz et |
| GET | `/search?q=THYAO+alım` | Hybrid semantik + metadata arama |
| GET | `/stocks` | Kayıtlı tüm hisse kodları |
| GET | `/verify?hisse=THYAO` | Anlık fiyat doğrulaması |
| WS | `/progress` | İlerleme bildirimi (WebSocket) |

Swagger UI: http://localhost:8000/docs

---

## Teknoloji Yığını

| Katman | Teknoloji | Detay |
|--------|-----------|-------|
| Transkripsiyon | mlx-whisper | `mlx-community/whisper-large-v3-mlx` |
| Ajan Çerçevesi | LangGraph + LangChain | Multi-agent pipeline |
| Ana LLM | Google Gemini 2.0 Flash | Sinyal çıkarımı |
| Yedek LLM | OpenAI GPT-4o-mini | Gemini başarısız olursa |
| Embedding | BGE-M3 (BAAI/bge-m3) | Dense + sparse hybrid, 1024 dim |
| Vektör DB | Qdrant | Docker, localhost:6333 |
| Backend | FastAPI + uvicorn | REST + WebSocket |
| UI | Streamlit | — |
| Fiyat Verisi | borsapy | Anlık BIST fiyatları |

---

## Proje Yapısı

```
finansal-ajan/
├── src/
│   ├── transcription/   # MLX Whisper + prompt sistemi
│   ├── agents/          # Chunker (Ajan 1) + Analyst (Ajan 2)
│   ├── qdrant/          # Bağlantı, yükleme, arama
│   ├── utils/           # Logger, API sarmalayıcı
│   └── verification/    # Anlık fiyat doğrulama
├── api/                 # FastAPI backend
├── ui/                  # Streamlit arayüzü
├── orchestrator.py      # Ana pipeline koordinatörü
└── docker-compose.yml   # Qdrant servisi
```
