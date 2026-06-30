# Türkçe Borsa Analizi Multi-Agent Sistemi

## Proje Amacı

Uzun Türkçe borsa analiz ses dosyalarını (95+ dakika) otonom olarak işleyen sistem:
1. MLX Whisper ile ses → metin (Türkçe)
2. LangGraph ajanları ile finansal sinyal çıkarımı
3. Qdrant vektör DB'ye kayıt
4. FastAPI backend + Streamlit UI ile sunum

---

## Teknoloji Yığını

| Katman | Teknoloji | Detay |
|--------|-----------|-------|
| Transkripsiyon | mlx-whisper | checkpoint: `mlx-community/whisper-large-v3-mlx` |
| Ajan Çerçevesi | langgraph + langchain-core | — |
| Ana LLM | Google Gemini 2.0 Flash | `gemini-2.0-flash`, max_output_tokens=1000 |
| Yedek LLM | OpenAI GPT-4o-mini | `gpt-4o-mini`, max_tokens=1000, json_object |
| Vektör DB | Qdrant (Docker) | localhost:6333, koleksiyon: `finansal_analiz` |
| Embedding | BGE-M3 (BAAI/bge-m3) | Lokal, dense+sparse hybrid, dim=1024 |
| Backend | FastAPI + uvicorn | — |
| UI | Streamlit | — |
| Fiyat Verisi | borsapy | `bp.Ticker("THYAO").fast_info["last_price"]` |
| Python | 3.12 | .venv proje kökünde |

**Önemli:** Fiyat verisi için `borsapy` kullanılır, `yfinance` KULLANILMAZ. Ticker'larda `.IS` suffix yok.

---

## Model Seçimleri

- **Transkripsiyon:** `mlx-community/whisper-large-v3-mlx` (turbo değil)
  - language="tr" (zorunlu, otomatik tespite bırakma)
  - vad_filter=True, word_timestamps=True, temperature=0.0
- **Ana LLM:** `gemini-2.0-flash` (GOOGLE_API_KEY)
- **Yedek LLM:** `gpt-4o-mini` (OPENAI_API_KEY)
- **Embedding:** `BAAI/bge-m3` (lokal, ücretsiz)
- **Ticker Kaynağı:** `bist_hisseler.json` (proje kökünde)

---

## Mimari Kurallar

1. **Modülerlik:** Ses işleme, ajan mantığı, DB işlemleri ayrı klasörlerde. Asla aynı dosyada birleştirme.
2. **Ajan İletişimi:** Ajanlar yalnızca JSON ile haberleşir. Hiçbir ajan diğerinin iç yapısını bilmez.
3. **API Çağrıları:** Tüm LLM çağrıları `utils/api_helpers.py` içindeki `safe_llm_call()` üzerinden geçer. Gemini başarısız → otomatik GPT-4o-mini. tenacity: stop_after_attempt(3), wait_exponential(min=4, max=30). Her çağrı öncesi time.sleep(1).
4. **Ajan 1 Kuralı:** Chunker ajanı (Ajan 1) asla LLM çağrısı yapmaz. Sadece Python + Whisper segmentleri kullanır.
5. **Hata Yönetimi:** Her modül kendi hatalarını yakalar ve loglar. Bir chunk başarısız olsa bile diğerleri işlenir.
6. **Konfigürasyon:** Tüm key ve sabitler `.env`'den gelir. Hiçbir hard-coded key veya path olmaz.

---

## Chunking Stratejisi — 3 Katmanlı Bağlam Koruması

**Problem:** Analistin cümlesi chunk sınırında kesilirse bir sonraki chunk hangi hisse için söylendiğini bilemez.

### Katman 1 — Segment Sınırlarında Kes
Whisper her cümle için segment döndürür (start, end, text). Chunk'lar bu segmentleri biriktirerek oluşturulur. Chunk boyutu dinamik hesaplanır, max 30 dk, `calculate_chunk_minutes()` (`src/agents/chunker.py`). Asla bir segmentin ortasında bölme yapma.

### Katman 2 — 300 Kelime Overlap
Her chunk'a önceki chunk'ın son 300 kelimesi "context_prefix" olarak eklenir. Bu veri Qdrant'a kaydedilmez, sadece prompt'ta kullanılır.

### Katman 3 — seen_stocks Inject
Her prompt'a o ana kadar bahsedilen hisse listesi eklenir. Model bu listeyi kullanarak hisse adı geçmeyen cümleleri son bahsedilen hisseye bağlar.

---

## Prompt Mimarisi

### WHISPER_INITIAL_PROMPT
- **Sadece Whisper için** (`src/transcription/prompts.py`)
- Amaç: Whisper'ın BIST kodlarını yanlış yazmaması
- ~70 token, kısa ve doğal metin formatında (liste değil)
- Whisper son 224 tokenı işler, kısa tutulmalı

### build_analyst_system_prompt()
- **Sadece Ajan 2 (analyst.py) için**
- `bist_hisselers.json`'dan yüklenen ticker seti alır
- Token limiti yok, tüm kurallar buraya girer
- Geçerli BIST kodlarını içerir (yanlış duyulan kodlar düzeltilebilsin)

---

## Ajan 2 JSON Çıktı Şeması

```json
{
  "chunk_id": "03",
  "bahsedilen_hisseler": ["THYAO", "GARAN"],
  "sinyaller": [
    {
      "hisse": "THYAO",
      "sinyal_tipi": "alım",
      "fiyat": 45.50,
      "para_birimi": "TL",
      "gerekce": "analistin tam ifadesi",
      "kaynak_cumle": "metinden birebir alıntı",
      "guven": "yuksek"
    }
  ],
  "teknik_terimler": ["RSI", "MACD"],
  "genel_yorum": "bu chunk'ın tek cümle özeti"
}
```

- `sinyal_tipi`: `alım | satım | stop_loss | destek | direnc | genel_yorum`
- `guven`: `yuksek | orta | dusuk`
- Belirsiz hisseler `"belirsiz"` olarak işaretlenir, Qdrant'a kaydedilmez

---

## Qdrant Metadata Şeması

```json
{
  "hisse":        "THYAO",
  "sinyal_tipi":  "alım",
  "fiyat":        45.50,
  "para_birimi":  "TL",
  "guven":        "yuksek",
  "kaynak_cumle": "...",
  "chunk_id":     "03",
  "zaman_sn":     1800,
  "video_title":  "...",
  "created_at":   "ISO datetime string"
}
```

---

## Dosya Yapısı

```
finansal-ajan/
├── src/
│   ├── __init__.py
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── transcriber.py
│   │   └── prompts.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── analyst.py
│   ├── qdrant/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── uploader.py
│   │   └── searcher.py
│   └── utils/
│       ├── __init__.py
│       ├── api_helpers.py
│       └── logger.py
├── api/
│   ├── __init__.py
│   └── main.py
├── ui/
│   └── app.py
├── orchestrator.py
├── CLAUDE.md
├── bist_hisseler.json
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
└── requirements.txt
```

---

## Modül Tamamlanma Durumu

- [x] `src/transcription/transcriber.py`
- [x] `src/transcription/prompts.py`
- [x] `src/agents/chunker.py`
- [x] `src/agents/analyst.py`
- [x] `src/qdrant/client.py`
- [x] `src/qdrant/uploader.py`
- [x] `src/qdrant/searcher.py`
- [x] `src/utils/api_helpers.py`
- [x] `src/utils/logger.py`
- [x] `api/main.py`
- [x] `ui/app.py`
- [x] `orchestrator.py`
- [x] `src/verification/agent.py`

Tüm modüller tamamlandı — proje production-ready.

---

## Yeni Oturum Şablonu

```
CLAUDE.md'yi oku. Modül durumlarını kontrol et.
Bugünkü görev: [MODÜL]. Bitince CLAUDE.md'yi
güncelle ve git commit at.
```

### Gün 3-4 — Transkripsiyon
- `src/transcription/transcriber.py`
- MLX Whisper large-v3, segment bazlı chunking, VAD filter, word_timestamps, JSON çıktı formatı
