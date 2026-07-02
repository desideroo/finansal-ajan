# Borsa Analizi Multi-Agent Sistemi

Uzun Türkçe borsa analiz ses kayıtlarını otomatik olarak transkript eden, içindeki finansal sinyalleri yapay zeka ile çıkaran ve vektör veritabanına kaydederek aranabilir hale getiren uçtan uca bir sistemdir.

---

## İçindekiler

1. [Sistem Mimarisi](#sistem-mimarisi)
2. [Nasıl Çalışır](#nasıl-çalışır)
3. [Gereksinimler](#gereksinimler)
4. [Kurulum](#kurulum)
5. [Yapılandırma](#yapılandırma)
6. [Sistemi Başlatma](#sistemi-başlatma)
7. [Arayüz Genel Bakış](#arayüz-genel-bakış)
8. [Kullanım Kılavuzu](#kullanım-kılavuzu)
9. [Proje Yapısı](#proje-yapısı)
10. [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)

---

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                        KULLANICI (Tarayıcı)                     │
│                     http://localhost:8501                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                      Streamlit UI (ui/app.py)                   │
│  • Ses dosyası yükleme    • Transkript görüntüleme              │
│  • Analiz başlatma        • Sinyal kartları                     │
│  • Oturum yönetimi        • Vektör arama                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (localhost:8000)
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Backend (api/main.py)              │
│                                                                 │
│   /jobs/transcribe          /jobs/analyze                       │
│   /jobs/{id}/poll           /sessions                           │
│   /signals/search           /signals/all                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AJAN 1 — Chunker                      │  │
│  │  mlx-whisper ile transkripsiyon → segment bazlı chunk    │  │
│  │  • Dinamik chunk boyutu (max 30 dk)                      │  │
│  │  • 300 kelime örtüşme (bağlam kaybı önleme)             │  │
│  │  • Durdur / Devam Et desteği                             │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ chunk listesi                         │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │                    AJAN 2 — Analyst                      │  │
│  │  Gemini 2.5 Flash (yedek: GPT-4o-mini)                  │  │
│  │  • Finansal sinyal çıkarımı (alım/satım/destek/direnc…)  │  │
│  │  • Hisse kodu eşleştirme (Whisper hataları dahil)        │  │
│  │  • Güven skoru ataması                                   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ sinyal JSON                           │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │                  AJAN 3 — Uploader                       │  │
│  │  BGE-M3 embedding → Qdrant upsert                        │  │
│  │  • Dense + sparse hibrit vektör                          │  │
│  │  • Duplikat önleme (fiyat + hisse + tip bazlı)           │  │
│  └──────────────────────┬───────────────────────────────────┘  │
└───────────────────────  │  ──────────────────────────────────── ┘
                          │ HTTP (localhost:6333)
┌─────────────────────────▼───────────────────────────────────────┐
│                    Qdrant Vektör DB (Docker)                    │
│              Koleksiyon: finansal_analiz                        │
│         Dense (1024 boyut) + Sparse hibrit arama               │
└─────────────────────────────────────────────────────────────────┘
```

### Veri Akışı

```
Ses Dosyası (.m4a / .mp3 / .wav)
        │
        ▼
[1] mlx-whisper transkripsiyon
    → segment listesi (start, end, text)
        │
        ▼
[2] Chunk birleştirme
    → dinamik boyutlu metin parçaları
    → her parçaya önceki 300 kelime bağlam eklenir
        │
        ▼  (her chunk için sırayla)
[3] Gemini 2.5 Flash analizi
    → JSON: hisse, sinyal_tipi, fiyat, güven, kaynak_cumle
        │
        ▼
[4] BGE-M3 embedding + Qdrant kayıt
    → duplikat kontrol → arama yapılabilir vektör
        │
        ▼
[5] Streamlit UI'da görüntüleme ve arama
```

---

## Nasıl Çalışır

### Transkripsiyon (Ajan 1)

Ses dosyası `mlx-whisper` ile Türkçe olarak yazıya dökülür. Model olarak `mlx-community/whisper-large-v3-mlx` kullanılır. Whisper'ın döndürdüğü segment sınırları korunarak metin parçalara (chunk) bölünür — hiçbir cümlenin ortasından kesilme yapılmaz.

**Dinamik chunk boyutu:** Kayıt 30 dakikadan kısaysa tek parça; daha uzunsa eşit parçalara bölünür (her biri en fazla 30 dakika). 95 dakikalık bir kayıt 4 parçaya ayrılır.

**Bağlam örtüşmesi:** Analist bir cümleye bir parçanın sonunda başlayıp diğerinde bitirdiğinde hisse adı kaybolabilir. Bunu önlemek için her parçanın başına önceki parçanın son 300 kelimesi eklenir. Bu bölüm Qdrant'a kaydedilmez, yalnızca model bağlamı içindir.

**Durdur / Devam Et:** Transkripsiyon istediğiniz an durdurulabilir. Devam edildiğinde kaldığı chunk'tan devam eder.

### Finansal Sinyal Analizi (Ajan 2)

Her chunk Gemini 2.5 Flash modeline gönderilir. Model şu sinyalleri çıkarır:

| Sinyal Tipi | Ne Zaman Kullanılır |
|-------------|---------------------|
| `alım` | Net alım tavsiyesi ("kesinlikle alın", "portföye ekle") |
| `satım` | Koşulsuz ve fiyatlı çıkış tavsiyesi |
| `stop_loss` | Koşullu zarar kes ("X'in altına inerse sat") |
| `destek` | Teknik destek seviyesi |
| `direnc` | Teknik direnç seviyesi (hedef fiyat dahil) |
| `genel_yorum` | Diğer tüm yorumlar |

**Güven skoru:** Her sinyal için `yuksek`, `orta` veya `dusuk` güven atanır. Kesin ifadeler yüksek, koşullu veya belirsiz ifadeler orta/düşük güven alır.

**Whisper hata düzeltme:** Whisper bazı şirket adlarını yanlış yazabilir (ses benzerliği, ünsüz düşmesi vb.). Sistem, `bist_hisseler.json`'daki alias tablosu ve LLM'in ses benzerliği kurallarıyla bu hataları otomatik düzeltir.

**Hata toleransı:** Gemini başarısız olursa sistem otomatik olarak GPT-4o-mini'ye geçer. Tenacity ile her model için 3 deneme yapılır.

### Vektör Kaydı ve Arama (Ajan 3)

Her sinyal BGE-M3 modeli ile hem dense (1024 boyut) hem de sparse vektöre dönüştürülür. Bu hibrit yaklaşım hem semantik benzerliği hem de anahtar kelime eşleşmesini destekler.

Kayıt öncesi duplikat kontrolü yapılır: aynı hisse + sinyal tipi + fiyat kombinasyonu zaten varsa kayıt atlanır.

### Doğrulama Ajanı — Gerçek Zamanlı Fiyat Kıyaslaması (Bonus Modül)

Sistem, analiz edilen her sinyali kaydettikten sonra otomatik olarak bir doğrulama ajanı (`src/verification/agent.py`) çalıştırır. Bu ajan:

1. Sinyaldeki hisse kodunu (`THYAO`, `GARAN` vb.) alır
2. `borsapy` kütüphanesi aracılığıyla BIST'ten o anki gerçek piyasa fiyatını çeker
3. Analistin söylediği fiyat ile mevcut piyasa fiyatını karşılaştırır

**Örnek çıktı:**

```
THYAO | Analist: 45.50 TL | Piyasa: 47.20 TL | Fark: +3.7% ↑
GARAN | Analist: 28.00 TL (destek) | Piyasa: 29.15 TL | Destek üstünde ✓
```

Bu kıyaslama UI'daki **Arama** sekmesinde, bir hisse seçili iken "✅ Fiyat Doğrula" butonuna basıldığında görüntülenir. Sonuçlar tablo formatında renk kodlu olarak listelenir: pozitif fark yeşil, negatif fark kırmızı.

### Ajanlar Arası İletişim (JSON Şeması)

Tüm ajanlar birbirleriyle yalnızca yapılandırılmış JSON ile haberleşir. Ajan 1'den Ajan 2'ye geçen chunk formatı:

```json
{
  "chunk_id": "02",
  "start_sec": 1800.0,
  "end_sec": 3600.0,
  "text": "...transkript metni...",
  "context_prefix": "...önceki 300 kelime..."
}
```

Ajan 2'den Ajan 3'e geçen sinyal formatı:

```json
{
  "chunk_id": "02",
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
  "genel_yorum": "chunk özeti"
}
```

Hiçbir ajan diğerinin iç yapısını bilmez; yalnızca bu JSON sözleşmesine göre çalışır.

---

## Gereksinimler

### Donanım

- **İşlemci:** Apple Silicon (M1/M2/M3/M4) — `mlx-whisper` yalnızca Apple Silicon'da çalışır
- **RAM:** En az 16 GB önerilir (Whisper large-v3 ~6 GB, BGE-M3 ~2 GB)
- **Disk:** En az 10 GB boş alan (modeller ilk çalıştırmada indirilir)

> ⚠️ **Önemli:** Bu sistem Apple Silicon Mac'e özeldir. Intel Mac veya Windows/Linux'ta `mlx-whisper` çalışmaz. Farklı bir platformda kullanmak için transkripsiyon katmanını `faster-whisper` gibi bir alternatifle değiştirmeniz gerekir.

### Yazılım

- **macOS:** Ventura (13) veya üzeri
- **Python:** 3.12 (kesinlikle 3.12 — diğer sürümlerde mlx uyumsuzluğu olabilir)
- **Docker Desktop:** Qdrant veritabanını çalıştırmak için gereklidir
- **Git:** Projeyi klonlamak için

### API Anahtarları

En az biri yeterlidir; ikisi birden varsa Gemini birincil, GPT-4o-mini yedek olarak çalışır:

- **Google Gemini API:** [aistudio.google.com](https://aistudio.google.com) → "Get API Key" (ücretsiz kota mevcut)
- **OpenAI API:** [platform.openai.com](https://platform.openai.com) → API Keys

---

## Kurulum

### Adım 1 — Docker Desktop Kurulumu

[docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) adresinden macOS için Docker Desktop'ı indirin ve kurun. Kurulumdan sonra uygulamayı açın; menü çubuğunda balina simgesi belirince Docker hazır demektir.

### Adım 2 — Python 3.12 Kurulumu

```bash
# Mevcut Python sürümünüzü kontrol edin
python3 --version
```

3.12 değilse [python.org/downloads](https://www.python.org/downloads/) adresinden Python 3.12'yi indirip kurun.

### Adım 3 — Projeyi İndirin

```bash
git clone https://github.com/KULLANICI_ADI/finansal-ajan.git
cd finansal-ajan
```

### Adım 4 — Sanal Ortam Oluşturun

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Komut isteminin başında `(.venv)` yazısı görünüyorsa başarılıdır.

### Adım 5 — Kütüphaneleri Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Internet hızınıza bağlı olarak 5-15 dakika sürebilir. `mlx-whisper` ve `FlagEmbedding` büyük kütüphanelerdir.

### Adım 6 — Modelleri Önceden İndirin (Opsiyonel ama Önerilir)

Transkripsiyon ve embedding modelleri ilk kullanımda otomatik indirilir. Önceden indirmek isterseniz:

```bash
# Whisper modelini indir (~3 GB)
python3 -c "
import mlx_whisper
mlx_whisper.transcribe('', path_or_hf_repo='mlx-community/whisper-large-v3-mlx')
" 2>/dev/null || true

# BGE-M3 embedding modelini indir (~2 GB)
python3 -c "
from FlagEmbedding import BGEM3FlagModel
BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print('BGE-M3 hazır')
"
```

> Modeller `~/.cache/huggingface/` klasörüne kaydedilir. İndirme yalnızca bir kez yapılır.

---

## Yapılandırma

Proje kökündeki `.env.example` dosyasını kopyalayın:

```bash
cp .env.example .env
```

`.env` dosyasını bir metin editörüyle açın ve API anahtarlarınızı girin:

```env
GOOGLE_API_KEY=AIza...           # Google Gemini API anahtarınız
OPENAI_API_KEY=sk-...            # OpenAI API anahtarınız (opsiyonel, yedek)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=finansal_analiz
```

> ⚠️ `.env` dosyasını asla Git'e commit etmeyin ve başkasıyla paylaşmayın. `.gitignore`'a zaten eklenmiştir.

---

## Sistemi Başlatma

### Yöntem A — Tek Tıkla Başlatma (Önerilen)

Proje kökündeki `launch.command` dosyasına çift tıklayın.

İlk açılışta "İzin verilmedi" hatası alırsanız terminalde şunu çalıştırın:

```bash
chmod +x launch.command
```

`launch.command` sırasıyla şunları yapar:
1. Docker Desktop'ı başlatır (zaten açıksa atlar)
2. Qdrant vektör veritabanını Docker'da çalıştırır
3. Streamlit arayüzünü arka planda başlatır
4. FastAPI backend'ini ön planda başlatır (loglar terminalde görünür)
5. Tarayıcıda `http://localhost:8501` adresini otomatik açar

### Yöntem B — Manuel Başlatma

Üç ayrı terminal penceresi açın:

**Terminal 1 — Qdrant:**
```bash
docker compose up -d qdrant
```

**Terminal 2 — FastAPI Backend:**
```bash
cd finansal-ajan
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 3 — Streamlit UI:**
```bash
cd finansal-ajan
source .venv/bin/activate
streamlit run ui/app.py --server.port 8501
```

Tarayıcıdan `http://localhost:8501` adresine gidin.

### Servislerin Çalışıp Çalışmadığını Kontrol Edin

```bash
# Qdrant
curl http://localhost:6333
# {"title":"qdrant - vector search engine",...} beklenir

# FastAPI
curl http://localhost:8000/sessions
# {"sessions":[...]} beklenir
```

Streamlit için tarayıcıdan `http://localhost:8501` adresini açın.

---

## Arayüz Genel Bakış

Streamlit arayüzü üç ana sekmeden oluşur:

```
┌──────────────────────────────────────────────────────────────┐
│  📈 Türkçe Borsa Analizi                                     │
│                                                              │
│  [ 🎙️ Analiz ] [ 🔍 Arama ] [ ℹ️ Hakkında ]                │
│                                                              │
│  ┌─── Sidebar ───┐  ┌─── Ana İçerik ──────────────────────┐ │
│  │ 📂 Kayıtlı   │  │                                      │ │
│  │ Oturumlar    │  │  1️⃣ Ses Dosyası                      │ │
│  │              │  │  2️⃣ Transkripsiyon                   │ │
│  │ • analiz-01  │  │  3️⃣ Finansal Sinyal Analizi          │ │
│  │ • analiz-02  │  │                                      │ │
│  │              │  │  [Sinyal kartları buraya akar]        │ │
│  └──────────────┘  └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Sol kenar çubuğu (Sidebar):** Daha önce tamamlanan oturumlar listelenir. Her oturumun kaç chunk ve sinyal içerdiği, tarihi gösterilir. Bir oturuma tıklandığında transkript ve sinyaller anında yüklenir, yeniden analiz yapmaya gerek kalmaz. 🗑 simgesiyle oturum silinebilir.

**Analiz sekmesi:** Ses yükleme, transkripsiyon ve sinyal analizi adımlarını sırasıyla yönetir.

**Arama sekmesi:** Qdrant vektör araması ve fiyat doğrulama özelliklerini sunar.

---

## Kullanım Kılavuzu

### 1. Ses Dosyası Yükleme ve Transkripsiyon

> **Video dosyanız mı var?** Sistem doğrudan ses dosyası kabul eder. Video dosyasını önce ses dosyasına dönüştürmeniz gerekir. `ffmpeg` kuruluysa terminalde şu komutu çalıştırın:
> ```bash
> ffmpeg -i video.mp4 -vn -acodec copy ses.m4a
> ```
> `ffmpeg` kurulu değilse: [ffmpeg.org/download](https://ffmpeg.org/download.html) adresinden indirin veya `brew install ffmpeg` komutunu kullanın.

1. Sayfanın sol kenar çubuğunda **"Oturum başlığı"** alanına bir isim girin (ör. `analiz-01`). Bu isimle oturum diske kaydedilecektir.
2. **"Ses dosyası yükle"** alanından `.m4a`, `.mp3` veya `.wav` formatında dosyanızı seçin.
3. **"▶ Başlat"** butonuna tıklayın.
4. Transkripsiyon ilerledikçe metin ekranda parça parça belirmeye başlar.

**Beklenen süreler (Apple M-serisi, yaklaşık):**

| Kayıt Süresi | Beklenen Transkripsiyon Süresi |
|--------------|-------------------------------|
| 30 dakika | 3-5 dakika |
| 60 dakika | 6-10 dakika |
| 95 dakika | 10-16 dakika |

**Durdurma ve Devam Etme:**
Transkripsiyon sırasında **"⏹ Durdur"** butonuna tıklayabilirsiniz. Devam ettirmek için **"▶ Devam Et"** butonuna tıklayın — sistem kaldığı chunk'tan devam eder, baştan başlamaz.

**"🔄 Sıfırla"** butonu transkripsiyon durumunu temizler ve baştan başlamanıza olanak tanır.

### 2. Finansal Sinyal Analizi

Transkripsiyon tamamlandıktan sonra:

1. Sayfanın **"3️⃣ Finansal Sinyal Analizi"** bölümüne gidin.
2. **"▶ Başlat"** butonuna tıklayın.
3. Her chunk analiz edildikçe sinyal kartları ekranda akmaya başlar.

Analiz tamamlanınca üstte özet metrikler görünür:

```
┌────────┬───────────┬──────────┬────────┐
│ Chunk  │  Başarılı │  Sinyal  │ Hisse  │
│   4    │     4     │   39     │   12   │
└────────┴───────────┴──────────┴────────┘
Tespit edilen hisseler: THYAO · GARAN · AKBNK · EREGL · ...
```

Sinyal listesinin üstünde iki filtre çubuğu vardır:
- **Hisse filtresi:** Sadece belirli hisseleri göster (çoklu seçim)
- **Sinyal tipi filtresi:** alım / satım / stop_loss / destek / direnc / genel_yorum

Analiz de durdurulup devam ettirilebilir. "🔄 Sıfırla" ile sonuçlar temizlenip baştan çalıştırılabilir.

### 3. Sinyal Kartlarını Okuma

Her sinyal kartı şunları gösterir:

```
┌──────────────────────────────────────────────┐
│  THYAO                      ● yuksek   #02  │
│  alım   45.50 TL                            │
│  "Kesinlikle 45.50 üzerinde alabilirsiniz"  │
└──────────────────────────────────────────────┘
```

- **Hisse kodu** (sol üst) — hangi BIST hissesi
- **Güven rengi** — 🟢 Yeşil: yuksek, 🟡 Sarı: orta, ⚫ Gri: dusuk
- **#02** — Kaynaklandığı chunk numarası (hangi zaman dilimine denk gelir)
- **Sinyal tipi ve fiyat** — alım / satım / stop_loss / destek / direnc / genel_yorum
- **Kaynak cümle** — analistin tam ifadesi

### 4. Arama Sekmesi

**Arama sekmesi** iki ayrı işlev sunar:

#### Vektör Arama ve Metadata Filtreleme

Sol paneldeki filtreler:
- **Hisse:** Açılır menüden hisse kodu seçin veya "Tümü" bırakın
- **Sinyal tipi:** alım / satım / stop_loss / destek / direnc / genel_yorum
- **Güven:** yuksek / orta / dusuk
- **Sonuç sayısı:** 5-50 arası kaydırıcı

Sağ panele doğal dil sorgusu yazıp **"🔍 Ara"** butonuna basın. Qdrant, hem semantik benzerlik hem de anahtar kelime eşleşmesini birlikte kullanır:

| Örnek sorgu | Ne döner |
|-------------|----------|
| "THYAO alım" | THYAO için tüm alım sinyalleri |
| "destek seviyesi" | Tüm destek sinyalleri, benzer ifadeler dahil |
| "45 lira" | Fiyat yaklaşık 45 TL olan sinyaller |

Filtreler + metin araması birlikte kullanılabilir: ör. `hisse=THYAO` + `sinyal_tipi=alım` + sorgu "kesinlikle" → "Sadece THYAO'nun güçlü alım noktalarını getir" anlamına gelir.

Her sinyal Qdrant'ta şu metadata alanlarıyla etiketlidir: `hisse`, `sinyal_tipi`, `fiyat`, `para_birimi`, `guven`, `chunk_id`, `zaman_sn`, `video_title`, `created_at`.

#### Fiyat Doğrulama

Sol panelde bir hisse seçili iken **"✅ Fiyat Doğrula"** butonuna tıklayın. Sistem o hisseye ait tüm fiyatlı sinyalleri çeker, her biri için BIST'ten anlık fiyat alır ve tablo olarak gösterir:

```
Hisse │ Sinyal │ Analist Fiyatı │ Anlık Fiyat │  Fark   │ Yorum
──────┼────────┼────────────────┼─────────────┼─────────┼───────────────────────
THYAO │  alım  │    45.50 TL    │   47.20 TL  │ +3.7% ↑ │ Analist 45.50 dedi...
GARAN │ destek │    28.00 TL    │   29.15 TL  │ +4.1% ↑ │ Analist 28.00 dedi...
```

Pozitif fark yeşil, negatif fark kırmızı renkte gösterilir.

**"🗑 Temizle"** butonu Qdrant'taki tüm kayıtları siler. Dikkatli kullanın.

### 5. Eski Oturumları Yükleme

Tamamlanan analizler otomatik olarak `data/sessions/` klasörüne kaydedilir. Sol kenar çubuğundaki oturum listesinden herhangi birine tıklayarak eski transkript ve sinyalleri yükleyebilirsiniz.

---

## Proje Yapısı

```
finansal-ajan/
│
├── src/                              # Kaynak kodlar
│   ├── transcription/
│   │   ├── transcriber.py            # mlx-whisper transkripsiyon
│   │   └── prompts.py                # Whisper ve LLM sistem promptları
│   ├── agents/
│   │   ├── chunker.py                # Metin parçalama — Ajan 1
│   │   └── analyst.py                # Sinyal çıkarımı — Ajan 2
│   ├── qdrant/
│   │   ├── client.py                 # Qdrant bağlantı yönetimi
│   │   ├── uploader.py               # BGE-M3 embedding + kayıt — Ajan 3
│   │   └── searcher.py               # Hibrit vektör arama
│   ├── verification/
│   │   └── agent.py                  # Gerçek zamanlı fiyat doğrulama — Bonus Ajan
│   └── utils/
│       ├── api_helpers.py            # Gemini / GPT-4o-mini sarmalayıcı
│       └── logger.py                 # Merkezi loglama
│
├── api/
│   └── main.py                       # FastAPI endpoint'leri, job sistemi
│
├── ui/
│   └── app.py                        # Streamlit arayüzü
│
├── data/
│   └── sessions/                     # Kaydedilen oturumlar
│       └── {oturum-adi}/
│           ├── chunks.json           # Transkript parçaları
│           ├── signals.json          # LLM çıktısı sinyaller
│           └── meta.json             # Oturum meta verisi
│
├── orchestrator.py                   # Pipeline orkestratörü (CLI)
├── bist_hisseler.json                # BIST hisse listesi + konuşma dili alias'ları
├── docker-compose.yml                # Qdrant Docker yapılandırması
├── launch.command                    # Tek tıkla başlatıcı (macOS)
├── requirements.txt                  # Python bağımlılıkları
├── .env.example                      # Örnek ortam değişkenleri
└── README.md                         # Bu dosya
```

---

## Teknoloji Yığını

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Transkripsiyon | `mlx-whisper` large-v3 | Apple Silicon optimizeli Whisper |
| Birincil LLM | Google Gemini 2.5 Flash | Sinyal analizi (thinking modu kapalı) |
| Yedek LLM | OpenAI GPT-4o-mini | Gemini başarısız olursa devreye girer |
| Embedding | `BAAI/bge-m3` | Dense + sparse hibrit vektör, 1024 boyut |
| Vektör DB | Qdrant (Docker) | Hibrit arama, metadata filtreleme |
| Backend | FastAPI + uvicorn | Async job sistemi, polling tabanlı ilerleme |
| Frontend | Streamlit | Gerçek zamanlı sinyal akışı |
| Fiyat Verisi | borsapy | BIST anlık fiyat sorgusu |
| Hata Toleransı | tenacity | 3 deneme, üstel bekleme (4-30 sn) |
| Python | 3.12 | — |

---

## Sık Karşılaşılan Sorunlar

### "mlx_whisper bulunamadı" veya import hatası

`mlx` yalnızca Apple Silicon Mac'lerde çalışır. Intel Mac kullanıyorsanız `mlx-whisper` yüklenemez. Bu durumda `requirements.txt`'ten `mlx-whisper` satırını kaldırın ve `src/transcription/transcriber.py` dosyasını `faster-whisper` gibi bir alternatifle güncelleyin.

### Transkripsiyon çok yavaş veya takılı kaldı

Whisper modeli ilk çalıştırmada Hugging Face'den indirilir (~3 GB). İndirme tamamlanana kadar beklemeniz normaldir. Terminalde indirme ilerlemesini görebilirsiniz. İndirme tamamlandıktan sonra işlem hızlanır.

### "GOOGLE_API_KEY geçersiz" veya 401 hatası

`.env` dosyasındaki API anahtarını kontrol edin:
- Anahtarın başında veya sonunda boşluk olmamalı
- Anahtarın tırnak işareti içinde yazılmamış olması gerekir (ör. `GOOGLE_API_KEY=AIza...` doğru, `GOOGLE_API_KEY="AIza..."` yanlış)
- [aistudio.google.com](https://aistudio.google.com) adresinden anahtarın aktif olduğunu doğrulayın

### Qdrant'a bağlanılamıyor

Docker Desktop'ın çalıştığından emin olun:

```bash
# Docker durumunu kontrol et
docker ps

# "qdrant" adında container yoksa:
docker compose up -d qdrant

# Qdrant'ın çalışıp çalışmadığını test et
curl http://localhost:6333
```

### Analiz başlatınca "0 sinyal" çıkıyor

Olası nedenler ve çözümler:

1. **Transkript boş veya çok kısa** → "Transkripsiyon" sekmesinde metnin gelip gelmediğini kontrol edin
2. **API kotası doldu** → Terminaldeki log mesajlarına bakın; her iki API da başarısız oluyorsa kota veya bağlantı sorunu var demektir
3. **Ses dosyası formatı desteklenmiyor** → `.m4a`, `.mp3`, `.wav` formatlarından birini kullanın

### Streamlit sayfası açılmıyor veya boş geliyor

FastAPI backend'inin çalıştığından emin olun:

```bash
curl http://localhost:8000/sessions
```

Hata alıyorsanız backend başlatılmamış demektir:

```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### "Port zaten kullanımda" hatası

8000 veya 8501 portunda başka bir işlem çalışıyor olabilir:

```bash
# 8000 portunu kullanan işlemi bul ve sonlandır
lsof -ti:8000 | xargs kill -9

# 8501 portunu kullanan işlemi bul ve sonlandır
lsof -ti:8501 | xargs kill -9
```

### Eski oturum yüklenince transkript görünmüyor

Oturumun `data/sessions/{isim}/chunks.json` dosyasının mevcut olup olmadığını kontrol edin. Eğer bu dosya yoksa (eski sürümde kaydedilmemişse) transkript alanı boş görünür; ancak analiz yeniden çalıştırılabilir.

---

## BIST Hisse Alias'ları Ekleme

Sistem, analistlerin şirket adlarını konuşma dilinde söyleme biçimlerini `bist_hisseler.json` dosyasındaki alias tablosundan öğrenir. Yeni bir hisse veya takma ad eklemek için:

```json
{
  "ticker": "THYAO",
  "name": "Turk Hava Yollari A.O.",
  "aliases": ["thy", "türk hava yolları", "türk havayolları", "thy hissesi"]
}
```

Whisper'ın yanlış transkribe ettiği formlar da alias olarak eklenebilir (ör. Whisper "Vakfa" → "Akpa" yazıyorsa `"akpa"` alias'ı VAKFA'ya eklenir).

---

## Lisans

Bu proje kişisel kullanım amaçlıdır. Ticari kullanım için Google Gemini, OpenAI ve BAAI/bge-m3 model lisanslarını ayrıca değerlendirin.
