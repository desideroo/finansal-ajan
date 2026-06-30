"""Streamlit kullanıcı arayüzü — adım adım pipeline ile borsa analizi."""

import json
import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Borsa Analizi", page_icon="📈", layout="wide")

# ── Stil ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.sinyal-karti {
    border: 1px solid #2d2d2d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    background: #1a1a1a;
}
.hisse-badge {
    display: inline-block;
    background: #0066cc;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: bold;
    font-size: 14px;
    margin-right: 6px;
}
.alim { color: #00c853; font-weight: bold; }
.satim { color: #ff1744; font-weight: bold; }
.stop_loss { color: #ff6d00; font-weight: bold; }
.destek { color: #00bcd4; }
.direnc { color: #9c27b0; }
.genel_yorum { color: #9e9e9e; }
.guven-yuksek { color: #00c853; }
.guven-orta { color: #ffd600; }
.guven-dusuk { color: #9e9e9e; }
.chunk-kart {
    border-left: 3px solid #0066cc;
    padding: 8px 12px;
    margin-bottom: 6px;
    background: #111;
    border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Türkçe Borsa Analizi")

# ── Session state ─────────────────────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "sinyaller" not in st.session_state:
    st.session_state.sinyaller = []
if "ozet" not in st.session_state:
    st.session_state.ozet = None
if "asama" not in st.session_state:
    st.session_state.asama = "yukle"  # yukle | transkripsiyon | analiz | bitti


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _sinyal_renk(tip: str) -> str:
    return {
        "alım": "alim", "satım": "satim", "stop_loss": "stop_loss",
        "destek": "destek", "direnc": "direnc", "genel_yorum": "genel_yorum",
    }.get(tip, "")


def _guven_renk(g: str) -> str:
    return {"yuksek": "guven-yuksek", "orta": "guven-orta", "dusuk": "guven-dusuk"}.get(g, "")


def _sinyal_html(s: dict) -> str:
    tip = s.get("sinyal_tipi", "")
    hisse = s.get("hisse", "?")
    fiyat = s.get("fiyat")
    guven = s.get("guven", "")
    gerekce = s.get("gerekce", "")
    kaynak = s.get("kaynak_cumle", "")
    chunk = s.get("chunk_id", "")

    fiyat_str = f"<b>{fiyat} TL</b>" if fiyat else ""
    return f"""
<div class="sinyal-karti">
  <span class="hisse-badge">{hisse}</span>
  <span class="{_sinyal_renk(tip)}">{tip.upper()}</span>
  {fiyat_str}
  <span class="{_guven_renk(guven)}" style="float:right">● {guven} &nbsp; chunk {chunk}</span>
  <br><small style="color:#aaa">{gerekce}</small>
  <br><small style="color:#666;font-style:italic">"{kaynak}"</small>
</div>"""


def _sse_stream(endpoint: str, file_bytes: bytes, filename: str, title: str):
    """SSE endpoint'ini okur, (event, data) tuple'ları yield eder."""
    with httpx.stream(
        "POST",
        f"{API_URL}{endpoint}",
        files={"file": (filename, file_bytes, "audio/mpeg")},
        data={"title": title},
        timeout=7200,
    ) as r:
        buffer = ""
        cur_event = "message"
        for line in r.iter_lines():
            if line.startswith("event:"):
                cur_event = line[6:].strip()
            elif line.startswith("data:"):
                buffer = line[5:].strip()
                try:
                    yield cur_event, json.loads(buffer)
                except Exception:
                    pass
                cur_event = "message"


# ── API sağlık kontrolü ───────────────────────────────────────────────────────
try:
    health = httpx.get(f"{API_URL}/health", timeout=3)
    api_ok = health.status_code == 200
except Exception:
    api_ok = False

if not api_ok:
    st.error("⚠️ API'ye bağlanılamıyor. Lütfen uygulamayı launcher ile başlatın.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎙️ Analiz", "🔍 Arama", "ℹ️ Hakkında"])

# ── SEKME 1: Analiz ──────────────────────────────────────────────────────────
with tab1:

    # — Aşama 0: Dosya yükleme —
    st.subheader("1️⃣ Ses Dosyası Yükle")
    uploaded = st.file_uploader(
        "Ses dosyası seçin", type=["m4a", "mp3", "wav", "mp4"],
        help="95+ dakikalık dosyaları da destekler"
    )
    video_title = st.text_input(
        "Analiz başlığı", placeholder="ör. 30 Haziran Borsa Analizi",
        value=st.session_state.get("video_title", "")
    )
    if video_title:
        st.session_state.video_title = video_title

    st.divider()

    # — Aşama 1: Transkripsiyon —
    st.subheader("2️⃣ Transkripsiyon")

    col_t1, col_t2 = st.columns([1, 3])
    btn_transkripsiyon = col_t1.button(
        "▶ Transkripsiyonu Başlat",
        type="primary",
        disabled=(uploaded is None),
    )

    if btn_transkripsiyon and uploaded:
        st.session_state.chunks = []
        st.session_state.sinyaller = []
        st.session_state.ozet = None

        progress_bar = st.progress(0, text="Başlatılıyor...")
        status_box = st.empty()

        try:
            for event, data in _sse_stream(
                "/stream/transcribe",
                uploaded.getvalue(),
                uploaded.name,
                video_title or "bilinmiyor",
            ):
                if event == "progress":
                    yuzde = data.get("yuzde", 0)
                    mesaj = data.get("mesaj", "")
                    progress_bar.progress(yuzde / 100, text=mesaj)
                    status_box.info(f"⏳ {mesaj}")

                elif event == "tamamlandi":
                    chunks = data.get("chunks", [])
                    st.session_state.chunks = chunks
                    progress_bar.progress(1.0, text="Transkripsiyon tamamlandı!")
                    status_box.success(
                        f"✅ {len(chunks)} chunk oluşturuldu, "
                        f"{data.get('segment_sayisi', 0)} segment işlendi"
                    )

                elif event == "hata":
                    progress_bar.empty()
                    status_box.error(f"❌ {data.get('mesaj')}")
                    break

        except Exception as exc:
            st.error(f"Bağlantı hatası: {exc}")

    # Chunk'ları göster
    if st.session_state.chunks:
        st.markdown(f"**{len(st.session_state.chunks)} Chunk** — okumak için genişletin:")
        for c in st.session_state.chunks:
            dakika = int(c.get("start_sec", 0) // 60)
            with st.expander(
                f"Chunk {c['chunk_id']} — {dakika}. dakika "
                f"({c.get('word_count', '?')} kelime)"
            ):
                st.markdown(
                    f'<div class="chunk-kart">{c["text"]}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # — Aşama 2: Analiz —
    st.subheader("3️⃣ Finansal Sinyal Analizi")

    col_a1, col_a2 = st.columns([1, 3])
    btn_analiz = col_a1.button(
        "▶ Analizi Başlat",
        type="primary",
        disabled=(uploaded is None),
    )

    if btn_analiz and uploaded:
        st.session_state.sinyaller = []
        st.session_state.ozet = None

        progress_bar2 = st.progress(0, text="Analiz başlatılıyor...")
        status_box2 = st.empty()
        sinyal_container = st.container()

        chunk_sonuclari: dict[str, list] = {}

        try:
            for event, data in _sse_stream(
                "/stream/analyze",
                uploaded.getvalue(),
                uploaded.name,
                video_title or "bilinmiyor",
            ):
                if event == "progress":
                    yuzde = data.get("yuzde", 0)
                    mesaj = data.get("mesaj", "")
                    progress_bar2.progress(yuzde / 100, text=mesaj)
                    status_box2.info(f"⏳ {mesaj}")

                elif event == "chunk_tamamlandi":
                    chunk_id = data.get("chunk_id")
                    sinyaller = data.get("sinyaller", [])
                    hisseler = data.get("hisseler", [])
                    if sinyaller:
                        chunk_sonuclari[chunk_id] = sinyaller
                        st.session_state.sinyaller.extend(sinyaller)
                        if hisseler:
                            status_box2.success(
                                f"✅ Chunk {chunk_id}: {', '.join(hisseler)} — "
                                f"{len(sinyaller)} sinyal"
                            )

                elif event == "chunk_hata":
                    status_box2.warning(
                        f"⚠️ Chunk {data.get('chunk_id')} atlandı: {data.get('mesaj')}"
                    )

                elif event == "tamamlandi":
                    st.session_state.ozet = data.get("ozet", {})
                    st.session_state.sinyaller = data.get("sinyaller", [])
                    progress_bar2.progress(1.0, text="Analiz tamamlandı!")

                elif event == "hata":
                    progress_bar2.empty()
                    status_box2.error(f"❌ {data.get('mesaj')}")
                    break

        except Exception as exc:
            st.error(f"Bağlantı hatası: {exc}")

    # Özet ve sinyaller
    if st.session_state.ozet:
        oz = st.session_state.ozet
        st.markdown("### Özet")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunk", oz.get("toplam_chunk", 0))
        c2.metric("Başarılı", oz.get("basarili_chunk", 0))
        c3.metric("Sinyal", oz.get("toplam_sinyal", 0))
        c4.metric("Hisse", len(oz.get("islenen_hisseler", [])))

        if oz.get("islenen_hisseler"):
            st.markdown("**Tespit edilen hisseler:** " + " · ".join(
                f"`{h}`" for h in oz.get("islenen_hisseler", []) if h
            ))

    if st.session_state.sinyaller:
        st.markdown(f"### Sinyaller ({len(st.session_state.sinyaller)})")

        # Hisse filtresi
        hisseler_listesi = sorted({
            s.get("hisse", "") for s in st.session_state.sinyaller
            if s.get("hisse") and s.get("hisse") != "belirsiz"
        })
        filtre_hisse = st.multiselect(
            "Hisseye göre filtrele", hisseler_listesi, default=hisseler_listesi
        )
        filtre_tip = st.multiselect(
            "Sinyal tipine göre filtrele",
            ["alım", "satım", "stop_loss", "destek", "direnc", "genel_yorum"],
            default=["alım", "satım", "stop_loss", "destek", "direnc"],
        )

        filtrelenmis = [
            s for s in st.session_state.sinyaller
            if s.get("hisse") in filtre_hisse
            and s.get("sinyal_tipi") in filtre_tip
        ]

        for s in filtrelenmis:
            st.markdown(_sinyal_html(s), unsafe_allow_html=True)

        if not filtrelenmis:
            st.info("Seçili filtrelere göre sinyal bulunamadı.")


# ── SEKME 2: Arama ───────────────────────────────────────────────────────────
with tab2:
    st.header("Sinyal Arama")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Filtreler")
        hisse_secenekler = ["Tümü"]
        try:
            r = httpx.get(f"{API_URL}/stocks", timeout=10)
            if r.status_code == 200:
                hisse_secenekler += r.json().get("hisseler", [])
        except Exception:
            pass

        secili_hisse = st.selectbox("Hisse", hisse_secenekler)
        secili_tip = st.selectbox(
            "Sinyal tipi",
            ["Tümü", "alım", "satım", "stop_loss", "destek", "direnc", "genel_yorum"],
        )
        secili_guven = st.selectbox("Güven", ["Tümü", "yuksek", "orta", "dusuk"])
        limit = st.slider("Sonuç sayısı", 5, 50, 10)

    with col_right:
        st.subheader("Sorgu")
        sorgu = st.text_input("Arama sorgusu", placeholder="ör. THYAO alım seviyesi")

        if st.button("Ara", type="primary"):
            if not sorgu.strip():
                st.warning("Lütfen bir arama sorgusu girin.")
            else:
                params = {"q": sorgu, "limit": limit}
                if secili_hisse != "Tümü":
                    params["hisse"] = secili_hisse
                if secili_tip != "Tümü":
                    params["sinyal_tipi"] = secili_tip
                if secili_guven != "Tümü":
                    params["guven"] = secili_guven

                try:
                    r = httpx.get(f"{API_URL}/search", params=params, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results", [])
                    st.caption(f"{data.get('count', 0)} sonuç")

                    for s in results:
                        st.markdown(_sinyal_html(s), unsafe_allow_html=True)

                    if not results:
                        st.info("Sonuç bulunamadı.")
                except Exception as exc:
                    st.error(f"Arama hatası: {exc}")

        st.divider()
        st.subheader("Fiyat Doğrulama")
        if st.button("Fiyat Doğrula", disabled=(secili_hisse == "Tümü")):
            with st.spinner(f"{secili_hisse} doğrulanıyor..."):
                try:
                    r = httpx.get(f"{API_URL}/verify", params={"hisse": secili_hisse}, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results", [])
                    st.caption(f"{data.get('count', 0)} sinyal")

                    if results:
                        import pandas as pd
                        df = pd.DataFrame(results)
                        cols = [c for c in ["hisse", "sinyal_tipi", "fiyat", "anlik_fiyat", "fark_yuzde", "yorum"] if c in df.columns]

                        def renk(val):
                            if not isinstance(val, (int, float)):
                                return ""
                            return "color: green" if val >= 0 else "color: red"

                        df_show = df[cols].copy()
                        if "fark_yuzde" in df_show.columns:
                            st.dataframe(df_show.style.applymap(renk, subset=["fark_yuzde"]), use_container_width=True)
                        else:
                            st.dataframe(df_show, use_container_width=True)
                    else:
                        st.info(f"{secili_hisse} için alım/satım sinyali bulunamadı.")
                except Exception as exc:
                    st.error(f"Doğrulama hatası: {exc}")


# ── SEKME 3: Hakkında ────────────────────────────────────────────────────────
with tab3:
    st.header("Proje Hakkında")
    st.markdown("""
    **Türkçe Borsa Analizi Multi-Agent Sistemi**

    Uzun Türkçe borsa analiz ses dosyalarını otonom olarak işler:
    ses dosyasını metne çevirir, finansal sinyalleri çıkarır ve vektör veritabanında saklar.

    ---

    ### Kullanılan Teknolojiler

    | Katman | Teknoloji |
    |--------|-----------|
    | Transkripsiyon | MLX Whisper `large-v3` |
    | Ana LLM | Google Gemini 2.0 Flash |
    | Yedek LLM | OpenAI GPT-4o-mini |
    | Embedding | BGE-M3 (dense + sparse hybrid) |
    | Vektör DB | Qdrant |
    | Backend | FastAPI |
    | UI | Streamlit |

    ---

    ### Pipeline Akışı

    1. 🎙️ Ses dosyası yükle
    2. 📝 Whisper ile transkripsiyon (cache destekli)
    3. 📦 ~10 dk'lık chunk'lara bölme
    4. 🤖 Gemini ile finansal sinyal çıkarımı
    5. 💾 Qdrant vektör DB'ye kayıt
    """)
