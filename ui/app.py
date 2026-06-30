"""Streamlit kullanıcı arayüzü — adım adım pipeline, progress bar, durdurma desteği."""

import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_INTERVAL = 0.5  # saniye

st.set_page_config(page_title="Borsa Analizi", page_icon="📈", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.chunk-kart {
    border-left: 3px solid #1f77b4;
    padding: 8px 14px;
    background: #0e1117;
    border-radius: 0 6px 6px 0;
    font-size: 0.9em;
    line-height: 1.6;
}
.sinyal-kart {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    background: #111;
}
.badge { display:inline-block; border-radius:4px; padding:2px 8px;
         font-weight:bold; font-size:13px; margin-right:5px; }
.b-hisse { background:#1f77b4; color:#fff; }
.b-alim  { background:#00c853; color:#000; }
.b-satim { background:#ff1744; color:#fff; }
.b-stop  { background:#ff6d00; color:#fff; }
.b-destek{ background:#0097a7; color:#fff; }
.b-direnc{ background:#7b1fa2; color:#fff; }
.b-genel { background:#424242; color:#fff; }
.guven-y { color:#00c853; }
.guven-o { color:#ffd600; }
.guven-d { color:#757575; }
</style>""", unsafe_allow_html=True)

st.title("📈 Türkçe Borsa Analizi")

# ── Session state başlangıç değerleri ────────────────────────────────────────
_defaults = {
    "t_job_id": None, "t_cursor": 0, "t_running": False, "t_done": False,
    "t_chunks": [], "t_stop": False,
    "a_job_id": None, "a_cursor": 0, "a_running": False, "a_done": False,
    "a_sinyaller": [], "a_ozet": None, "a_stop": False, "a_log": [],
    "audio_bytes": None, "audio_name": None, "video_title": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _tip_badge(tip: str) -> str:
    cls = {"alım":"b-alim","satım":"b-satim","stop_loss":"b-stop",
           "destek":"b-destek","direnc":"b-direnc","genel_yorum":"b-genel"}.get(tip,"b-genel")
    return f'<span class="badge {cls}">{tip}</span>'


def _guven_cls(g: str) -> str:
    return {"yuksek":"guven-y","orta":"guven-o","dusuk":"guven-d"}.get(g,"")


def _sinyal_html(s: dict) -> str:
    hisse = s.get("hisse","?")
    tip   = s.get("sinyal_tipi","")
    fiyat = s.get("fiyat")
    guven = s.get("guven","")
    gerekce = s.get("gerekce","")
    kaynak  = s.get("kaynak_cumle","")
    chunk   = s.get("chunk_id","")
    fiyat_str = f"&nbsp;<b>{fiyat} TL</b>" if fiyat else ""
    return (
        f'<div class="sinyal-kart">'
        f'<span class="badge b-hisse">{hisse}</span>'
        f'{_tip_badge(tip)}{fiyat_str}'
        f'<span class="{_guven_cls(guven)}" style="float:right">● {guven} &nbsp; #{chunk}</span>'
        f'<br><small style="color:#aaa">{gerekce}</small>'
        f'<br><small style="color:#555;font-style:italic">"{kaynak}"</small>'
        f'</div>'
    )


def _api(method: str, path: str, **kw):
    return getattr(httpx, method)(f"{API_URL}{path}", timeout=30, **kw)


def _poll(job_id: str, cursor: int) -> dict:
    r = httpx.get(f"{API_URL}/jobs/{job_id}/poll", params={"cursor": cursor}, timeout=10)
    r.raise_for_status()
    return r.json()


def _cancel(job_id: str):
    try:
        httpx.delete(f"{API_URL}/jobs/{job_id}", timeout=5)
    except Exception:
        pass


# ── API sağlık kontrolü ───────────────────────────────────────────────────────
try:
    _ok = httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
except Exception:
    _ok = False

if not _ok:
    st.error("⚠️ API'ye bağlanılamıyor. Lütfen uygulamayı **launch.command** ile başlatın.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎙️ Analiz", "🔍 Arama", "ℹ️ Hakkında"])

# ─────────────────────────────────────────────────────────────────────────────
# SEKME 1 — ANALİZ
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── BLOK A: Dosya yükleme ────────────────────────────────────────────────
    st.subheader("1️⃣ Ses Dosyası")
    uploaded = st.file_uploader(
        "Ses dosyası seçin (.m4a / .mp3 / .wav / .mp4)",
        type=["m4a","mp3","wav","mp4"],
    )
    if uploaded:
        st.session_state.audio_bytes = uploaded.getvalue()
        st.session_state.audio_name  = uploaded.name

    st.session_state.video_title = st.text_input(
        "Analiz başlığı", value=st.session_state.video_title,
        placeholder="ör. 30 Haziran Borsa Analizi",
    )

    st.divider()

    # ── BLOK B: Transkripsiyon ───────────────────────────────────────────────
    st.subheader("2️⃣ Transkripsiyon")

    col_b1, col_b2, col_b3 = st.columns([1, 1, 4])

    # Başlat butonu
    if col_b1.button("▶ Başlat", disabled=(not st.session_state.audio_bytes or st.session_state.t_running)):
        # Önceki job'ı temizle
        if st.session_state.t_job_id:
            _cancel(st.session_state.t_job_id)
        st.session_state.t_chunks  = []
        st.session_state.t_done    = False
        st.session_state.t_cursor  = 0
        st.session_state.t_stop    = False

        # Job başlat
        r = httpx.post(
            f"{API_URL}/jobs/transcribe",
            files={"file": (st.session_state.audio_name,
                            st.session_state.audio_bytes, "audio/mpeg")},
            data={"title": st.session_state.video_title or "bilinmiyor"},
            timeout=30,
        )
        r.raise_for_status()
        st.session_state.t_job_id  = r.json()["job_id"]
        st.session_state.t_running = True
        st.rerun()

    # Durdur butonu
    if col_b2.button("⏹ Durdur", disabled=(not st.session_state.t_running)):
        st.session_state.t_stop    = True
        st.session_state.t_running = False
        _cancel(st.session_state.t_job_id)

    # ── Polling döngüsü (transkripsiyon) ─────────────────────────────────────
    if st.session_state.t_running and st.session_state.t_job_id:

        if st.session_state.t_stop:
            st.session_state.t_running = False
        else:
            try:
                snap = _poll(st.session_state.t_job_id, st.session_state.t_cursor)
                new_events = snap.get("events", [])
                st.session_state.t_cursor += len(new_events)

                for ev in new_events:
                    typ  = ev["type"]
                    data = ev["data"]

                    if typ == "progress":
                        st.session_state["t_last_progress"] = data
                    elif typ == "chunk":
                        st.session_state.t_chunks.append(data)
                        st.session_state["t_last_progress"] = {
                            "mesaj": f"✅ Chunk {data['chunk_id']} tamamlandı ({data.get('dakika',0)}. dakika)",
                            "yuzde": int((data["chunk_idx"]+1) / max(data["total_chunks"],1) * 100),
                        }
                    elif typ in ("done", "cancelled", "error"):
                        st.session_state.t_running = False
                        st.session_state.t_done    = (typ == "done")

                if snap.get("done"):
                    st.session_state.t_running = False
                    st.session_state.t_done    = not snap.get("cancelled")

            except Exception as exc:
                st.warning(f"Polling hatası: {exc}")
                st.session_state.t_running = False

            if st.session_state.t_running:
                time.sleep(POLL_INTERVAL)
                st.rerun()

    # ── Transkripsiyon durumu göstergesi ─────────────────────────────────────
    t_progress_bar  = st.empty()
    t_status_box    = st.empty()

    chunks = st.session_state.t_chunks
    last_prog = st.session_state.get("t_last_progress", {})

    if chunks or st.session_state.t_running or st.session_state.t_done:
        done_count = len(chunks)
        total_est  = chunks[-1].get("total_chunks", done_count) if chunks else 10

        # Backend'den gelen son yüzde varsa onu kullan, yoksa chunk sayısından tahmin
        yuzde_backend = last_prog.get("yuzde", 0) / 100 if last_prog else 0
        yuzde_chunk   = done_count / max(total_est, 1)
        yuzde = max(yuzde_backend, yuzde_chunk, 0.02)

        mesaj_backend = last_prog.get("mesaj", "")

        if st.session_state.t_running:
            label = mesaj_backend or f"⏳ {done_count}/{total_est} chunk tamamlandı..."
            t_progress_bar.progress(min(yuzde, 0.99), text=label)
            t_status_box.info(f"🔄 {label}")
        elif st.session_state.t_done:
            t_progress_bar.progress(1.0, text=f"✅ Transkripsiyon tamamlandı — {done_count} chunk")
            t_status_box.success(f"✅ {done_count} chunk oluşturuldu")
        else:
            t_progress_bar.progress(yuzde, text=f"⏸ Durduruldu — {done_count} chunk alındı")
            t_status_box.warning(f"⏸ Durduruldu. Kaldığı yerden devam etmek için tekrar ▶ Başlat'a basın.")

    # ── Chunk listesi ─────────────────────────────────────────────────────────
    if chunks:
        st.markdown(f"**{len(chunks)} chunk** — tıklayarak metni okuyun:")
        for c in chunks:
            dakika = int(c.get("start_sec", 0) // 60)
            cache_icon = "💾" if c.get("from_cache") else "🎙"
            with st.expander(
                f"{cache_icon} Chunk {c['chunk_id']} — {dakika}. dakika "
                f"({c.get('word_count', '?')} kelime)"
            ):
                st.markdown(
                    f'<div class="chunk-kart">{c["text"]}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── BLOK C: Analiz ───────────────────────────────────────────────────────
    st.subheader("3️⃣ Finansal Sinyal Analizi")

    col_c1, col_c2, col_c3 = st.columns([1, 1, 4])

    if col_c1.button("▶ Başlat", key="analiz_start",
                     disabled=(not st.session_state.audio_bytes or st.session_state.a_running)):
        if st.session_state.a_job_id:
            _cancel(st.session_state.a_job_id)
        st.session_state.a_sinyaller = []
        st.session_state.a_ozet      = None
        st.session_state.a_done      = False
        st.session_state.a_cursor    = 0
        st.session_state.a_stop      = False
        st.session_state.a_log       = []

        r = httpx.post(
            f"{API_URL}/jobs/analyze",
            files={"file": (st.session_state.audio_name,
                            st.session_state.audio_bytes, "audio/mpeg")},
            data={"title": st.session_state.video_title or "bilinmiyor"},
            timeout=60,
        )
        r.raise_for_status()
        st.session_state.a_job_id  = r.json()["job_id"]
        st.session_state.a_running = True
        st.rerun()

    if col_c2.button("⏹ Durdur", key="analiz_stop", disabled=(not st.session_state.a_running)):
        st.session_state.a_stop    = True
        st.session_state.a_running = False
        _cancel(st.session_state.a_job_id)

    # ── Polling döngüsü (analiz) ──────────────────────────────────────────────
    if st.session_state.a_running and st.session_state.a_job_id:

        if st.session_state.a_stop:
            st.session_state.a_running = False
        else:
            try:
                snap = _poll(st.session_state.a_job_id, st.session_state.a_cursor)
                new_events = snap.get("events", [])
                st.session_state.a_cursor += len(new_events)

                for ev in new_events:
                    typ  = ev["type"]
                    data = ev["data"]

                    if typ == "progress":
                        st.session_state["a_progress"] = data
                        msg = data.get("mesaj", "")
                        if msg and (not st.session_state.a_log or st.session_state.a_log[-1] != msg):
                            st.session_state.a_log.append(msg)

                    elif typ == "chunk_done":
                        for s in data.get("sinyaller", []):
                            st.session_state.a_sinyaller.append(s)
                        st.session_state.a_log.append(
                            f"✅ Chunk {data['chunk_id']} — "
                            f"{', '.join(data['hisseler']) or 'sinyal yok'} "
                            f"({data['sinyal_sayisi']} sinyal)"
                        )

                    elif typ == "chunk_error":
                        st.session_state.a_log.append(
                            f"⚠️ Chunk {data['chunk_id']}: {data['mesaj']}"
                        )

                    elif typ == "done":
                        st.session_state.a_ozet    = data.get("ozet", {})
                        st.session_state.a_sinyaller = data.get("sinyaller", [])
                        st.session_state.a_running = False
                        st.session_state.a_done    = True

                    elif typ in ("cancelled", "error"):
                        st.session_state.a_running = False

                if snap.get("done") and not st.session_state.a_done:
                    st.session_state.a_running = False

            except Exception as exc:
                st.warning(f"Analiz polling hatası: {exc}")
                st.session_state.a_running = False

            if st.session_state.a_running:
                time.sleep(POLL_INTERVAL)
                st.rerun()

    # ── Analiz ilerleme göstergesi ────────────────────────────────────────────
    prog = st.session_state.get("a_progress")
    if prog or st.session_state.a_running or st.session_state.a_done:
        yuzde = (prog.get("yuzde", 0) / 100) if prog else (1.0 if st.session_state.a_done else 0.02)
        if st.session_state.a_running:
            label = prog.get("mesaj", "Analiz yapılıyor...") if prog else "Analiz başlatılıyor..."
        elif st.session_state.a_done:
            label = "✅ Analiz tamamlandı"
        else:
            label = "⏸ Analiz durduruldu"
        st.progress(yuzde, text=label)

    # Canlı log (son 6 satır)
    if st.session_state.a_log:
        log_text = "\n".join(st.session_state.a_log[-6:])
        st.code(log_text, language=None)

    # ── Özet metrikler ────────────────────────────────────────────────────────
    if st.session_state.a_ozet:
        oz = st.session_state.a_ozet
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunk", oz.get("toplam_chunk", 0))
        c2.metric("Başarılı", oz.get("basarili_chunk", 0))
        c3.metric("Sinyal", oz.get("toplam_sinyal", 0))
        c4.metric("Hisse", len(oz.get("islenen_hisseler", [])))
        if oz.get("islenen_hisseler"):
            st.markdown("**Tespit edilen hisseler:** " +
                        " · ".join(f"`{h}`" for h in oz["islenen_hisseler"]))

    # ── Sinyal kartları ───────────────────────────────────────────────────────
    sinyaller = st.session_state.a_sinyaller
    if sinyaller:
        st.markdown(f"### Sinyaller ({len(sinyaller)})")

        hisseler_listesi = sorted({s.get("hisse","") for s in sinyaller
                                    if s.get("hisse") and s.get("hisse") != "belirsiz"})
        col_f1, col_f2 = st.columns(2)
        filtre_hisse = col_f1.multiselect("Hisse", hisseler_listesi, default=hisseler_listesi)
        filtre_tip   = col_f2.multiselect(
            "Sinyal tipi",
            ["alım","satım","stop_loss","destek","direnc","genel_yorum"],
            default=["alım","satım","stop_loss","destek","direnc"],
        )

        goster = [s for s in sinyaller
                  if s.get("hisse") in filtre_hisse and s.get("sinyal_tipi") in filtre_tip]
        if goster:
            for s in goster:
                st.markdown(_sinyal_html(s), unsafe_allow_html=True)
        else:
            st.info("Seçili filtrelere uyan sinyal yok.")


# ─────────────────────────────────────────────────────────────────────────────
# SEKME 2 — ARAMA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Sinyal Arama")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Filtreler")
        hisse_sec = ["Tümü"]
        try:
            r = httpx.get(f"{API_URL}/stocks", timeout=10)
            if r.status_code == 200:
                hisse_sec += r.json().get("hisseler", [])
        except Exception:
            pass

        sec_hisse = st.selectbox("Hisse", hisse_sec)
        sec_tip   = st.selectbox("Sinyal tipi",
                                 ["Tümü","alım","satım","stop_loss","destek","direnc","genel_yorum"])
        sec_guven = st.selectbox("Güven", ["Tümü","yuksek","orta","dusuk"])
        limit     = st.slider("Sonuç sayısı", 5, 50, 10)

    with col_right:
        st.subheader("Sorgu")
        sorgu = st.text_input("Arama sorgusu", placeholder="ör. THYAO alım seviyesi")

        if st.button("🔍 Ara", type="primary"):
            if not sorgu.strip():
                st.warning("Lütfen bir arama sorgusu girin.")
            else:
                params = {"q": sorgu, "limit": limit}
                if sec_hisse != "Tümü": params["hisse"]       = sec_hisse
                if sec_tip   != "Tümü": params["sinyal_tipi"] = sec_tip
                if sec_guven != "Tümü": params["guven"]       = sec_guven
                try:
                    r = httpx.get(f"{API_URL}/search", params=params, timeout=30)
                    r.raise_for_status()
                    data    = r.json()
                    results = data.get("results", [])
                    st.caption(f"{data.get('count',0)} sonuç")
                    for s in results:
                        st.markdown(_sinyal_html(s), unsafe_allow_html=True)
                    if not results:
                        st.info("Sonuç bulunamadı.")
                except Exception as exc:
                    st.error(f"Arama hatası: {exc}")

        st.divider()
        st.subheader("Fiyat Doğrulama")
        if st.button("✅ Fiyat Doğrula", disabled=(sec_hisse == "Tümü")):
            with st.spinner(f"{sec_hisse} doğrulanıyor..."):
                try:
                    r = httpx.get(f"{API_URL}/verify", params={"hisse": sec_hisse}, timeout=30)
                    r.raise_for_status()
                    data    = r.json()
                    results = data.get("results", [])
                    st.caption(f"{data.get('count',0)} sinyal")
                    if results:
                        import pandas as pd
                        df   = pd.DataFrame(results)
                        cols = [c for c in ["hisse","sinyal_tipi","fiyat","anlik_fiyat","fark_yuzde","yorum"]
                                if c in df.columns]
                        def renk(val):
                            if not isinstance(val, (int, float)): return ""
                            return "color:green" if val >= 0 else "color:red"
                        df_s = df[cols].copy()
                        if "fark_yuzde" in df_s.columns:
                            st.dataframe(df_s.style.applymap(renk, subset=["fark_yuzde"]),
                                         use_container_width=True)
                        else:
                            st.dataframe(df_s, use_container_width=True)
                    else:
                        st.info(f"{sec_hisse} için alım/satım sinyali bulunamadı.")
                except Exception as exc:
                    st.error(f"Doğrulama hatası: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# SEKME 3 — HAKKINDA
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Proje Hakkında")
    st.markdown("""
    **Türkçe Borsa Analizi Multi-Agent Sistemi**

    Uzun Türkçe borsa analiz ses dosyalarını otonom olarak işler.

    ---

    ### Pipeline Akışı

    | Adım | Açıklama |
    |------|----------|
    | 🎙️ Yükleme | Ses dosyası API'ye aktarılır |
    | 📝 Transkripsiyon | Whisper large-v3 ile ~10 dk'lık parçalar halinde gerçek zamanlı |
    | 📦 Chunking | Segment sınırlarına saygılı bölme + 300 kelime overlap |
    | 🤖 Analiz | Gemini 2.0 Flash ile sinyal çıkarımı (yedek: GPT-4o-mini) |
    | 💾 Kayıt | BGE-M3 hybrid embedding → Qdrant |

    ### Teknolojiler
    MLX Whisper · Gemini 2.0 Flash · GPT-4o-mini · BGE-M3 · Qdrant · FastAPI · Streamlit
    """)
