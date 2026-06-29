"""Streamlit kullanıcı arayüzü — ses yükleme, sinyal arama ve proje bilgisi."""

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Borsa Analizi", page_icon="📈", layout="wide")
st.title("📈 Türkçe Borsa Analizi")

tab1, tab2, tab3 = st.tabs(["Analiz", "Arama", "Hakkında"])

# ── SEKME 1: Analiz ──────────────────────────────────────────────────────────

with tab1:
    st.header("Ses Dosyası Analizi")

    uploaded = st.file_uploader("Ses dosyası yükle", type=["m4a", "mp3", "wav", "mp4"])
    video_title = st.text_input("Video başlığı (opsiyonel)", placeholder="ör. 15 Ocak Borsa Analizi")

    if st.button("Analizi Başlat", type="primary"):
        if uploaded is None:
            st.warning("Lütfen önce bir ses dosyası yükleyin.")
        else:
            with st.spinner("Analiz yapılıyor... Bu işlem uzun sürebilir."):
                try:
                    response = httpx.post(
                        f"{API_URL}/analyze",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data={"title": video_title or "bilinmiyor"},
                        timeout=3600,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success("Analiz tamamlandı!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Toplam Chunk", result["toplam_chunk"])
                    col2.metric("Toplam Sinyal", result["toplam_sinyal"])
                    col3.metric("İşlenen Hisse", len(result["islenen_hisseler"]))
                    st.write("İşlenen hisseler:", result["islenen_hisseler"])
                    with st.expander("Tam özet"):
                        st.json(result)
                except httpx.HTTPStatusError as exc:
                    st.error(f"API hatası: {exc.response.status_code} — {exc.response.text}")
                except Exception as exc:
                    st.error(f"Hata: {exc}")

# ── SEKME 2: Arama ───────────────────────────────────────────────────────────

with tab2:
    st.header("Sinyal Arama")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Filtreler")

        # Hisse listesini API'den çek
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
        limit = st.slider("Sonuç sayısı", min_value=5, max_value=50, value=10)

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

                    st.caption(f"{data.get('count', 0)} sonuç bulundu")

                    if results:
                        import pandas as pd
                        df = pd.DataFrame(results)
                        gösterilecek = [c for c in
                            ["hisse", "sinyal_tipi", "fiyat", "guven", "kaynak_cumle", "zaman_sn"]
                            if c in df.columns]
                        st.dataframe(df[gösterilecek], use_container_width=True)
                    else:
                        st.info("Sonuç bulunamadı.")
                except Exception as exc:
                    st.error(f"Arama hatası: {exc}")

        st.divider()
        st.subheader("Fiyat Doğrulama")

        if st.button("Fiyat Doğrula", disabled=(secili_hisse == "Tümü")):
            if secili_hisse == "Tümü":
                st.warning("Fiyat doğrulamak için sol panelden bir hisse seçin.")
            else:
                with st.spinner(f"{secili_hisse} sinyalleri doğrulanıyor..."):
                    try:
                        r = httpx.get(f"{API_URL}/verify", params={"hisse": secili_hisse}, timeout=30)
                        r.raise_for_status()
                        data = r.json()
                        results = data.get("results", [])
                        st.caption(f"{data.get('count', 0)} sinyal doğrulandı")

                        if results:
                            import pandas as pd
                            df = pd.DataFrame(results)
                            cols = [c for c in
                                ["hisse", "sinyal_tipi", "fiyat", "anlik_fiyat", "fark_yuzde", "yorum"]
                                if c in df.columns]
                            df_show = df[cols].copy()

                            def renk(val):
                                if val is None or not isinstance(val, (int, float)):
                                    return ""
                                return "color: green" if val >= 0 else "color: red"

                            if "fark_yuzde" in df_show.columns:
                                st.dataframe(
                                    df_show.style.applymap(renk, subset=["fark_yuzde"]),
                                    use_container_width=True,
                                )
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
    ses dosyasını metne çevirir, finansal sinyalleri çıkarır ve vektör
    veritabanında saklar.

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

    ### API Endpoint'leri

    | Method | Endpoint | Açıklama |
    |--------|----------|----------|
    | GET | `/health` | Sağlık kontrolü |
    | POST | `/analyze` | Ses dosyası analizi |
    | GET | `/search` | Hybrid sinyal arama |
    | GET | `/stocks` | Kayıtlı hisse listesi |
    | WS | `/progress` | İlerleme bildirimi |

    Swagger UI: [localhost:8000/docs](http://localhost:8000/docs)
    """)
