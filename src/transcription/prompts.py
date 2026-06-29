"""Prompt sabitleri ve üretici fonksiyonlar.

WHISPER_INITIAL_PROMPT → yalnızca mlx-whisper için
build_analyst_system_prompt() → yalnızca Ajan 2 (analyst.py) için
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── KISIM A: WHISPER_INITIAL_PROMPT ──────────────────────────────────────────
# Whisper'ın BIST ticker kodlarını yanlış yazmaması için kısa bağlam.
# Whisper son 224 tokenı işler — liste değil doğal metin formatı kullan.

WHISPER_INITIAL_PROMPT = (
    "Borsa İstanbul teknik analiz. "
    "THYAO GARAN AKBNK EREGL TUPRS BIMAS SAHOL KCHOL "
    "ASELS SISE YKBNK HALKB VAKBN ISCTR TCELL FROTO "
    "TOASO ARCLK MGROS PGSUS KRDMD KOZAL VESTL ENKAI "
    "EKGYO TTKOM TKFEN TAVHL PETKM CCOLA analiz. "
    "Destek direnç kırılım RSI MACD Bollinger stop-loss "
    "fibonacci hacim konsolidasyon makas alım satım trend."
)


# ── KISIM B: load_bist_tickers() ──────────────────────────────────────────────

def load_bist_tickers(json_path: str = "bist_hisseler.json") -> set[str]:
    """bist_hisseler.json'dan ticker kodlarını yükler ve set olarak döndürür.

    Args:
        json_path: JSON dosyasının yolu (varsayılan: proje kökü).

    Returns:
        Ticker kodlarından oluşan set (O(1) lookup için).
    """
    path = Path(json_path)
    if not path.exists():
        logger.warning("bist_hisseler.json bulunamadı: %s — boş set dönülüyor", path.resolve())
        return set()

    try:
        with path.open(encoding="utf-8") as f:
            tickers = json.load(f)
        return set(tickers)
    except Exception as exc:
        logger.warning("bist_hisseler.json okunamadı: %s — boş set dönülüyor", exc)
        return set()


# ── KISIM C: build_analyst_system_prompt() ────────────────────────────────────

def build_analyst_system_prompt(bist_tickers: set[str]) -> str:
    """Ajan 2 (analyst.py) için sistem promptunu oluşturur.

    Args:
        bist_tickers: load_bist_tickers() çıktısı — geçerli BIST kodları.

    Returns:
        LLM'e gönderilecek tam sistem prompt metni.
    """
    ticker_list = ", ".join(sorted(bist_tickers))

    schema = """{
  "chunk_id": "string",
  "bahsedilen_hisseler": ["THYAO"],
  "sinyaller": [
    {
      "hisse": "THYAO",
      "sinyal_tipi": "alım|satım|stop_loss|destek|direnc|genel_yorum",
      "fiyat": 45.50,
      "para_birimi": "TL",
      "gerekce": "analistin tam ifadesi",
      "kaynak_cumle": "metinden alıntı",
      "guven": "yuksek|orta|dusuk"
    }
  ],
  "teknik_terimler": ["RSI"],
  "genel_yorum": "tek cümle özet"
}"""

    return f"""Sen Türk borsası teknik analiz uzmanısın.
Verilen transkript bölümünden finansal sinyalleri çıkarıp JSON formatında döndürüyorsun.

BAĞLAM KURALLARI:
- [ÖNCEKİ BAĞLAM] bölümündeki son hisseyi takip et
- [ŞİMDİYE KADAR BAHSEDİLENLER] listesini kullan
- Hisse adı geçmiyorsa son bahsedilen hisseye bağla
- Belirsizse: guven=dusuk, hisse=belirsiz
- Asla tahmin etme, bilmiyorsan belirsiz yaz

GEÇERLİ BIST KODLARI (yalnızca bunlar kabul edilir):
{ticker_list}

ÇIKTI KURALLARI:
- Sadece JSON döndür, başka hiçbir şey yazma
- Markdown kod bloğu kullanma
- Geçersiz ticker → belirsiz olarak işaretle
- Fiyat bilgisi yoksa fiyat: null

JSON ŞEMASI:
{schema}"""


# ── KISIM D: Kullanım yorumları ───────────────────────────────────────────────

# KULLANIM:
# Whisper için:
#   from src.transcription.prompts import WHISPER_INITIAL_PROMPT
#   result = mlx_whisper.transcribe(audio, initial_prompt=WHISPER_INITIAL_PROMPT)
#
# Ajan 2 için:
#   from src.transcription.prompts import load_bist_tickers, build_analyst_system_prompt
#   tickers = load_bist_tickers()
#   system  = build_analyst_system_prompt(tickers)
#   response = safe_llm_call(prompt=user_msg, system=system)
