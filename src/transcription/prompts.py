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

    JSON formatı: [{"ticker": "THYAO", "name": "Turk Hava Yollari"}] veya ["THYAO", ...]
    """
    path = Path(json_path)
    if not path.exists():
        logger.warning("bist_hisseler.json bulunamadı: %s — boş set dönülüyor", path.resolve())
        return set()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            return {r["ticker"] for r in data}
        return set(data)
    except Exception as exc:
        logger.warning("bist_hisseler.json okunamadı: %s — boş set dönülüyor", exc)
        return set()


def load_bist_ticker_names(json_path: str = "bist_hisseler.json") -> dict[str, str]:
    """Ticker → şirket adı eşleştirme sözlüğü döndürür.

    Returns:
        {"THYAO": "Turk Hava Yollari A.O.", ...}
    """
    path = Path(json_path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            return {r["ticker"]: r["name"] for r in data}
        return {}
    except Exception as exc:
        logger.warning("Ticker isimleri yüklenemedi: %s", exc)
        return {}


def load_bist_aliases(json_path: str = "bist_hisseler.json") -> dict[str, str]:
    """Takma ad / konuşma dili → ticker kodu eşleştirmesi döndürür.

    Returns:
        {"paholi": "PAHOL", "vakfa": "VAKFA", "bim": "BIMAS", ...}
    """
    path = Path(json_path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        result: dict[str, str] = {}
        for r in data:
            if not isinstance(r, dict):
                continue
            ticker = r.get("ticker", "")
            for alias in r.get("aliases", []):
                result[alias.lower()] = ticker
        return result
    except Exception as exc:
        logger.warning("Alias listesi yüklenemedi: %s", exc)
        return {}


# ── KISIM C: build_analyst_system_prompt() ────────────────────────────────────

def build_analyst_system_prompt(
    bist_tickers: set[str],
    ticker_names: dict[str, str] | None = None,
) -> str:
    """Ajan 2 (analyst.py) için sistem promptunu oluşturur.

    Args:
        bist_tickers: load_bist_tickers() çıktısı — geçerli BIST kodları.
        ticker_names: load_bist_ticker_names() çıktısı — ticker → şirket adı.

    Returns:
        LLM'e gönderilecek tam sistem prompt metni.
    """
    ticker_list = ", ".join(sorted(bist_tickers))

    # Şirket adı → ticker eşleştirme tablosu (sadece adı olan ticker'lar)
    name_map_lines = ""
    if ticker_names:
        entries = [f"{name} → {t}" for t, name in sorted(ticker_names.items()) if name]
        name_map_lines = "\n".join(entries[:300])  # prompt sınırı için ilk 300

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
}

sinyal_tipi YALNIZCA şu 6 değerden biri olabilir:
  alım       → net alım tavsiyesi
  satım      → net satım / çıkış tavsiyesi
  stop_loss  → zarar kes seviyesi
  destek     → teknik destek seviyesi
  direnc     → teknik direnç seviyesi (hedef fiyat dahil)
  genel_yorum → diğer tüm yorumlar

GÜVEN SKORU KRİTERLERİ (çoğunluk "yuksek" olmamalı, gerçekçi dağıt):
  yuksek → Analist kesin ve net konuşuyor:
            "kesinlikle", "net", "orada dur", "burası önemli", "dikkat edin",
            açık fiyat seviyesi + net yön, "portföyünde olmayanlar bile al"
  orta   → Koşullu ifade veya hafif belirsizlik:
            "sanırım", "olabilir", "belki", "inşallah", "eğer X olursa Y",
            "bekliyorum ama garantisi yok", fiyat aralığı geniş (±%5+)
  dusuk  → Spekülasyon, tahmin veya bağlamdan çıkarım:
            "tahmin", "görebiliriz", "aklınızda bulunsun", "fikrim şu ki",
            hisse adı belirsiz ama bağlamdan atandı, analist tereddütlü

DÖNÜŞTÜRME KURALLARI:
  "hedef fiyat", "hedef" → sinyal_tipi="direnc" (fiyatı doldur)
  "negatif", "olumsuz" → sinyal_tipi="genel_yorum"
  "X'in altına inerse düşüş/negatif/çıkarım", "X kırılırsa sat" → sinyal_tipi="stop_loss" (X fiyatını doldur)
  "eğer X gerçekleşmezse çıkarım/satarım", koşula bağlı çıkış niyeti → sinyal_tipi="stop_loss"
  "tepki gelir", "tepki bekleriz", "yukarı gider" (net fiyat hedefi yoksa) → sinyal_tipi="genel_yorum"
  Bu 6 tip dışında HİÇBİR tip üretme.

GEÇMİŞ KİP / VARSAYIM KURALI (önemli):
  Analist GEÇMİŞTE gerçekleşmemiş bir senaryodan bahsediyorsa → sinyal_tipi="genel_yorum", guven="dusuk"
  Belirtiler: "düşmüş olsaydı", "gelmiş olsaydı", "inmiş olsaydı", "olsaydı daha iyi olurdu"
  Bu ifadeler analistin geriye dönük dileklerini yansıtır, gerçek bir fiyat sinyali DEĞİLDİR.
  ör. "106'ya düşmüş olsaydı cazip olurdu" → genel_yorum (106 TL destek değil)

KOŞULLU GELECEK + FİYAT KURALI:
  "X seviyesinin üstünde kapanış yaparsa yukarı döner" → sinyal_tipi="direnc", fiyat=X
  "X seviyesinin altında kapanış yaparsa düşer" → sinyal_tipi="stop_loss", fiyat=X
  Bu ifadelerde analist bir eşik fiyat belirtiyor — genel_yorum değil, direnc veya stop_loss kullan.

satım vs stop_loss AYIRT ETME (en sık karıştırılan):
  satım     → Analist koşulsuz ve şimdiki zamanda çıkış söylüyor: "sat", "çık", "pozisyonu kapat", "portföyden çıkar"
              + fiyat seviyesi OLMALIDIR (fiyat=null ise satım DEĞİLDİR)
  stop_loss → Koşullu gelecek: "X seviyesinin altına inerse", "kırılırsa sat", "olmasa çıkarım", "gelmezse satarım"
  Koşul içeren her ifade → stop_loss. Fiyat seviyesi yoksa → genel_yorum veya stop_loss (fiyat=null).
  satım + fiyat=null → genel_yorum olarak işaretle (fiyatsız satım geçersiz sinyal sayılır)

BANT / ARALIK SİNYALLERİ (aynı cümleden gereksiz ikili üretme):
  "X-Y bandı", "X ile Y arasında" → tek sinyal üret, fiyat olarak alt sınırı (destek için) veya üst sınırı (direnc için) kullan
  Aynı kaynak cümleden iki farklı fiyat için ayrı sinyal üretme; en anlamlı olanı seç
  İstisna: analist her iki seviyeyi de açıkça ayrı ayrı vurguluyorsa (ör. "50 destek, 60 direnc") iki sinyal üretebilirsin

alım KRİTERİ:
  Analist açıkça "al", "alabilirsiniz", "portföye ekle" diyorsa → alım
  Fiyat seviyesiyle birlikte "gelince alırım", "o seviyede alım yerim", "oradan alacağım",
  "banda gelince alım" gibi niyet belirtiyorsa → alım (fiyatı doldur)
  "tepki beklerim", "yukarı gider", "güzel grafik", "bekliyorum" (tek başına) → alım DEĞİL, genel_yorum

direnc/destek KELİME KURALI:
  Analist konuşmasında açıkça "direnç", "direnci", "direnç seviyesi" kelimesini kullanıyorsa → sinyal_tipi="direnc"
  Analist açıkça "destek", "dip", "taban" diyorsa → sinyal_tipi="destek"
  Model kendi yorumuyla tip ataması yaparken analistin kullandığı kelimeye öncelik ver.
  ör. "beş lirada kanal direnci var" → direnc (model "destek gibi davranıyor" diye değiştirme)

ARACILAR VE BROKERLAR:
  A1 Capital, İş Yatırım, Gedik, Ata, Deniz Yatırım, Garanti Yatırım, Yapı Kredi Yatırım gibi
  aracı kurum / broker isimleri HİSSE DEĞİLDİR. Bunların işlemlerini (satış, alış) sinyal olarak
  üretme. Sadece analistin bizzat tavsiye ettiği hisseleri raporla."""

    # Alias tablosu — konuşma dilindeki kısa/takma adlar
    _aliases = load_bist_aliases()
    alias_lines = "\n".join(f"{alias} → {ticker}" for alias, ticker in sorted(_aliases.items()))

    name_section = f"""
KONUŞMA DİLİ TAKMA AD → BIST KODU (önce buna bak):
Analistler şirket adını kısaltarak veya takma adıyla söyler:
{alias_lines}

Resmi şirket adları (İngilizce):
{name_map_lines}
""" if name_map_lines else ""

    return f"""Sen Türk borsası teknik analiz uzmanısın.
Verilen transkript bölümünden finansal sinyalleri çıkarıp JSON formatında döndürüyorsun.

HİSSE TESPİT KURALLARI (EN ÖNEMLİ):
- Konuşmada şirket adı veya BIST kodu AÇIKÇA geçiyorsa → aşağıdaki eşleştirme tablosundan BIST kodunu bul
- Hisse adı/kodu geçmiyorsa → hisse="belirsiz", guven="dusuk" yaz, ASLA tahmin etme
- [ŞİMDİYE KADAR BAHSEDİLENLER] listesini YALNIZCA önceki cümledeki konuyu sürdüren açık atıflar için kullan
- "bu hisse", "söz konusu şirket", "onun grafiği" gibi açık zamir atıfları kabul edilir
- Sadece rakam/teknik seviye geçiyorsa ve hisse belli değilse → hisse="belirsiz"
- Transkripsiyon hataları olabilir: ses benzerliğine göre BIST koduna eşleştir
  (ör. ünsüz düşmesi, hece kayması, benzer sesli harf — "Dov Roboti" → DOFRB, "Akfa" → VAKFA)
  Eşleşme bulunamazsa → hisse="belirsiz"
{name_section}
BAĞLAM KIRILMASI KURALLARI (ÇOK ÖNEMLİ):
Analist yeni bir hisseye geçtiğinde önceki hisse bağlamını DÜŞÜR. Aşağıdaki sinyaller yeni hisseye geçildiğini gösterir:
- Yeni bir şirket adı / ticker açıkça söyleniyorsa
- Fiyat seviyesi öncekiyle çelişiyor ve hisse adı değişmişse (ör. 14 TL'lik hisseden 1.67 TL'ye geçiş)
- "Şimdi X'e bakalım", "bir de şu var", "diğer hisse" gibi geçiş ifadeleri
- Farklı bir sektör / grup adı geçiyorsa (ör. "Pasifik grubu", "halka arz")

Bu durumlarda: YENİ hissenin adı açık değilse hisse="belirsiz" yaz. ESKİ hissenin adını yeni fiyat seviyelerine ATAMA.

BAĞLAM KURALLARI:
- [ÖNCEKİ BAĞLAM] bölümü bağlamı korumak içindir, oradan hisse adı çıkarabilirsin
- [ŞİMDİYE KADAR BAHSEDİLENLER] listesi referans içindir, otomatik atama için değil

genel_yorum SINIRLAMA (KESİN KURAL):
- Her hisse için YALNIZCA 1 adet genel_yorum üret, asla 2 veya daha fazla üretme
- Aynı hisse için birden fazla genel cümle varsa hepsini tek genel_yorum'un "gerekce" alanında birleştir
- Fiyat seviyesi olan sinyaller için genel_yorum üretme, direnc/destek/alım/satım kullan
- Bu kuralı ihlal etmek kesinlikle yasaktır: hisse=ATATP, sinyal_tipi=genel_yorum kombinasyonu sinyaller listesinde yalnızca 1 kez görünebilir

GEÇERLİ BIST KODLARI (yalnızca bunlar kabul edilir):
{ticker_list}

ÇIKTI KURALLARI:
- Sadece JSON döndür, başka hiçbir şey yazma
- Markdown kod bloğu kullanma
- Geçersiz ticker → belirsiz olarak işaretle
- Fiyat bilgisi yoksa fiyat: null
- Şüpheli hisse ataması → guven="dusuk" yaz

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
