"""LLM API sarmalayıcı — Gemini 2.0 Flash (yedek: GPT-4o-mini).

Tüm ajan LLM çağrıları bu modüldeki safe_llm_call() üzerinden geçer.
Gemini başarısız olursa otomatik olarak GPT-4o-mini'ye geçilir.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_openai_client: OpenAI | None = None
_gemini_client: genai.Client | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _gemini_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=30))
def _call_gemini(prompt: str, system: str) -> str:
    time.sleep(1)
    client = _get_gemini_client()
    config = genai_types.GenerateContentConfig(
        system_instruction=system if system else None,
        max_output_tokens=1000,
    )
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=config,
    )
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=30))
def _call_gpt4o_mini(prompt: str, system: str) -> str:
    time.sleep(1)
    client = _get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def safe_llm_call(prompt: str, system: str = "") -> str:
    """LLM çağrısı yapar; Gemini başarısız olursa GPT-4o-mini'ye geçer.

    Args:
        prompt: Kullanıcı mesajı / analiz edilecek metin.
        system: Sistem promptu (varsayılan boş).

    Returns:
        Model yanıtı string olarak.

    Raises:
        Exception: Her iki model de başarısız olursa.
    """
    try:
        result = _call_gemini(prompt, system)
        logger.info("Gemini yanıtı alındı (%d karakter)", len(result))
        return result
    except Exception as gemini_exc:
        logger.warning("Gemini başarısız, GPT-4o-mini'ye geçiliyor: %s", gemini_exc)

    result = _call_gpt4o_mini(prompt, system)
    logger.info("GPT-4o-mini yanıtı alındı (%d karakter)", len(result))
    return result
