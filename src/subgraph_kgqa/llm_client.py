"""Shared LLM client for KGQA skill mining modules.

Consolidates 8 previously-duplicated LLM call implementations into a single
configurable function. All skill_mining callers should use `call_llm` instead
of local copies.

Environment variables (same as before):
    KGQA_LLM_API_URL       - API endpoint (default: http://127.0.0.1:8000/v1)
    KGQA_LLM_API_KEY       - Bearer token (default: EMPTY)
    KGQA_MODEL_NAME        - Model identifier (default: qwen35-9b-local)
    KGQA_ENABLE_THINKING   - Enable chain-of-thought (default: 0)
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Dict, List

import aiohttp


def _get_llm_config() -> Dict[str, str]:
    """Read LLM connection config from environment."""
    return {
        "api_url": os.getenv("KGQA_LLM_API_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
        "api_key": os.getenv("KGQA_LLM_API_KEY", "EMPTY"),
        "model_name": os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local"),
    }


async def call_llm(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 1400,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_seconds: float = 180.0,
    retries: int = 0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Call the LLM chat completions endpoint.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature.
    top_p : float
        Top-p (nucleus) sampling parameter.
    timeout_seconds : float
        Request timeout in seconds.
    retries : int
        Number of retry attempts on transient failures (0 = no retry).
    presence_penalty : float
        Presence penalty for token generation.
    repetition_penalty : float
        Repetition penalty for token generation.
    session : aiohttp.ClientSession, optional
        Existing session to reuse. If None, creates a new one per call.

    Returns
    -------
    str
        The assistant's response content.
    """
    config = _get_llm_config()
    enable_thinking = os.getenv("KGQA_ENABLE_THINKING", "0").strip().lower() in {"1", "true"}

    payload: Dict[str, Any] = {
        "model": config["model_name"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    url = f"{config['api_url']}/chat/completions"

    async def _do_call(s: aiohttp.ClientSession) -> str:
        for attempt in range(retries + 1):
            try:
                async with s.post(url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= retries:
                    raise
                await asyncio.sleep(1.0 * (attempt + 1) + random.uniform(0, 0.3))
        # Defensive: should be unreachable — the loop always returns or raises
        raise RuntimeError("LLM call failed after retries")

    if session is not None:
        return await _do_call(session)

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as s:
        return await _do_call(s)
