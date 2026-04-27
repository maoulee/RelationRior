#!/usr/bin/env python3
"""Simple CLI chat with local LLM."""
from __future__ import annotations

import sys
import aiohttp
import asyncio

LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"


async def chat(user_msg: str, history: list, max_tokens: int = 500) -> str:
    history.append({"role": "user", "content": user_msg})
    payload = {
        "model": LLM_MODEL,
        "messages": history,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
    reply = data["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    history: list = []
    max_tokens = 500
    print("Local LLM chat (Ctrl+C to exit)\n")
    print("Commands: /clear /history /tokens <N> /system <text> /quit\n")
    while True:
        try:
            user_msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_msg:
            continue
        if user_msg in ("/quit", "/exit", "/q"):
            break
        if user_msg.startswith("/tokens "):
            try:
                max_tokens = int(user_msg.split()[1])
                print(f"max_tokens = {max_tokens}\n")
            except (ValueError, IndexError):
                print("Usage: /tokens <number>\n")
            continue
        if user_msg == "/clear":
            history.clear()
            print("History cleared.\n")
            continue
        if user_msg.startswith("/system "):
            history.insert(0, {"role": "system", "content": user_msg[8:]})
            print(f"System prompt set ({len(user_msg[8:])} chars)\n")
            continue
        if user_msg == "/history":
            for i, m in enumerate(history):
                preview = m['content'][:150].replace('\n', '\\n')
                print(f"  [{i}] {m['role']}: {preview}")
            print(f"  max_tokens={max_tokens}\n")
            continue

        reply = asyncio.run(chat(user_msg, history, max_tokens=max_tokens))
        print(reply)
        print()


if __name__ == "__main__":
    main()
