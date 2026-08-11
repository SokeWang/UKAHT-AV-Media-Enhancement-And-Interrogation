"""
experiments/yisheng/memory_agent.py
Owner: Yisheng Zhang — Agent Memory Experiment

Responsibilities:
  - Build prompts under four ablation conditions:
      1) no_memory
      2) short_term
      3) long_term
      4) short_and_long
  - Call an OpenAI-compatible LLM endpoint (e.g. Ollama) when available.
  - Keep the memory experiment independent from production agent code.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from memory_store import LongTermMemory, ShortTermMemory


DEFAULT_SYSTEM_PROMPT = """
You are an archive assistant for the UK Antarctic Heritage Trust (UKAHT).
Use supplied conversation and archive memory only when it is relevant.
Resolve follow-up references such as 'these', 'that building', 'the same site',
or 'another angle' using the available context.
Do not invent archive facts that are not present in the supplied context.
Keep answers concise and retrieval-oriented.
""".strip()


class ExperimentalMemoryAgent:
    """Small agent wrapper used for controlled memory ablation experiments."""

    def __init__(
        self,
        condition: str = "no_memory",
        short_memory: ShortTermMemory | None = None,
        long_memory: LongTermMemory | None = None,
        namespace: str = "default",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 90,
    ):
        valid = {"no_memory", "short_term", "long_term", "short_and_long"}
        if condition not in valid:
            raise ValueError(f"condition must be one of {sorted(valid)}")

        self.condition = condition
        self.short_memory = short_memory or ShortTermMemory(max_messages=8)
        self.long_memory = long_memory
        self.namespace = namespace

        self.base_url = (base_url or os.getenv(
            "UKAHT_LLM_BASE_URL", "http://localhost:11434/v1"
        )).rstrip("/")
        self.model = model or os.getenv("UKAHT_LLM_MODEL", "gemma2")
        self.api_key = api_key or os.getenv("UKAHT_LLM_API_KEY", "ollama")
        self.timeout = timeout

    # -----------------------------------------------------------------------
    # Context construction
    # -----------------------------------------------------------------------

    def _short_context(self) -> str:
        if self.condition not in {"short_term", "short_and_long"}:
            return ""
        text = self.short_memory.as_prompt_text()
        return text.strip()

    def _long_context(self, query: str) -> str:
        if self.condition not in {"long_term", "short_and_long"}:
            return ""
        if self.long_memory is None:
            return ""

        memories = self.long_memory.search(
            query=query,
            namespace=self.namespace,
            top_k=5,
            min_score=0.0,
        )
        if not memories:
            return ""

        lines = []
        for memory in memories:
            lines.append(
                f"- {memory['content']} "
                f"(memory_score={memory['score']:.3f})"
            )
        return "\n".join(lines)

    def build_messages(self, query: str) -> list[dict[str, str]]:
        sections = []

        short_context = self._short_context()
        if short_context:
            sections.append(
                "RECENT CONVERSATION CONTEXT:\n" + short_context
            )

        long_context = self._long_context(query)
        if long_context:
            sections.append(
                "PERSISTENT ARCHIVE/SESSION MEMORY:\n" + long_context
            )

        context = "\n\n".join(sections)
        user_content = query
        if context:
            user_content = f"{context}\n\nCURRENT USER QUERY:\n{query}"

        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # -----------------------------------------------------------------------
    # LLM invocation
    # -----------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def respond(self, query: str, save_turn: bool = True) -> dict[str, Any]:
        messages = self.build_messages(query)
        answer = self._call_llm(messages)

        if save_turn and self.condition in {"short_term", "short_and_long"}:
            self.short_memory.add("user", query)
            self.short_memory.add("assistant", answer)

        return {
            "condition": self.condition,
            "query": query,
            "answer": answer,
            "messages": messages,
        }

    def remember_long_term(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        if self.long_memory is None:
            return None
        return self.long_memory.add(
            content=content,
            namespace=self.namespace,
            metadata=metadata,
        )
