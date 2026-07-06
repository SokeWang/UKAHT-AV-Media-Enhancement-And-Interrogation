"""
backend/agent/react_agent.py
Owner: Tian Luo — Milestone 4 (LLM agent + RAG Q&A)

Responsibilities:
  - Implement a ReAct-style agent loop:
      Thought → Action (tool call) → Observation → ... → Answer
  - Register two tools:
      * semantic_search_tool  — vector similarity search via retrieval/search.py
      * sql_filter_tool       — structured metadata SQL query
  - Maintain multi-turn conversation history
  - Feed retrieved context back to LLM to produce a grounded answer

Configuration:
  Set the environment variable UKAHT_LLM_MODEL to choose the LLM backend.
  Supported values (via Ollama by default):
    "gemma2"       (default, lightweight, runs locally)
    "llama3.1"
    "mistral"
  For OpenAI-compatible APIs, set UKAHT_LLM_BASE_URL and UKAHT_LLM_API_KEY.
"""

import json
import os
from typing import Any

from backend.retrieval.search import semantic_search, sql_metadata_filter

# ---------------------------------------------------------------------------
# Tool definitions — passed to the LLM as callable functions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "semantic_search",
        "description": (
            "Search the UKAHT image archive by meaning. "
            "Use this when the user asks for images based on visual content or themes. "
            "Input: a natural-language query string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search phrase"},
                "category": {
                    "type": "string",
                    "description": "Optional category filter, e.g. 'Landscape'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "sql_filter",
        "description": (
            "Filter images by structured metadata such as category or keyword. "
            "Use when the user asks for a specific category or keyword match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category name"},
                "keyword": {"type": "string", "description": "Keyword in title or description"},
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, args: dict) -> list[dict]:
    """Dispatch a tool call and return results."""
    if tool_name == "semantic_search":
        return semantic_search(
            query=args.get("query", ""),
            category_filter=args.get("category"),
        )[:6]  # cap at 6 results for context window economy
    elif tool_name == "sql_filter":
        return sql_metadata_filter(
            category=args.get("category"),
            keyword=args.get("keyword"),
        )[:6]
    else:
        return []


# ---------------------------------------------------------------------------
# LLM call (Ollama by default, easily swapped)
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict], tools: list | None = None) -> dict:
    """
    Call the configured LLM. Returns a dict with keys:
      'content'    (str | None)
      'tool_calls' (list of {name, arguments} | None)
    """
    model = os.getenv("UKAHT_LLM_MODEL", "gemma2")
    base_url = os.getenv("UKAHT_LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("UKAHT_LLM_API_KEY", "ollama")

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]

        return {"content": msg.content, "tool_calls": tool_calls}

    except Exception as exc:
        return {"content": f"[LLM unavailable: {exc}]", "tool_calls": None}


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent assistant for the UK Antarctic Heritage Trust (UKAHT)
image archive. You help users find historical expedition photographs and answer questions about
the collection. When relevant, use the provided tools to search the archive before answering.
Always cite the image titles you retrieved to support your answer."""


class ReActAgent:
    """
    Stateful ReAct agent that maintains conversation history across turns.

    Usage:
        agent = ReActAgent()
        response = agent.run("Show me landscape photos from the expedition")
        print(response["answer"])
        print(response["retrieved_assets"])  # list of matching image dicts
    """

    def __init__(self):
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        """Clear conversation history."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_message: str) -> dict:
        """
        Process one user turn.

        Returns:
            {
              "answer":           str — final LLM response,
              "retrieved_assets": list[dict] — images found via tools,
            }
        """
        self.history.append({"role": "user", "content": user_message})
        retrieved_assets: list[dict] = []

        # --- ReAct loop (max 3 iterations to avoid infinite loops) ----------
        for _ in range(3):
            llm_resp = _call_llm(self.history, tools=TOOLS)

            if llm_resp["tool_calls"]:
                # Execute each requested tool
                for tc in llm_resp["tool_calls"]:
                    results = _execute_tool(tc["name"], tc["arguments"])
                    retrieved_assets.extend(results)

                    # Feed observation back as an assistant message
                    observation = json.dumps(
                        [{"id": r["id"], "title": r["title"],
                          "description": r["description"]} for r in results],
                        ensure_ascii=False,
                    )
                    self.history.append({
                        "role": "assistant",
                        "content": f"[Tool: {tc['name']}] Results: {observation}",
                    })
            else:
                # LLM produced a final answer — exit loop
                answer = llm_resp["content"] or ""
                self.history.append({"role": "assistant", "content": answer})
                return {"answer": answer, "retrieved_assets": retrieved_assets}

        # Fallback if loop exhausted without a text answer
        fallback = "I found some results but could not summarise them. Please refine your query."
        self.history.append({"role": "assistant", "content": fallback})
        return {"answer": fallback, "retrieved_assets": retrieved_assets}
