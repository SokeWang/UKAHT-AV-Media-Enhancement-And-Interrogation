"""
backend/agent/react_agent.py
Owner: Tian Luo — Milestone 4 (LLM agent + RAG Q&A)

Responsibilities:
  - Implement a ReAct-style agent loop using LangChain
  - Register two tools:
      * semantic_search  — vector similarity search via retrieval/search.py
      * sql_filter       — structured metadata SQL query
  - Maintain multi-turn conversation history
  - Feed retrieved context back to LLM to produce a grounded answer
"""

import json
import os
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ---------------------------------------------------------------------------
# Tool definitions — using LangChain's @tool decorator
# ---------------------------------------------------------------------------

@tool
def semantic_search(query: str, category: str = None) -> str:
    """Search the UKAHT image archive by meaning. Use this when the user asks for images based on visual content or themes."""
    import requests
    backend_url = os.getenv("BACKEND_API_BASE", "http://localhost:8000")
    payload = {"query": query}
    if category:
        payload["category"] = category
    try:
        resp = requests.post(f"{backend_url}/api/search", json=payload, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("data", [])[:6]  # cap at 6 results for context window economy
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@tool
def sql_filter(
    category: str = None,
    keyword: str = None,
    base_code: str = None,
    subject_type: str = None,
    shooting_year: str = None,
    copyright: str = None,
    data_source: str = None
) -> str:
    """Filter images by structured metadata. Use when the user asks for specific attributes like years, bases, subjects, copyright credit, or data source.
    
    Args:
        category:      Exact category (case-insensitive).
        keyword:       General search string in title/description.
        base_code:     Base letter (e.g. 'E' for Base E, 'W' for Base W, 'A' for Base A).
        subject_type:  Subject name (e.g. 'Exterior', 'Main Hut', 'Artifact', 'SfM').
        shooting_year: Year/Season string (e.g. '1958', '2011-12', '2025').
        copyright:     Credit or copyright owner (e.g. 'Mike Cousins', 'Gordon MacDonald').
        data_source:   Where the data is from ('original' or 'new_addition').
    """
    import requests
    backend_url = os.getenv("BACKEND_API_BASE", "http://localhost:8000")
    payload = {}
    if category: payload["category"] = category
    if keyword: payload["keyword"] = keyword
    if base_code: payload["base_code"] = base_code
    if subject_type: payload["subject_type"] = subject_type
    if shooting_year: payload["shooting_year"] = shooting_year
    if copyright: payload["copyright"] = copyright
    if data_source: payload["data_source"] = data_source
    try:
        resp = requests.post(f"{backend_url}/api/sql-filter", json=payload, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("data", [])[:6]
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


TOOLS = [semantic_search, sql_filter]

# ---------------------------------------------------------------------------
# ReAct Agent Prompt & Setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent assistant for the UK Antarctic Heritage Trust (UKAHT)
image archive. You help users find historical expedition photographs and answer questions about
the collection. When relevant, use the provided tools to search the archive before answering.
Always cite the image titles you retrieved to support your answer."""


class ReActAgent:
    """
    Stateful ReAct agent that maintains conversation history across turns.
    Built on LangChain's tool calling agent framework.
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
        model_name = os.getenv("UKAHT_LLM_MODEL", "gemma2")
        base_url = os.getenv("UKAHT_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("UKAHT_LLM_API_KEY", "ollama")

        # 1. Initialize LangChain model
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0,
        )

        # 2. Convert history to LangChain message formats
        messages = []
        for h in self.history[1:]:  # skip system prompt
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))
        messages.append(HumanMessage(content=user_message))

        # 3. Create the agent with checkpointer
        from langgraph.checkpoint.memory import InMemorySaver
        agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

        # 4. Invoke agent
        retrieved_assets: list[dict] = []
        try:
            from langchain_core.utils.uuid import uuid7
            config = {"configurable": {"thread_id": str(uuid7())}}
            
            result = agent.invoke(
                {"messages": messages},
                config=config,
            )
            
            latest_message = result["messages"][-1]
            answer = latest_message.content

            # Look through messages to find tool call outputs
            for msg in result["messages"]:
                if msg.type == "tool" or isinstance(msg, ToolMessage):
                    try:
                        assets_list = json.loads(msg.content)
                        if isinstance(assets_list, list):
                            retrieved_assets.extend(assets_list)
                    except Exception:
                        pass

        except Exception as exc:
            answer = f"[LLM unavailable: {exc}]"

        # 7. Update internal conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "retrieved_assets": retrieved_assets}
