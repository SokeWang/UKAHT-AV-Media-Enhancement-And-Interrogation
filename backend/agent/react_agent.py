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
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Tool definitions — using LangChain's @tool decorator
# ---------------------------------------------------------------------------

@tool
def semantic_search(query: str, category: str = None) -> str:
    """Search the UKAHT image archive by meaning. Use this when the user asks for images based on visual content or themes."""
    from backend.retrieval.search import semantic_search as run_semantic_search
    results = run_semantic_search(
        query=query,
        category_filter=category,
    )[:6]  # cap at 6 results for context window economy
    return json.dumps(results, ensure_ascii=False)


@tool
def sql_filter(category: str = None, keyword: str = None) -> str:
    """Filter images by structured metadata such as category or keyword. Use when the user asks for a specific category or keyword match."""
    from backend.retrieval.search import sql_metadata_filter
    results = sql_metadata_filter(
        category=category,
        keyword=keyword,
    )[:6]
    return json.dumps(results, ensure_ascii=False)


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

        # 2. Build prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # 3. Create the agent and agent executor
        agent = create_tool_calling_agent(llm, TOOLS, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            return_intermediate_steps=True,
            verbose=True,
        )

        # 4. Convert history to LangChain message formats
        chat_history_messages = []
        for h in self.history[1:]:  # skip system prompt
            if h["role"] == "user":
                chat_history_messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                chat_history_messages.append(AIMessage(content=h["content"]))

        # 5. Invoke LangChain agent
        retrieved_assets: list[dict] = []
        try:
            result = agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history_messages,
            })
            answer = result.get("output", "")

            # 6. Extract observations from intermediate steps
            for action, observation in result.get("intermediate_steps", []):
                try:
                    assets_list = json.loads(observation)
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
