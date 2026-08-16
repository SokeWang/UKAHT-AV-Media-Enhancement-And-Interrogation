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

# Module-level store for assets retrieved by tools during the current turn
_current_retrieved_assets: list[dict] = []

def _format_asset_rag_payload(idx: int, item: dict) -> str:
    title = item.get("title", "Untitled")
    asset_id = item.get("id", "N/A")
    base = item.get("base_code") or "N/A"
    year = item.get("shooting_year") or "N/A"
    cat = item.get("category") or "Antarctic Heritage"
    sub_type = item.get("subject_type") or "N/A"
    raw_caption = item.get("raw_caption") or item.get("description") or "No visual caption available"
    credit = item.get("copyright") or "UKAHT Collection"
    score_str = f" | Score: {round(item.get('score'), 3)}" if item.get("score") is not None else ""

    # Comprehensive multi-attribute metadata string for LLM RAG analysis
    full_meta_summary = (
        f"[Title: {title}] | [Base: Base {base}] | [Year: {year}] | "
        f"[Category: {cat}] | [Subject Type: {sub_type}] | [Photographer: {credit}] | "
        f"[Visual Caption: \"{raw_caption}\"]"
    )

    return (
        f"{idx}. [Photo: {title}] (ID: {asset_id}) | Base: Base {base} | Year: {year} | Category: {cat} | Subject: {sub_type}{score_str}\n"
        f"   Full Multi-Attribute Metadata & Visual RAG Evidence: {full_meta_summary}"
    )


@tool
def semantic_search(query: str, category: str = None) -> str:
    """Search the UKAHT image archive by meaning. Use this when the user asks for images based on visual content or themes."""
    import requests
    global _current_retrieved_assets
    backend_url = os.getenv("BACKEND_API_BASE", "http://localhost:8000")
    payload = {"query": query}
    if category:
        payload["category"] = category
    try:
        resp = requests.post(f"{backend_url}/api/search", json=payload, timeout=25)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        if not raw:
            return "No matching images found in the archive for this query."
            
        _current_retrieved_assets.extend(raw)
        
        # Limit LLM text context to top 6 items enriched with multi-modal RAG visual information
        top_items = raw[:6]
        output_lines = [f"Retrieved {len(raw)} total matching archive images from database. Top {len(top_items)} items with Image-Extracted RAG Information:"]
        for idx, item in enumerate(top_items, 1):
            output_lines.append(_format_asset_rag_payload(idx, item))
        output_lines.append("\nUse these extracted visual details, years, base codes, and photo IDs to synthesize a grounded executive analysis. Cite photo IDs like [ID: asset_id]. Do not re-query for the exact same term.")
        return "\n".join(output_lines)
    except Exception as exc:
        return f"Error executing search: {str(exc)}"


@tool
def sql_query(where_clause: str = None, sql: str = None) -> str:
    """Direct PostgreSQL Query Tool. Generate a raw SQL WHERE clause or SELECT statement to retrieve archive assets.
    
    Database Table Schema: assets
    Columns:
    - base_code: 'E' (Stonington), 'W' (Detaille), 'A' (Port Lockroy)
    - shooting_year: string e.g. '1965', '1958', '2011'
    - category: 'Polar Landscape & Glaciers', 'Artifacts & Museum Display', 'Exterior Heritage & Huts', 'Expedition Equipment & Vessels'
    - subject_type: 'Exterior', 'Artifact', 'Main Hut', 'SfM', 'Landscape', 'Interior'
    - copyright: Photographer credit string
    - title, description: Textual descriptions
    
    Examples:
    - where_clause="base_code = 'E' AND shooting_year < '1980'"
    - where_clause="base_code = 'W' AND category ILIKE '%Huts%'"
    - where_clause="shooting_year IN ('1965', '1990', '2011')"
    - where_clause="base_code = 'E' ORDER BY shooting_year ASC"
    """
    import requests
    global _current_retrieved_assets
    backend_url = os.getenv("BACKEND_API_BASE", "http://localhost:8000")
    payload = {}
    if where_clause: payload["where_clause"] = where_clause
    if sql: payload["sql"] = sql
    try:
        resp = requests.post(f"{backend_url}/api/sql-query", json=payload, timeout=25)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        if not raw:
            return "No matching images found for the specified SQL query."
            
        _current_retrieved_assets.extend(raw)
        
        top_items = raw[:6]
        output_lines = [f"Retrieved {len(raw)} matching images via Direct SQL. Top {len(top_items)} items with Image-Extracted RAG Information:"]
        for idx, item in enumerate(top_items, 1):
            output_lines.append(_format_asset_rag_payload(idx, item))
        output_lines.append("\nUse these extracted visual details, years, base codes, and photo IDs to synthesize a grounded executive analysis. Cite photo IDs like [ID: asset_id].")
        return "\n".join(output_lines)
    except Exception as exc:
        return f"Error executing SQL query: {str(exc)}"


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
    """Filter archive images by structured metadata.
    
    Parameters:
    - base_code: Single letter for Base ('E' for Base E/Stonington, 'W' for Base W/Detaille, 'A' for Base A/Port Lockroy).
    - category: One of ['Polar Landscape & Glaciers', 'Artifacts & Museum Display', 'Exterior Heritage & Huts', 'Expedition Equipment & Vessels'].
    - subject_type: One of ['Exterior', 'Artifact', 'Main Hut', 'SfM', 'Landscape', 'Interior'].
    - shooting_year: Year or mathematical expression e.g. '1958', '1965', '1950s', '<1980', '>1950'. Omit if requesting photos across different years.
    - copyright: Photographer credit substring e.g. 'Mike Cousins', 'Neil Marsden'.
    - keyword: Substring search in title or caption description.
    """
    import requests
    global _current_retrieved_assets
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
        resp = requests.post(f"{backend_url}/api/sql-filter", json=payload, timeout=25)
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        if not raw:
            # Progressive Disclosure Skill Fallback: Automatically fallback to semantic search to prevent empty dead-ends
            fallback_parts = [f"Base {base_code}" if base_code else "", category or "", subject_type or "", keyword or ""]
            fallback_q = " ".join([p for p in fallback_parts if p]).strip() or "Antarctic heritage site"
            return semantic_search.invoke({"query": fallback_q, "category": category})
            
        _current_retrieved_assets.extend(raw)
        
        top_items = raw[:6]
        output_lines = [f"Retrieved {len(raw)} matching images via metadata filter. Top {len(top_items)} items with Image-Extracted RAG Information:"]
        for idx, item in enumerate(top_items, 1):
            output_lines.append(_format_asset_rag_payload(idx, item))
        output_lines.append("\nUse these extracted visual details, years, base codes, and photo IDs to synthesize a grounded executive analysis. Cite photo IDs like [ID: asset_id].")
        return "\n".join(output_lines)
    except Exception as exc:
        return f"Error executing metadata filter: {str(exc)}"


TOOLS = [semantic_search, sql_query, sql_filter]

# ---------------------------------------------------------------------------
# ReAct Agent Prompt & Setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent Deep Track AI Assistant for the UK Antarctic Heritage Trust (UKAHT) historical photograph archive.
You help users search, analyze, and explore historical expedition photographs.
For ANY user query, you MUST call either `semantic_search`, `sql_query`, or `sql_filter` to retrieve relevant archive photos.

Database RAG Schema & Data Statistics:
Table Name: `assets` (1,584 historical photographic assets)
Columns & Types:
- `id`: string asset identifier (e.g. 'ukaht_2c5ce8c29e')
- `title`: string photo title (e.g. 'Scan 0040', 'Base E Exterior')
- `description`: string visual BLIP caption describing exact photo content
- `base_code`: string single letter ('E' = Stonington Island / Base E, 'W' = Detaille Island / Base W, 'A' = Port Lockroy / Base A)
- `category`: string cluster category ('Polar Landscape & Glaciers', 'Artifacts & Museum Display', 'Exterior Heritage & Huts', 'Expedition Equipment & Vessels')
- `subject_type`: string subject category ('Exterior', 'Artifact', 'Main Hut', 'SfM', 'Landscape', 'Interior')
- `shooting_year`: string year or season ('1958', '1965', '1990', '2005', '2011', '2014', '2021-22', '2025')
- `copyright`: string photographer credit (e.g. 'Mike Cousins', 'Gordon MacDonald', 'Neil Marsden')

Text-to-SQL Instructions:
When the user asks structured metadata or historical timeline questions, generate clean PostgreSQL SQL `where_clause` using `sql_query`:
- Example: `sql_query(where_clause="base_code = 'E' AND shooting_year < '1980'")`
- Example: `sql_query(where_clause="base_code = 'E' ORDER BY shooting_year ASC")`
- Example: `sql_query(where_clause="category ILIKE '%Huts%' AND shooting_year BETWEEN '1950' AND '1970'")`
- Example: `sql_query(where_clause="base_code = 'W' AND subject_type = 'Exterior'")`

STRICT ANTI-HALLUCINATION & FACT-GROUNDING RULES:
1. Pure Evidence Grounding: Base your analysis STRICTLY on the retrieved photo captions, years, base codes, and visual descriptions returned by your tools. Do NOT invent fictional dates, unrecorded 100-year timelines, or imaginary historical events.
2. Timeline Accuracy: If the user asks about an extended timeframe (e.g. "how Base E changed in 100 years"), explicitly clarify the exact years present in the archive records (e.g., "The retrieved UKAHT photographic record for Base E covers 1965 to 2025..."), and describe changes ONLY for those recorded years.
3. Citation Requirement: Every visual observation or historical assertion MUST be backed by an explicit photo citation [ID: asset_id].

When formatting your answer:
1. Executive Deep Track Analysis: Provide a structured, insightful summary of historical context and expedition details at the very beginning.
2. Explicit Image Citation & Grounding: Cite all supporting archive photographs explicitly using [ID: asset_id] or [Photo: exact_title] format so users can inspect high-resolution images directly.
3. Be professional, accurate, and focus purely on photographic and historical evidence in the UKAHT collection."""



def _perform_vlm_vision_synthesis(llm: ChatOpenAI, user_query: str, retrieved_assets: list[dict], base_answer: str) -> str:
    """
    Direct VLM Image Vision Pass:
    Passes top retrieved historical photo URLs directly to gemma4:e4b VLM model,
    allowing the AI to observe timber weathering, physical condition, and structural details with its own visual eyes.
    """
    if not retrieved_assets:
        return base_answer

    content = [
        {
            "type": "text",
            "text": (
                f"User Request: '{user_query}'\n\n"
                "You are a Multimodal Vision-Language Model (VLM). Below are the top historical archive photographs directly retrieved from the UKAHT database.\n"
                "Observe these images directly with your visual capabilities. Analyze their physical condition, timber weathering, snow/ice levels, structural changes, and preservation state across the years:\n"
            )
        }
    ]

    valid_images = 0
    for idx, asset in enumerate(retrieved_assets[:4], 1):
        url = asset.get("url")
        if url:
            valid_images += 1
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            })
            content.append({
                "type": "text",
                "text": f"Photo {idx} [ID: {asset.get('id')}] | Title: \"{asset.get('title')}\" | Base: Base {asset.get('base_code')} | Year: {asset.get('shooting_year')}"
            })

    if valid_images == 0:
        return base_answer

    content.append({
        "type": "text",
        "text": "\nProvide a detailed VLM Visual Inspection & Deep Track Analysis based on your direct visual observation of these photos. Cite each photo using [ID: asset_id]."
    })

    try:
        vlm_msg = HumanMessage(content=content)
        vlm_resp = llm.invoke([vlm_msg])
        if vlm_resp and vlm_resp.content and len(vlm_resp.content.strip()) > 30:
            return vlm_resp.content
    except Exception as exc:
        print(f"[VLM Direct Vision Pass Exception]: {exc}")
    return base_answer


class ReActAgent:
    """
    Stateful ReAct agent that maintains conversation history across turns.
    Built on LangChain's tool calling agent framework with VLM Vision Synthesis.
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
        model_name = os.getenv("UKAHT_LLM_MODEL", "gemma4:e4b")
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
        global _current_retrieved_assets
        _current_retrieved_assets = []
        tool_steps: list[dict] = []
        try:
            from langchain_core.utils.uuid import uuid7
            config = {"configurable": {"thread_id": str(uuid7())}}
            
            result = agent.invoke(
                {"messages": messages},
                config=config,
            )
            
            latest_message = result["messages"][-1]
            answer = latest_message.content

            # Look through messages to find tool calls (AIMessage with tool_calls)
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_steps.append({
                            "tool": tc.get("name", "unknown"),
                            "input": tc.get("args", {}),
                            "result_count": len(_current_retrieved_assets)
                        })

            # Perform Direct VLM Vision Synthesis if assets were retrieved
            if _current_retrieved_assets:
                answer = _perform_vlm_vision_synthesis(llm, user_message, _current_retrieved_assets, answer)

        except Exception as exc:
            answer = f"[LLM unavailable: {exc}]"

        # 7. Update internal conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "retrieved_assets": _current_retrieved_assets, "tool_steps": tool_steps}

    def run_stream(self, user_message: str):
        """
        Stream agent execution events step-by-step.
        Yields dicts representing SSE events:
          - {"event": "assets", "data": {"retrieved_assets": [...], "tool_steps": [...]}}
          - {"event": "text", "data": {"chunk": "..."}}
          - {"event": "done", "data": {"answer": "...", "retrieved_assets": [...], "tool_steps": [...]}}
        """
        model_name = os.getenv("UKAHT_LLM_MODEL", "gemma4:e4b")
        base_url = os.getenv("UKAHT_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("UKAHT_LLM_API_KEY", "ollama")

        llm = ChatOpenAI(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0,
            streaming=True,
        )

        messages = []
        for h in self.history[1:]:  # skip system prompt
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))
        messages.append(HumanMessage(content=user_message))

        from langgraph.checkpoint.memory import InMemorySaver
        from langchain_core.utils.uuid import uuid7
        agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

        global _current_retrieved_assets
        _current_retrieved_assets = []
        tool_steps: list[dict] = []
        full_text = ""
        assets_emitted = False
        config = {"configurable": {"thread_id": str(uuid7())}}

        try:
            for chunk, meta in agent.stream({"messages": messages}, config=config, stream_mode="messages"):
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        t_name = tc.get("name", "unknown")
                        t_args = tc.get("args", {})
                        if not any(ts.get("tool") == t_name and ts.get("input") == t_args for ts in tool_steps):
                            tool_steps.append({"tool": t_name, "input": t_args})
                elif hasattr(chunk, "name") and chunk.name in ["semantic_search", "sql_filter"]:
                    # Tool call executed
                    yield {
                        "event": "assets",
                        "data": {
                            "retrieved_assets": list(_current_retrieved_assets),
                            "tool_steps": tool_steps,
                        }
                    }
                    assets_emitted = True
                elif isinstance(chunk, AIMessage) or hasattr(chunk, "content"):
                    if chunk.content and isinstance(chunk.content, str):
                        full_text += chunk.content
                        yield {
                            "event": "text",
                            "data": {"chunk": chunk.content}
                        }
        except Exception as exc:
            if not full_text:
                full_text = f"[LLM unavailable: {str(exc)}]"
                yield {
                    "event": "text",
                    "data": {"chunk": full_text}
                }

        if not assets_emitted:
            yield {
                "event": "assets",
                "data": {
                    "retrieved_assets": list(_current_retrieved_assets),
                    "tool_steps": tool_steps,
                }
            }

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_text})

        yield {
            "event": "done",
            "data": {
                "answer": full_text,
                "retrieved_assets": _current_retrieved_assets,
                "tool_steps": tool_steps,
            }
        }

