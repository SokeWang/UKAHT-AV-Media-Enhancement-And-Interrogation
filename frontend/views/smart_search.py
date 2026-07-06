"""
frontend/pages/smart_search.py
Owner: Chenyu Yuan & Tian Luo — Unified Smart Search and Conversational Q&A Portal

This page implements:
  - Top Search: User enters search term or query.
  - Image Gallery: Renders all images on first load, or direct search results in a grid.
  - Interactive Q&A: Split panel displaying the conversational AI agent alongside the search results.
  - Upload panel: Indexing new photos.
"""

import os
import uuid
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

def _search(query: str, category: str = "") -> list[dict]:
    payload = {"query": query}
    if category:
        payload["category"] = category
    try:
        resp = requests.post(f"{API_BASE}/api/search", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return []


def _recommend(asset_id: str, limit: int = 4) -> list[dict]:
    try:
        resp = requests.post(
            f"{API_BASE}/api/recommend",
            json={"id": asset_id, "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Recommendation failed: {exc}")
        return []


def _upload_and_index(file) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/api/upload-and-index",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
        return None


def _get_demo_reply(message: str, session_id: str) -> dict:
    msg_lower = message.lower()
    
    # Try to fetch actual relevant images from the database for the demo query
    retrieved_assets = []
    try:
        resp = requests.post(f"{API_BASE}/api/search", json={"query": message}, timeout=5)
        if resp.status_code == 200:
            retrieved_assets = resp.json()[:3]  # cap at 3 for inline chat reference
    except Exception:
        pass

    if "dog" in msg_lower or "sledge" in msg_lower:
        answer = (
            "🐕 **[Demo Mode - Simulated Response]**\n\n"
            "Dogs (specifically huskies) were indispensable partners in British Antarctic expeditions. "
            "They were primarily used for pulling sledges loaded with geological equipment and food rations "
            "across crevassed terrain where vehicles could not go. "
            "The photos show sledge dogs resting near the tents at Base E (Stonington Island) "
            "and active dog teams during field surveys."
        )
    elif "people" in msg_lower or "man" in msg_lower or "crew" in msg_lower or "person" in msg_lower:
        answer = (
            "👥 **[Demo Mode - Simulated Response]**\n\n"
            "I found several images depicting expedition personnel. These include scientists, station leaders, "
            "and radio operators engaged in daily station maintenance, meteorological observations, "
            "and recreational activities during the long polar winters."
        )
    elif "building" in msg_lower or "station" in msg_lower or "base" in msg_lower or "hut" in msg_lower:
        answer = (
            "🏠 **[Demo Mode - Simulated Response]**\n\n"
            "The archive contains historical views of early British scientific bases, such as Port Lockroy (Base A) "
            "and Horseshoe Island (Base Y). These structures were built with timber panels and served as "
            "living quarters, laboratories, and radio stations."
        )
    else:
        answer = (
            "❄️ **[Demo Mode - Simulated Response]**\n\n"
            f"Regarding your query about **\"{message}\"**:\n"
            "I searched the archive and found matching historical visual resources. "
            "The collection documents early mid-20th century expedition life, scientific instrumentation, "
            "and polar geography under the Falkland Islands Dependencies Survey (FIDS).\n\n"
            "*Note: The Ollama LLM backend is currently offline. This is a pre-configured demo reply.*"
        )
        
    return {
        "answer": answer,
        "retrieved_assets": retrieved_assets,
        "session_id": session_id
    }


def _send_message(message: str, session_id: str) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/api/agent/chat",
            json={"message": message, "session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Check if the backend completed successfully but reported LLM unavailable
        if "[LLM unavailable" in data.get("answer", ""):
            return _get_demo_reply(message, session_id)
            
        return data
    except Exception:
        # Fall back to custom demo answers if the backend server is offline entirely
        return _get_demo_reply(message, session_id)


def _reset_session(session_id: str):
    try:
        requests.delete(f"{API_BASE}/api/agent/session/{session_id}", timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def _display_grid(assets: list[dict], cols: int = 3) -> str | None:
    """
    Render assets as a clean responsive image grid.
    Returns the ID of the asset clicked, if any.
    """
    clicked_id = None
    rows = [assets[i : i + cols] for i in range(0, len(assets), cols)]
    for row in rows:
        columns = st.columns(cols)
        for col, asset in zip(columns, row):
            with col:
                try:
                    img_url = (
                        asset["url"]
                        if asset["url"].startswith("http")
                        else f"{API_BASE}{asset['url']}"
                        if asset["url"].startswith("/")
                        else asset["url"]
                    )
                    st.image(img_url, use_container_width=True)
                except Exception:
                    st.warning("Image unavailable")
                st.caption(f"**{asset['title']}**")
                st.caption(
                    f"_{asset.get('category', '')}_ — score: {asset.get('score', '—')}"
                )
                # Cap description length to keep the grid tidy
                st.caption(asset.get("description", "")[:80] + "...")
                if st.button("Similar →", key=f"rec_{asset['id']}"):
                    clicked_id = asset["id"]
    return clicked_id


def _render_asset_thumbnails(assets: list[dict]):
    """Render small thumbnails inline in chat bubbles."""
    if not assets:
        return
    cols = st.columns(min(len(assets), 4))
    for col, asset in zip(cols, assets[:4]):
        with col:
            try:
                img_url = (
                    asset["url"]
                    if asset["url"].startswith("http")
                    else f"{API_BASE}{asset['url']}"
                    if asset["url"].startswith("/")
                    else asset["url"]
                )
                st.image(img_url, use_container_width=True)
            except Exception:
                pass
            st.caption(asset.get("title", "")[:30])


# ---------------------------------------------------------------------------
# Session Initialization
# ---------------------------------------------------------------------------

def _init_session():
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = f"sess_{uuid.uuid4().hex[:8]}"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_query" not in st.session_state:
        st.session_state.active_query = ""
    if "active_category" not in st.session_state:
        st.session_state.active_category = ""
    if "similar_for_id" not in st.session_state:
        st.session_state.similar_for_id = None
    if "last_top_query" not in st.session_state:
        st.session_state.last_top_query = ""


# ---------------------------------------------------------------------------
# Page Render
# ---------------------------------------------------------------------------

def render():
    st.title("🔍 Unified Smart Search & Q&A Portal")
    st.markdown(
        "Search the UKAHT historical photograph archive instantly, "
        "and ask follow-up questions to explore stories behind the collection."
    )

    _init_session()

    # --- Search Input Form --------------------------------------------------
    col_q, col_cat, col_btn = st.columns([3, 1.2, 1])
    with col_q:
        query_input = st.text_input(
            "Search or Ask",
            value=st.session_state.active_query,
            placeholder="e.g. dogs, station leaders, or Ask 'What was the task of Milestone 3?'",
        )
    with col_cat:
        cat_options = ["", "Landscape", "Building", "Equipment", "Wildlife", "People", "Uploaded"]
        try:
            default_cat_idx = cat_options.index(st.session_state.active_category)
        except ValueError:
            default_cat_idx = 0
        category_input = st.selectbox(
            "Category Filter",
            cat_options,
            index=default_cat_idx,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_search = st.button("Search & Ask", use_container_width=True)

    # Process search submission
    if submit_search or (query_input != st.session_state.active_query):
        st.session_state.active_query = query_input
        st.session_state.active_category = category_input
        st.session_state.similar_for_id = None
        
        # Reset LLM Agent Session for a brand new search term to start a clean conversational topic
        if query_input != st.session_state.last_top_query:
            _reset_session(st.session_state.chat_session_id)
            st.session_state.chat_session_id = f"sess_{uuid.uuid4().hex[:8]}"
            st.session_state.chat_history = []
            st.session_state.last_top_query = query_input

    # --- Upload Expander ----------------------------------------------------
    with st.expander("📤 Upload & Index a New Image"):
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded and st.button("Upload & Index"):
            with st.spinner("Captioning and indexing…"):
                result = _upload_and_index(uploaded)
            if result:
                st.success(f"Indexed: **{result['title']}**")
                st.caption(result.get("caption", ""))

    st.divider()

    # --- Content Layout -----------------------------------------------------
    if not st.session_state.active_query:
        # Initial Load: Show full gallery
        st.subheader("Archived Historical Photos")
        with st.spinner("Loading archive…"):
            all_assets = _search("")
        if all_assets:
            st.markdown(f"**{len(all_assets)} images** in the archive")
            clicked_id = _display_grid(all_assets, cols=3)
            if clicked_id:
                st.session_state.similar_for_id = clicked_id
                st.rerun()

        # Show similar images if requested
        if st.session_state.similar_for_id:
            st.divider()
            st.subheader(f"Images Similar to {st.session_state.similar_for_id}")
            with st.spinner("Finding similar images…"):
                recs = _recommend(st.session_state.similar_for_id)
            if recs:
                _display_grid(recs, cols=4)
            else:
                st.info("No similar images found.")
    else:
        # Split Layout: Direct Search on Left, AI Chat on Right
        col_left, col_right = st.columns([5, 4])

        # ----------------------------------------------------
        # Left Side: Direct Retrieval
        # ----------------------------------------------------
        with col_left:
            st.subheader("🔍 Retrieval Results")
            with st.spinner("Searching archive..."):
                results = _search(st.session_state.active_query, st.session_state.active_category)

            if results:
                st.markdown(f"Found **{len(results)} images** for *{st.session_state.active_query}*")
                clicked_id = _display_grid(results, cols=2)
                if clicked_id:
                    st.session_state.similar_for_id = clicked_id
                    st.rerun()
            else:
                st.info("No images match the query directly.")

            # Show similar images in sub-layout if requested
            if st.session_state.similar_for_id:
                st.divider()
                st.subheader("Similar Images")
                with st.spinner("Finding similar images…"):
                    recs = _recommend(st.session_state.similar_for_id)
                if recs:
                    _display_grid(recs, cols=2)
                else:
                    st.info("No similar images found.")

        # ----------------------------------------------------
        # Right Side: AI Assistant & Q&A
        # ----------------------------------------------------
        with col_right:
            st.subheader("💬 AI Assistant Analysis")
            
            # If search query has changed, trigger the initial AI answer
            if not st.session_state.chat_history:
                with st.spinner("AI is analyzing search topic..."):
                    result = _send_message(st.session_state.active_query, st.session_state.chat_session_id)
                    answer = result.get("answer", "No answer returned.")
                    assets = result.get("retrieved_assets", [])
                    st.session_state.chat_history.append(
                        {"role": "user", "content": st.session_state.active_query, "assets": []}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "assets": assets}
                    )
                    st.rerun()

            # Session Reset Control
            col_sess, col_reset = st.columns([3, 1])
            with col_sess:
                st.caption(f"Session Context: `{st.session_state.chat_session_id}`")
            with col_reset:
                if st.button("Reset Chat", use_container_width=True):
                    _reset_session(st.session_state.chat_session_id)
                    st.session_state.chat_session_id = f"sess_{uuid.uuid4().hex[:8]}"
                    st.session_state.chat_history = []
                    st.rerun()

            # Render Chat History
            chat_container = st.container(height=400)
            with chat_container:
                for turn in st.session_state.chat_history:
                    if turn["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(turn["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(turn["content"])
                            if turn.get("assets"):
                                _render_asset_thumbnails(turn["assets"])

            # Follow-up conversational input
            follow_up = st.chat_input("Ask a follow-up question…", key="follow_up_query")
            if follow_up:
                # Add to history and query AI
                st.session_state.chat_history.append(
                    {"role": "user", "content": follow_up, "assets": []}
                )
                with st.spinner("AI is thinking..."):
                    result = _send_message(follow_up, st.session_state.chat_session_id)
                    answer = result.get("answer", "No answer returned.")
                    assets = result.get("retrieved_assets", [])
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "assets": assets}
                    )
                st.rerun()
