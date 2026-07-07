"""
frontend/pages/annotate.py
Owner: Chenyu Yuan — Milestone 2 UI (human-in-the-loop caption annotation)

Responsibilities:
  - Display images from the database one at a time
  - Show the auto-generated BLIP caption in an editable text area
  - Save the corrected caption back to the database via the API
  - Export verified entries to golden_test_set.json
"""

import json
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "golden_test_set.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_assets() -> list[dict]:
    try:
        resp = requests.post(f"{API_BASE}/api/search", json={"query": ""}, timeout=20)
        resp.raise_for_status()
        res_json = resp.json()
        if res_json.get("code") == 200:
            return res_json.get("data", [])
        st.error(f"Could not load assets: {res_json.get('message')}")
        return []
    except Exception as exc:
        st.error(f"Could not load assets: {exc}")
        return []


def _save_caption(asset_id: str, caption: str) -> bool:
    """
    Persist the corrected caption.  Currently calls the search endpoint to
    verify the asset exists; the actual UPDATE goes via the backend DB helper.
    In a full implementation, add a PATCH /api/assets/{id} endpoint.
    """
    # Temporary: write directly using the backend DB module if running locally
    try:
        from backend.db.database import update_asset_description
        update_asset_description(asset_id, caption)
        return True
    except Exception as exc:
        st.error(f"Save failed: {exc}")
        return False


def _load_golden() -> list[dict]:
    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_golden(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render():
    st.title("✏️ Caption Annotation")
    st.markdown(
        "Review auto-generated captions, correct any errors, and mark verified "
        "image-text pairs as part of the **Golden Test Set**."
    )

    assets = _get_all_assets()
    if not assets:
        st.warning("No images in the database yet. Run ingestion first.")
        return

    # Pagination
    total = len(assets)
    idx = st.number_input(
        f"Image (1 – {total})", min_value=1, max_value=total, value=1, step=1
    ) - 1
    asset = assets[idx]

    # Display image
    col_img, col_form = st.columns([1, 2])
    with col_img:
        try:
            img_url = (
                asset["url"]
                if asset["url"].startswith("http")
                else f"{API_BASE}{asset['url']}"
                if asset["url"].startswith("/")
                else asset["url"]
            )
            if "http://backend:8000" in img_url:
                img_url = img_url.replace("http://backend:8000", "http://localhost:8000")
            st.image(img_url, use_container_width=True)
        except Exception:
            st.warning("Image not displayable")
        st.caption(f"**ID:** {asset['id']}")
        st.caption(f"**Category:** {asset.get('category', '—')}")

    with col_form:
        st.subheader(asset.get("title", "Untitled"))

        caption = st.text_area(
            "Caption (edit to correct)",
            value=asset.get("description", ""),
            height=120,
            key=f"caption_{asset['id']}",
        )

        col_save, col_golden = st.columns(2)

        with col_save:
            if st.button("💾 Save Caption", use_container_width=True):
                if _save_caption(asset["id"], caption):
                    st.success("Caption saved.")

        with col_golden:
            if st.button("⭐ Add to Golden Set", use_container_width=True):
                if _save_caption(asset["id"], caption):
                    golden = _load_golden()
                    # Check if this asset already has a golden entry
                    existing_ids = {e["asset_id"] for e in golden}
                    if asset["id"] not in existing_ids:
                        golden.append({
                            "asset_id": asset["id"],
                            "caption": caption,
                            "query": caption,          # use caption as the query
                            "relevant_ids": [asset["id"]],
                        })
                        _save_golden(golden)
                        st.success(f"Added to golden set ({len(golden)} entries).")
                    else:
                        st.info("Already in golden set.")

    # Golden test-set summary
    st.divider()
    golden = _load_golden()
    st.markdown(f"**Golden Test Set:** {len(golden)} / 50–100 entries verified")
    if golden:
        st.progress(min(len(golden) / 100, 1.0))
        if st.button("📥 Download golden_test_set.json"):
            st.download_button(
                label="Download",
                data=json.dumps(golden, indent=2, ensure_ascii=False),
                file_name="golden_test_set.json",
                mime="application/json",
            )
