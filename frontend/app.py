"""
frontend/app.py
Owner: Chenyu Yuan — Streamlit multi-page application entry point
"""

import streamlit as st

st.set_page_config(
    page_title="UKAHT Antarctic Archive",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar decoration & Page definition
# ---------------------------------------------------------------------------

from frontend.views.smart_search import render as render_search
from frontend.views.annotate import render as render_annotate

search_page = st.Page(
    render_search,
    title="Smart Search & Chat",
    icon="🔍",
    url_path="smart_search",
    default=True,
)
annotate_page = st.Page(
    render_annotate,
    title="Annotate",
    icon="✏️",
    url_path="annotate",
)

# Render custom sidebar header
st.sidebar.title("🧊 UKAHT Archive")
st.sidebar.markdown("UK Antarctic Heritage Trust — Historical Image System")
st.sidebar.divider()

# Run the navigation routing
pg = st.navigation([search_page, annotate_page], position="sidebar")
pg.run()

# Render custom sidebar footer
st.sidebar.divider()
st.sidebar.caption("Team 17 · University of Bristol · 2026")
