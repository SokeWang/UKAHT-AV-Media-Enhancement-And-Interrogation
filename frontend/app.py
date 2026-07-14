import os
import streamlit as st

st.set_page_config(
    page_title="UKAHT Antarctic Archive",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentication Flow
# ---------------------------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def render_login():
    st.markdown(
        """
        <style>
        .login-box {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2.5rem;
            border-radius: 12px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-top: 15%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>🧊 UKAHT Antarctic Archive</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; margin-bottom: 2rem;'>UK Antarctic Heritage Trust — Historical Image System</p>", unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        admin_user = os.getenv("UKAHT_ADMIN_USER", "admin")
        admin_password = os.getenv("UKAHT_ADMIN_PASSWORD", "ukaht2026")
        
        if st.button("Login", use_container_width=True, type="primary"):
            if username == admin_user and password == admin_password:
                st.session_state["logged_in"] = True
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    render_login()
    st.stop()

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

# Logout button
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state["logged_in"] = False
    st.rerun()

st.sidebar.caption("Team 17 · University of Bristol · 2026")
