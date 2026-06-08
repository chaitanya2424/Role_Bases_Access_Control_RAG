"""
streamlit_app.py
================
Demo RAG Assistant - Streamlit UI

Features:
  - User login (simulated) with role selection
  - Chat interface with message history
  - RBAC access control feedback
  - Retrieval trace panel
  - Source citations
  - Confidence indicators
  - Auto-setup: generates data + builds index on first run

Run:
  streamlit run streamlit_app.py
"""

import os
import sys
import time
import streamlit as st

# ---- Page config (must be first Streamlit call) --------------------------------------------------------
st.set_page_config(
    page_title="Demo RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown("""
<style>
/* ---- Global font ---- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ---- Chat messages ---- */
.user-msg {
    background: #1e40af;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: white;
    margin-left: 15%;
}
.bot-msg {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    margin-right: 5%;
}
.denied-msg {
    background: #450a0a;
    border: 1px solid #b91c1c;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #fca5a5;
}

/* ---- Badges ---- */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-low      { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-medium   { background: #1c1917; color: #fbbf24; border: 1px solid #92400e; }
.badge-high     { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
.badge-admin    { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; }
.badge-manager  { background: #0c1a2e; color: #38bdf8; border: 1px solid #0369a1; }
.badge-employee { background: #0f2e1a; color: #34d399; border: 1px solid #065f46; }

/* ---- Confidence bars ---- */
.conf-high   { color: #4ade80; font-weight: 600; }
.conf-medium { color: #fbbf24; font-weight: 600; }
.conf-low    { color: #f87171; font-weight: 600; }
.conf-none   { color: #94a3b8; font-weight: 600; }

/* ---- Citation cards ---- */
.citation-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 13px;
    color: #94a3b8;
}
.citation-card strong { color: #e2e8f0; }

/* ---- Trace steps ---- */
.trace-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #64748b;
    padding: 2px 0;
    border-left: 2px solid #1e40af;
    padding-left: 8px;
    margin: 3px 0;
}

/* ---- Main header ---- */
.main-header {
    text-align: center;
    padding: 20px 0 10px 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 20px;
}
.main-header h1 { color: #e2e8f0; font-size: 28px; font-weight: 600; }
.main-header p  { color: #64748b; font-size: 14px; }

/* ---- Input area ---- */
.stTextInput > div > div > input {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover { background: #1e40af !important; }

/* ---- Expander ---- */
details summary { color: #94a3b8 !important; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# CUSTOM CSS

st.markdown("""
    <div class="main-header">
        <h1>🏢 Demo RAG Assistant</h1>
        <p>Secure knowledge retrieval with Role-Based Access Control · Gemini + ChromaDB + LangChain</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Init log (collapsed) ----------------------------------------------------------------------------------------------
    with st.expander("⚙️ System Initialisation Log", expanded=False):
        for msg in init_messages:
            st.text(msg)

    # ---- Chat history --------------------------------------------------------------------------------------------------------------
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                '<div style="text-align:center;color:#475569;padding:40px 0;">'
                '📩 Ask a question to get started. Your access level determines what you can see.'
                '</div>',
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_message("user", msg["content"])
            else:
                render_rag_response(msg.get("display", {"answer": msg["content"], "denied": msg.get("denied", False)}))

    # ---- Query input ----------------------------------------------------------------------------------------------------------------
    st.markdown("---")

    # Handle prefilled queries from sidebar buttons
    default_query = st.session_state.pop("prefill_query", "")

    col_input, col_btn, col_clear = st.columns([7, 1, 1])

    with col_input:
        query = st.text_input(
            "Your question:",
            value=default_query,
            placeholder="e.g. What is the leave policy? / Show salary information",
            label_visibility="collapsed",
            key="query_input",
        )

    with col_btn:
        send = st.button("Send 🚀", use_container_width=True)

    with col_clear:
        if st.button("Clear 🧹", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---- Process query ------------------------------------------------------------------------------------------------------------
    if (send or default_query) and query.strip():
        # Add user message
        st.session_state.messages.append({
            "role":    "user",
            "content": query.strip(),
        })

        with st.spinner("🔍 Retrieving and generating answer..."):
            response = pipeline.query(query.strip(), user)
            display  = pipeline.format_response_for_display(response)

        # Add assistant message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": response.answer,
            "display": display,
            "denied":  response.denied,
        })

        st.rerun()

    # ---- Role quick-switch notice --------------------------------------------------------------------------------------
    st.markdown(
        '<div style="color:#475569;font-size:12px;text-align:center;margin-top:8px;">'
        '💡 Switch users in the sidebar to test different access levels. '
        'Try "Show salary information" as Employee vs Admin.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()




