import streamlit as st

# ---------------------- TEMA Y CABECERAS UI (Globales) ----------------------

def setup_ui_theme(theme: str = "light"):
    is_dark = theme == "dark"
    primary = "#5B8DEF"
    accent = "#F59E0B"
    bg_light = "#F7F9FC"
    bg_dark = "#0B1220"
    text_light = "#111827"
    text_dark = "#E5E7EB"
    card_light = "#FFFFFF"
    card_dark = "#111827"
    border_light = "#eaecef"
    border_dark = "#334155"

    bg = bg_dark if is_dark else bg_light
    text = text_dark if is_dark else text_light
    card = card_dark if is_dark else card_light
    border = border_dark if is_dark else border_light

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif !important;
        }}
        :root {{ --primary:{primary}; --accent:{accent}; --bg:{bg}; --text:{text}; --card:{card}; --border:{border}; }}
        .stApp {{ background: var(--bg); color: var(--text); }}
        .hero {{padding:1rem 1.25rem; border-radius:14px; background: linear-gradient(90deg,#5B8DEF 0%, #7C3AED 100%); color:#fff; margin: 0 0 1rem 0; box-shadow: 0 6px 20px rgba(91,141,239,.25);}}
        .hero .title {{font-size:1.4rem; font-weight:800; letter-spacing:.2px;}}
        .hero .subtitle {{opacity:.95; font-size:.95rem; margin-top:.25rem;}}
        .card {{background:var(--card); padding:1rem 1.2rem; border-radius:14px; border:1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom:1rem;}}
        .sidebar-card {{padding:.7rem .8rem; background:{'#0F172A' if is_dark else '#eef2ff'}; border-radius:12px; margin-bottom:.75rem; border:1px solid {('#1F2937' if is_dark else '#dbeafe')}; color: var(--text);}}
        .badge {{display:inline-block; background:{'#334155' if is_dark else '#e5e7eb'}; color:{'#E5E7EB' if is_dark else '#111827'}; padding:.18rem .5rem; border-radius:8px; font-size:.8rem; margin-right:.35rem;}}
        .section-title {{font-size:1.2rem; font-weight:700; margin-bottom:.4rem; color:var(--text);}}
        .stButton>button {{background:var(--primary); color:#fff; border:none; border-radius:10px; padding:.5rem 1rem; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,.15);}}
        .stButton>button:hover {{background:#4A76D1}}
        .download-btn .stButton>button {{background:var(--accent);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", emoji: str = ""):
    st.markdown(
        f"""
        <div class='hero'>
            <div class='title'>{emoji} {title}</div>
            <div class='subtitle'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )