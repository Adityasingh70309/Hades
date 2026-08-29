import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from text_pipeline import classify_text
from slang_module import get_slang_context

st.set_page_config(
    page_title="HADES",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "history" not in st.session_state:
    st.session_state.history = []

dark = st.session_state.dark_mode

if dark:
    bg         = "#0f1117"
    card_bg    = "#1a1d2e"
    card_bg2   = "#13151f"
    border     = "#2a2d45"
    text       = "#e8e8f8"
    muted      = "#6b6d8a"
    accent     = "#3b6ef6"
    accent2    = "#2952d9"
    grad_top   = "#0f1117"
    grad_bot   = "#0a0c16"
    shadow     = "rgba(0,0,0,0.5)"
    bar_bg     = "#232640"
else:
    bg         = "#e8edf8"
    card_bg    = "#ffffff"
    card_bg2   = "#f4f6fd"
    border     = "#dde2f5"
    text       = "#1a1d35"
    muted      = "#8890b0"
    accent     = "#3b6ef6"
    accent2    = "#2952d9"
    grad_top   = "#c8d8f8"
    grad_bot   = "#e8edf8"
    shadow     = "rgba(60,80,180,0.12)"
    bar_bg     = "#e8ecf8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}

.stApp {{
    background: linear-gradient(160deg, {grad_top} 0%, {grad_bot} 100%);
    min-height: 100vh;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 1.5rem 1rem 4rem 1rem;
    max-width: 680px;
}}

.stTextArea {{
    margin-bottom: 0.75rem !important;
}}

.stTextArea [data-baseweb="textarea"],
.stTextArea div[data-testid="stTextAreaRootElement"] {{
    background: {card_bg} !important;
    border: 1.5px solid {border} !important;
    border-radius: 18px !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: 0 10px 30px {shadow}, inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.stTextArea [data-baseweb="textarea"]:hover,
.stTextArea div[data-testid="stTextAreaRootElement"]:hover {{
    border-color: rgba(59, 110, 246, 0.45) !important;
    box-shadow: 0 12px 35px {shadow}, 0 0 0 1px rgba(59, 110, 246, 0.15) !important;
}}

.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextArea div[data-testid="stTextAreaRootElement"]:focus-within {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 3px rgba(59, 110, 246, 0.22), 0 14px 40px {shadow} !important;
    background: {card_bg} !important;
}}

.stTextArea [data-baseweb="base-input"],
.stTextArea [data-baseweb="textarea"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

.stTextArea textarea {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: {text} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    font-weight: 400 !important;
    padding: 0.1rem 0.2rem !important;
    resize: none !important;
    box-shadow: none !important;
    outline: none !important;
    line-height: 1.65 !important;
}}

.stTextArea textarea::placeholder {{
    color: {muted} !important;
    opacity: 0.75 !important;
    font-weight: 400 !important;
}}

.stTextArea textarea:focus {{
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}

.stTextArea textarea::-webkit-scrollbar {{
    width: 6px;
}}
.stTextArea textarea::-webkit-scrollbar-track {{
    background: transparent;
}}
.stTextArea textarea::-webkit-scrollbar-thumb {{
    background: {border};
    border-radius: 10px;
}}
.stTextArea textarea::-webkit-scrollbar-thumb:hover {{
    background: {muted};
}}

/* Hide default 'Press Ctrl+Enter' helper instructions */
[data-testid="InputInstructions"],
.stTextArea [data-testid="InputInstructions"],
div[data-testid="InputInstructions"],
.stTextArea small {{
    display: none !important;
}}

/* buttons */
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {accent} 0%, {accent2} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.6rem 1.6rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 18px rgba(59,110,246,0.35) !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
}}
.stButton button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(135deg, #4d7cf7 0%, #305de8 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(59,110,246,0.5) !important;
}}
.stButton button[kind="primary"]:active, .stButton button[data-testid="stBaseButton-primary"]:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px rgba(59,110,246,0.3) !important;
}}

.stButton button[kind="secondary"], .stButton button[data-testid="stBaseButton-secondary"] {{
    background: {card_bg} !important;
    color: {text} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 1.05rem !important;
    box-shadow: 0 2px 8px {shadow} !important;
    transition: all 0.2s ease !important;
}}
.stButton button[kind="secondary"]:hover, .stButton button[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {accent} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px {shadow} !important;
}}

/* general fallback for buttons */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}

/* result card */
.result-card {{
    background: {card_bg};
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 10px 40px {shadow};
    border: 1px solid {border};
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
}}

/* badges */
.result-badge {{
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}}
.badge-hate {{ background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.25); }}
.badge-offensive {{ background: rgba(255,154,60,0.12); color: #ff9a3c; border: 1px solid rgba(255,154,60,0.25); }}
.badge-normal {{ background: rgba(74,222,128,0.1); color: #22c55e; border: 1px solid rgba(74,222,128,0.2); }}

.result-label {{
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 1rem 0;
    line-height: 1;
}}
.label-hate {{ color: #ff4d6d; }}
.label-offensive {{ color: #ff9a3c; }}
.label-normal {{ color: #22c55e; }}

/* confidence bars */
.conf-row {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.45rem 0;
}}
.conf-label {{
    width: 90px;
    font-size: 0.72rem;
    color: {muted};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}
.conf-bar-bg {{
    flex: 1;
    height: 5px;
    background: {bar_bg};
    border-radius: 99px;
    overflow: hidden;
}}
.conf-bar-fill {{ height: 100%; border-radius: 99px; }}
.conf-value {{
    width: 38px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: {muted};
    text-align: right;
}}

/* shap pills */
.shap-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
}}
.shap-pill {{
    padding: 0.22rem 0.65rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}}
.shap-pos {{ background: rgba(255,77,109,0.1); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.2); }}
.shap-neg {{ background: rgba(74,222,128,0.08); color: #22c55e; border: 1px solid rgba(74,222,128,0.18); }}

/* slang */
.slang-card {{
    background: {card_bg2};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
}}
.slang-term {{ color: {accent}; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 500; }}
.slang-def {{ color: {muted}; font-size: 0.8rem; margin-top: 0.2rem; line-height: 1.5; }}

/* section label */
.sec-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {muted};
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    opacity: 0.7;
}}

/* history */
.hist-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 0.7rem 1rem;
    margin: 0.35rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.8rem;
}}
.hist-text {{
    color: {muted};
    font-size: 0.82rem;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* checkbox */
.stCheckbox label {{
    color: {muted} !important;
    font-size: 0.82rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ---- toggle button ----
_, tcol = st.columns([8, 1])
with tcol:
    icon = "☀️" if dark else "🌙"
    if st.button(icon, key="toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# ---- hero ----
st.markdown(f"""
<div style="text-align:center; padding: 1rem 0 1.6rem 0;">
    <p style="font-size:0.72rem; font-weight:600; letter-spacing:0.25em; text-transform:uppercase; color:{accent}; margin:0 0 0.5rem 0;">
        Hate speech Detection Engineered System
    </p>
    <h1 style="font-size:4.2rem; font-weight:800; letter-spacing:-0.04em; color:{text}; margin:0; line-height:1;">
        HADES
    </h1>
</div>
""", unsafe_allow_html=True)


# ---- main input ----
text_input = st.text_area(
    label="Input text to analyze",
    placeholder="Type or paste text here to analyze for hate speech, offensive content, or slang...",
    height=125,
    label_visibility="collapsed",
    key="input"
)

col_opt, col_btn = st.columns([1.4, 1], vertical_alignment="center")
with col_opt:
    run_shap = st.checkbox("SHAP explanation (slower)", value=False)
with col_btn:
    analyze = st.button("Analyze →", type="primary", use_container_width=True)


# ---- result ----
if analyze and text_input.strip():
    with st.spinner(""):
        result = classify_text(text_input.strip(), explain=run_shap)
        slang = get_slang_context(text_input.strip()) if result["confidence"] < 0.75 else None

    st.session_state.history.append({"text": text_input.strip(), "label": result["label"]})

    label    = result["label"]
    probs    = result["probabilities"]
    ldisplay = "Hate Speech" if label == "hatespeech" else label.capitalize()
    lcls     = {"hatespeech":"label-hate","offensive":"label-offensive","normal":"label-normal"}[label]
    bcls     = {"hatespeech":"badge-hate","offensive":"badge-offensive","normal":"badge-normal"}[label]
    bcolors  = {"hatespeech":"#ff4d6d","offensive":"#ff9a3c","normal":"#22c55e"}
    blabels  = {"hatespeech":"Hate Speech","offensive":"Offensive","normal":"Normal"}

    conf_rows_html = "".join([
        f"""<div class="conf-row">
            <span class="conf-label">{blabels[lbl]}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{int(prob * 100)}%;background:{bcolors[lbl]};"></div>
            </div>
            <span class="conf-value">{int(prob * 100)}%</span>
        </div>"""
        for lbl, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)
    ])

    st.markdown(f"""
    <div class="result-card">
        <span class="result-badge {bcls}">{ldisplay}</span>
        <p class="result-label {lcls}">{ldisplay}</p>
        <div class="sec-label">Confidence Breakdown</div>
        {conf_rows_html}
    </div>
    """, unsafe_allow_html=True)

    if run_shap and "top_contributing_words" in result:
        st.markdown('<div class="sec-label">Key Words · SHAP</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="shap-pill {"shap-pos" if v>0 else "shap-neg"}">{w} ({v:+.3f})</span>'
            for w, v in result["top_contributing_words"][:6]
        )
        st.markdown(f'<div class="shap-wrap">{pills}</div>', unsafe_allow_html=True)

    if slang and slang["oov_tokens"]:
        valid = [(t,d) for t,d in slang["definitions"].items() if d and t.lower() not in d.lower()[:50]]
        if valid:
            st.markdown('<div class="sec-label">Slang / OOV Detected</div>', unsafe_allow_html=True)
            for term, defn in valid:
                st.markdown(f"""
                <div class="slang-card">
                    <span class="slang-term">{term}</span>
                    <p class="slang-def">{defn[:120]}...</p>
                </div>""", unsafe_allow_html=True)

elif analyze and not text_input.strip():
    st.warning("Please type something first.")


# ---- history ----
if st.session_state.history:
    st.markdown('<div class="sec-label">Recent</div>', unsafe_allow_html=True)
    for item in reversed(st.session_state.history[-4:]):
        lbl = item["label"]
        bc  = {"hatespeech":"badge-hate","offensive":"badge-offensive","normal":"badge-normal"}[lbl]
        ld  = "Hate Speech" if lbl == "hatespeech" else lbl.capitalize()
        st.markdown(f"""
        <div class="hist-card">
            <span class="hist-text">{item['text'][:70]}{'...' if len(item['text'])>70 else ''}</span>
            <span class="result-badge {bc}" style="margin:0;">{ld}</span>
        </div>""", unsafe_allow_html=True)