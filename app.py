from html import escape
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
SCHEMA_PATH = ROOT / "artifacts" / "input_schema.json"
WINNER_PATH = ROOT / "artifacts" / "winner.json"

st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --navy: #12263F;
  --navy-2: #1B3557;
  --sky: #1F6FEB;
  --sky-soft: #E6F0FF;
  --amber: #F2A93B;
  --teal: #12A594;
  --coral: #E5604D;
  --cloud: #F4F7FB;
  --line: #DCE4EE;
  --muted: #5B6B80;
}

html, body, [class*="css"], .stApp {
  font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--navy);
}
.stApp { background: var(--cloud); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Manrope', sans-serif; letter-spacing: -0.01em; color: var(--navy); }
header[data-testid="stHeader"] { background: transparent; }

/* Hero */
.hero {
  position: relative; overflow: hidden;
  background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 60%, #23497A 100%);
  border-radius: 20px; padding: 2.2rem 2.4rem; color: #fff; margin-bottom: 1.4rem;
}
.hero h1 { color: #fff; font-size: 2.3rem; font-weight: 800; margin: 0 0 .5rem 0; padding: 0; }
.hero p { color: #D5E2F3; font-size: 1.05rem; max-width: 640px; margin: 0 0 1rem 0; line-height: 1.55; }
.hero .chips span {
  display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  color: #fff; padding: .25rem .75rem; border-radius: 999px; font-size: .82rem; margin: 0 .4rem .4rem 0;
}
.hero svg { position: absolute; right: 0; top: 0; height: 100%; width: 46%; opacity: .55; pointer-events: none; }

/* Cards */
.card {
  background: #fff; border: 1px solid var(--line); border-radius: 16px;
  padding: 1.2rem 1.3rem; height: 100%;
}
.card h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.card p { margin: 0; color: var(--muted); font-size: .93rem; line-height: 1.5; }
.step-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 50%; background: var(--sky); color: #fff;
  font-weight: 700; font-family: 'Bricolage Grotesque', sans-serif; margin-bottom: .6rem;
}
.section-note { color: var(--muted); font-size: .92rem; margin: -.25rem 0 .8rem 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 1.1rem; border-radius: 10px 10px 0 0; font-weight: 600; color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--sky); }

/* Inputs */
div[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.stButton > button, .stFormSubmitButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--line);
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(18,38,63,.12);
}
.stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--sky), #3B8BFF); border: none; color: #fff;
  padding: .7rem 1rem; font-size: 1.02rem;
}
:focus-visible { outline: 3px solid var(--amber) !important; outline-offset: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--navy); }
section[data-testid="stSidebar"] * { color: #E8F0FB; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--amber); }
.side-card {
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14);
  border-radius: 14px; padding: .9rem 1rem; margin-bottom: .8rem; font-size: .92rem;
}

/* Boarding pass result */
.pass { display: flex; background: #fff; border: 1px solid var(--line); border-radius: 20px;
  overflow: hidden; box-shadow: 0 10px 30px rgba(18,38,63,.10); margin: .6rem 0 1rem 0; }
.pass .stub { flex: 0 0 280px; padding: 1.6rem 1.4rem; color: #fff; display: flex; flex-direction: column; justify-content: center; }
.pass .stub.ok { background: linear-gradient(160deg, #0E8C7D, #12A594); }
.pass .stub.no { background: linear-gradient(160deg, #D98A1F, var(--amber)); color: #2A1A00; }
.pass .stub .label { font-size: .85rem; opacity: .9; }
.pass .stub .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.4rem; font-weight: 800; line-height: 1.05; }
.pass .perf { width: 0; border-left: 3px dashed var(--line); margin: 14px 0; }
.pass .body { flex: 1; padding: 1.5rem 1.7rem; }
.pass .verdict { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: .2rem; }
.pass .sub { color: var(--muted); font-size: .95rem; margin-bottom: 1rem; }
.meter { position: relative; height: 12px; border-radius: 999px; overflow: hidden;
  background: linear-gradient(90deg, var(--coral) 0%, var(--amber) 50%, var(--teal) 100%); }
.meter .mask { position: absolute; top: 0; right: 0; bottom: 0; background: #EEF2F7; }
.meter .tick { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--navy); left: 50%; }
.meter-legend { display: flex; justify-content: space-between; color: var(--muted); font-size: .78rem; margin-top: .35rem; }
@media (max-width: 760px) {
  .pass { flex-direction: column; } .pass .perf { border-left: none; border-top: 3px dashed var(--line); margin: 0 14px; width: auto; height: 0; }
  .pass .stub { flex-basis: auto; } .hero svg { display: none; } .hero h1 { font-size: 1.8rem; }
}
/* -------------------------------------------------------------------------
   Accessible Streamlit form contrast.
   Use global Streamlit selectors (not section[data-testid="stMain"]),
   because Streamlit versions differ in their main-section DOM wrappers.
   ------------------------------------------------------------------------- */

/* All widget labels: slider, number input, selectbox, text input, etc. */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stSlider"] label *,
.stApp [data-testid="stNumberInput"] label,
.stApp [data-testid="stNumberInput"] label *,
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stSelectbox"] label *,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stTextInput"] label * {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
  opacity: 1 !important;
}

/* Streamlit sometimes places the visible label in Markdown containers. */
.stApp [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stSlider"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
  opacity: 1 !important;
}

/* Captions, section descriptions, and slider endpoint labels. */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stSlider"] small,
.stApp [data-testid="stSlider"] small * {
  color: #53657B !important;
  -webkit-text-fill-color: #53657B !important;
  opacity: 1 !important;
}

/* Help / tooltip icons. */
.stApp [data-testid="stTooltipIcon"],
.stApp [data-testid="stTooltipIcon"] *,
.stApp [data-testid="stTooltipIcon"] svg {
  color: #53657B !important;
  fill: #53657B !important;
  opacity: 1 !important;
}

/* Light input surfaces with readable dark values. */
.stApp [data-baseweb="input"],
.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"] > div {
  background-color: #FFFFFF !important;
  border-color: #C9D5E3 !important;
}

.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] div,
.stApp [data-baseweb="input"] * {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
}

/* Selectbox selected value and dropdown text. */
.stApp [data-baseweb="select"] *,
.stApp [role="listbox"] *,
.stApp [role="option"] {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
}

/* Main content headings. */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
  color: #12263F;
}

/* Tabs: readable inactive and active labels. */
.stTabs [role="tab"],
.stTabs [role="tab"] * {
  color: #53657B !important;
  -webkit-text-fill-color: #53657B !important;
  opacity: 1 !important;
}
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"] * {
  color: #1F6FEB !important;
  -webkit-text-fill-color: #1F6FEB !important;
}

/* Preset buttons: light surface and dark text. */
.stButton > button:not([kind="primary"]) {
  background: #FFFFFF !important;
  color: #19324F !important;
  border: 1px solid #C9D5E3 !important;
}
.stButton > button:not([kind="primary"]) * {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
}
.stButton > button:not([kind="primary"]):hover {
  background: #EAF2FF !important;
  border-color: #1F6FEB !important;
}

/* Preserve the primary prediction button's white-on-blue treatment. */
.stFormSubmitButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] * {
  background: linear-gradient(90deg, #1F6FEB, #3B8BFF) !important;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* White ROC-AUC card in the dark sidebar. */
section[data-testid="stSidebar"] div[data-testid="stMetric"],
section[data-testid="stSidebar"] div[data-testid="stMetric"] *,
section[data-testid="stSidebar"] div[data-testid="stMetric"] label {
  color: #12263F !important;
  -webkit-text-fill-color: #12263F !important;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
  background: #FFFFFF !important;
  border: 1px solid #DCE4EE !important;
  border-radius: 16px !important;
}

/* Restore light text for sidebar content outside the white metric card. */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
  color: #E8F0FB;
}

/* -------------------------------------------------------------------------
   Sliders: always show the lowest / highest labels (Streamlit only shows
   them on hover or focus), keep the value readable, and tighten spacing.
   ------------------------------------------------------------------------- */
.stApp [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
.stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
.stApp [data-testid="stSlider"] [data-testid="stTickBarMax"],
.stApp [data-testid="stSlider"] [data-testid="stSliderTickBar"],
.stApp [data-testid="stSlider"] [data-testid="stTickBar"] {
  visibility: visible !important;
  opacity: 1 !important;
  color: #53657B !important;
  -webkit-text-fill-color: #53657B !important;
  font-size: .8rem !important;
}
.stApp [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
.stApp [data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: #12263F !important;
  -webkit-text-fill-color: #12263F !important;
  font-weight: 700 !important;
}
.stApp [data-testid="stSlider"] [role="slider"]:focus-visible {
  box-shadow: 0 0 0 4px rgba(31,111,235,.30) !important;
  outline: none !important;
}
.stApp [data-testid="stCaptionContainer"] { margin-top: -.35rem; margin-bottom: .35rem; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# Sliders default to Streamlit's red accent. If no theme.primaryColor is set in
# .streamlit/config.toml, recolour the sliders to the app's blue so the theme is
# consistent. (When the config sets the colour, nothing is overridden.)
try:
    _primary_set = st.get_option("theme.primaryColor") is not None
except Exception:
    _primary_set = False
if not _primary_set:
    st.markdown(
        """
<style>
.stApp [data-testid="stSlider"] [data-baseweb="slider"] {
  filter: hue-rotate(222deg) saturate(1.15);
}
</style>
""",
        unsafe_allow_html=True,
    )


def html(markup: str) -> None:
    """Render HTML without Markdown treating indented lines as code blocks."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize()


# ----------------------------------------------------------------------------
# Model assets
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    winner = (
        json.loads(WINNER_PATH.read_text(encoding="utf-8"))
        if WINNER_PATH.exists() else {}
    )
    return model, schema, winner


model, schema, winner = load_assets()

html("""
<div class="hero">
<svg viewBox="0 0 400 220" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<path d="M10 200 C 120 40, 260 20, 390 90" stroke="#F2A93B" stroke-width="2" stroke-dasharray="6 8"/>
<circle cx="10" cy="200" r="6" fill="#F2A93B"/><circle cx="390" cy="90" r="6" fill="#fff"/>
<path d="M300 38 l26 10 -26 10 6 -10z" fill="#fff"/>
</svg>
<h1>✈️ Airline Passenger Satisfaction Predictor</h1>
<p>Describe a passenger and their flight experience, and get an instant estimate of whether
they will leave satisfied, along with what could change the outcome.</p>
<div class="chips"><span>Machine-learning model</span><span>Instant results</span><span>What-if insights</span></div>
</div>
""")

if model is None:
    st.warning(
        "Trained model files are not available. Run `python train.py` first. "
        "It creates artifacts/best_model.joblib and artifacts/input_schema.json."
    )
    st.stop()

features = schema["features"]


def is_rating(f: dict) -> bool:
    """Small whole-number scales (e.g. 0-5 service ratings) are shown as sliders."""
    return (
        f["type"] == "numeric"
        and f.get("integer", False)
        and float(f["min"]) >= 0
        and float(f["max"]) <= 10
    )


rating_feats = [f for f in features if is_rating(f)]
category_feats = [f for f in features if f["type"] != "numeric"]
number_feats = [f for f in features if f["type"] == "numeric" and not is_rating(f)]


def default_for(f: dict):
    if f["type"] == "numeric":
        d = float(f["default"])
        lo, hi = float(f["min"]), float(f["max"])
        d = min(max(d, lo), hi)
        return int(round(d)) if f.get("integer", False) else d
    options = f.get("options", ["Unknown"])
    d = f.get("default", options[0])
    return d if d in options else options[0]


def key_of(f: dict) -> str:
    return f"in_{f['name']}"


for f in features:
    st.session_state.setdefault(key_of(f), default_for(f))
st.session_state.setdefault("history", [])


def apply_preset(kind: str) -> None:
    for f in features:
        if kind == "typical":
            st.session_state[key_of(f)] = default_for(f)
        elif f in rating_feats and kind == "happy":
            st.session_state[key_of(f)] = int(float(f["max"]))
        elif f in rating_feats and kind == "unhappy":
            st.session_state[key_of(f)] = int(float(f["min"]))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("About the model")
    html(f"""
    <div class="side-card"><b>{escape(str(winner.get("model_name", "Best tuned classifier")))}</b><br>
    Chosen as the best of five algorithms after automated tuning.</div>
    """)
    if "cv_roc_auc" in winner:
        st.metric("Cross-validated ROC-AUC", f"{winner['cv_roc_auc']:.4f}")
        st.caption("Closer to 1.00 means the model separates satisfied from "
                   "unsatisfied passengers more reliably.")
    st.divider()
    st.subheader("Quick guide")
    st.markdown(
        "1. Open **Predict**\n"
        "2. Fill in the passenger profile and ratings\n"
        "3. Press **Predict passenger satisfaction**\n"
        "4. Read the result and the what-if tips"
    )
    st.divider()
    st.caption("Estimates are statistical predictions based on past survey data. "
               "Use them to guide decisions, not as a guarantee.")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_predict, tab_guide, tab_about = st.tabs(["Predict", "How to use", "About the tool"])

# ---- How to use --------------------------------------------------------------
with tab_guide:
    st.subheader("What this tool does")
    st.write(
        "This tool predicts whether an airline passenger will be **satisfied** or "
        "**neutral / dissatisfied** with their journey. It learned from thousands of "
        "past passenger surveys, linking trip details and service ratings to the "
        "final satisfaction outcome."
    )
    st.subheader("Use it in four steps")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "Describe the passenger", "Choose travel class, trip type, customer type and other profile details."),
        ("2", "Rate the experience", "Set the service scores with the sliders, from the lowest to the highest rating."),
        ("3", "Run the prediction", "Press the blue button to score the passenger. Results appear instantly."),
        ("4", "Act on the insights", "See which service improvements would raise the chance of satisfaction most."),
    ]
    for col, (num, title, text) in zip((c1, c2, c3, c4), steps):
        with col:
            html(f'<div class="card"><div class="step-badge">{num}</div><h4>{title}</h4><p>{text}</p></div>')

    st.subheader("Reading the result")
    st.markdown(
        "- **Probability of satisfaction** is the model's confidence that the passenger is satisfied. "
        "Above 50% is predicted as *Satisfied*.\n"
        "- **The meter** shows where the score sits between unlikely and very likely, with a marker at the 50% decision line.\n"
        "- **What-if insights** test raising each service rating to its best score and show the estimated gain.\n"
        "- **Quick scenarios** on the Predict tab fill in the ratings for you, so you can see best-case and worst-case outcomes."
    )
    st.info("Tip: change one rating at a time and predict again to see how much each service matters.")

# ---- About -------------------------------------------------------------------
with tab_about:
    st.subheader("Behind the scenes")
    a, b = st.columns(2)
    with a:
        html("""<div class="card"><h4>Who it helps</h4><p>Airline service teams, analysts and
        planners who want to understand which parts of the passenger experience drive satisfaction,
        and test improvements before investing in them.</p></div>""")
    with b:
        html("""<div class="card"><h4>How the model was built</h4><p>Five algorithms were compared:
        Logistic Regression, Decision Tree, Random Forest, XGBoost and CatBoost. Optuna tuned each one
        (20 trials by default), MLflow tracked the experiments, and the model with the highest
        cross-validated ROC-AUC was saved for this app.</p></div>""")
    st.write("")
    st.markdown(
        f"**Inputs used:** {len(features)} features "
        f"({len(rating_feats)} service ratings, {len(category_feats)} profile categories, "
        f"{len(number_feats)} flight measurements)."
    )
    st.caption("Predictions reflect patterns in historical survey data and may not capture every real-world factor.")

# ---- Predict -----------------------------------------------------------------
with tab_predict:
    st.subheader("Passenger and flight details")
    html('<div class="section-note">Start from a quick scenario or fill in the fields yourself.</div>')

    p1, p2, p3, _ = st.columns([1, 1, 1, 2])
    p1.button("Typical passenger", on_click=apply_preset, args=("typical",), use_container_width=True,
              help="Reset every field to the typical (default) value.")
    p2.button("Delighted scenario", on_click=apply_preset, args=("happy",), use_container_width=True,
              help="Set all service ratings to their highest score.")
    p3.button("Frustrated scenario", on_click=apply_preset, args=("unhappy",), use_container_width=True,
              help="Set all service ratings to their lowest score.")

    def render_feature(f: dict) -> None:
        name, key = f["name"], key_of(f)
        label = pretty(name)
        if f["type"] == "numeric":
            lo, hi = float(f["min"]), float(f["max"])
            if f.get("integer", False):
                if is_rating(f):
                    st.slider(label, int(lo), int(hi), key=key,
                              help=f"{int(lo)} = lowest, {int(hi)} = highest")
                else:
                    st.number_input(label, min_value=int(lo), max_value=int(hi), step=1, key=key,
                                    help=f"Allowed range: {int(lo)} to {int(hi)}")
            else:
                st.number_input(label, min_value=lo, max_value=hi, key=key,
                                help=f"Allowed range: {lo:g} to {hi:g}")
        else:
            st.selectbox(label, f.get("options", ["Unknown"]), key=key)

    if rating_feats:
        _lo = int(min(float(f["min"]) for f in rating_feats))
        _hi = int(max(float(f["max"]) for f in rating_feats))
        rating_note = f"Rate each service from {_lo} (lowest) to {_hi} (highest)."
    else:
        rating_note = ""

    with st.form("passenger_form"):
        for title, note, group, ncols in (
            ("Passenger profile", "Who is travelling and how.", category_feats, 3),
            ("Flight details", "Distance, timing and other measurements.", number_feats, 3),
            ("Service ratings", rating_note, rating_feats, 2),
        ):
            if not group:
                continue
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(note)
                cols = st.columns(ncols)
                for i, f in enumerate(group):
                    with cols[i % ncols]:
                        render_feature(f)

        submitted = st.form_submit_button(
            "Predict passenger satisfaction", type="primary", use_container_width=True
        )

    if submitted:
        values = {f["name"]: st.session_state[key_of(f)] for f in features}
        input_df = pd.DataFrame([values])
        try:
            probability = float(model.predict_proba(input_df)[0, 1])
        except Exception as error:
            st.error(f"Prediction failed: {error}. Check that the inputs match the trained model's schema.")
            st.stop()

        satisfied = probability >= 0.5
        cls = "ok" if satisfied else "no"
        verdict = "Satisfied" if satisfied else "Neutral or dissatisfied"
        confidence = max(probability, 1 - probability)
        if confidence >= 0.85:
            strength = "High confidence"
        elif confidence >= 0.65:
            strength = "Moderate confidence"
        else:
            strength = "Close call"
        detail = (
            "This passenger is likely to be happy with the journey."
            if satisfied else
            "This passenger is unlikely to be happy. See the suggestions below."
        )

        html(f"""
        <div class="pass">
        <div class="stub {cls}">
        <div class="label">Probability of satisfaction</div>
        <div class="big">{probability:.1%}</div>
        <div class="label">{strength}</div>
        </div>
        <div class="perf"></div>
        <div class="body">
        <div class="verdict">{verdict}</div>
        <div class="sub">{detail}</div>
        <div class="meter"><div class="mask" style="left:{probability*100:.1f}%"></div><div class="tick"></div></div>
        <div class="meter-legend"><span>Unlikely</span><span>50% decision line</span><span>Very likely</span></div>
        </div>
        </div>
        """)

        # What-if insights: raise each rating to its best score, one at a time.
        gains = []
        candidates, meta = [], []
        for f in rating_feats:
            best = int(float(f["max"]))
            if values[f["name"]] < best:
                row = dict(values)
                row[f["name"]] = best
                candidates.append(row)
                meta.append((pretty(f["name"]), values[f["name"]], best))
        if candidates:
            try:
                new_probs = model.predict_proba(pd.DataFrame(candidates))[:, 1]
                gains = sorted(
                    ((n, cur, best, float(p) - probability) for (n, cur, best), p in zip(meta, new_probs)),
                    key=lambda x: x[3], reverse=True,
                )
            except Exception:
                gains = []

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**What-if insights**")
            top = [g for g in gains if g[3] > 0.001][:3]
            if top:
                st.caption("Estimated change if a single rating were raised to its best score.")
                for n, cur, best, delta in top:
                    with st.container(border=True):
                        c_a, c_b = st.columns([3, 1])
                        c_a.markdown(f"**{n}**  \nfrom {cur} to {best}")
                        c_b.metric("Gain", f"+{delta*100:.1f} pts", label_visibility="collapsed")
            elif candidates:
                st.info("Improving any single service rating would not noticeably change this result.")
            else:
                st.info("All service ratings are already at their best score.")
        with right:
            st.markdown("**Save this result**")
            st.caption("Download the inputs and prediction as a CSV file.")
            out = input_df.copy()
            out["satisfaction_probability"] = round(probability, 4)
            out["prediction"] = verdict
            st.download_button("Download result (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="satisfaction_prediction.csv", mime="text/csv",
                               use_container_width=True)

        st.session_state["history"].append({
            "Run": len(st.session_state["history"]) + 1,
            "Probability": f"{probability:.1%}",
            "Prediction": verdict,
        })

    if st.session_state["history"]:
        with st.expander(f"Session history ({len(st.session_state['history'])} predictions)"):
            st.dataframe(pd.DataFrame(st.session_state["history"]).iloc[::-1],
                         use_container_width=True, hide_index=True)
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()
