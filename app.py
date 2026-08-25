import json
import os
from datetime import date, datetime, timedelta

import base64
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
LOG_PATH = os.path.join("logs", "sleep_log.json")

st.set_page_config(
    page_title="Slumbr | Sleep better. Wake smarter.",
    page_icon="assets/slumbr_logo.webp",
    layout="wide",
)

BG = "#E9E8DC"
INK = "#26262A"
MUTED = "#5F5D57"
PRIMARY = "#B56D2A"
LINE = "#B9B7AA"
EXCELLENT = "#56745D"
AVERAGE = "#B56D2A"
POOR = "#A64B42"

CATEGORY_STYLE = {
    "Good": {"label": "Good", "color": EXCELLENT, "points": 100},
    "Average": {"label": "Average", "color": AVERAGE, "points": 65},
    "Poor": {"label": "Poor", "color": POOR, "points": 30},
}

# Keep the CSS deliberately small and structural. No fake card wrappers around Streamlit widgets.
st.markdown(
    f"""
<style>
:root {{ --bg:{BG}; --ink:{INK}; --muted:{MUTED}; --accent:{PRIMARY}; --line:{LINE}; }}
html, body, [class*="css"] {{ font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); }}
#MainMenu, footer, header {{ visibility:hidden; }}
.stApp {{ background-color:var(--bg); background-image:linear-gradient(rgba(70,70,65,.11) 1px,transparent 1px),linear-gradient(90deg,rgba(70,70,65,.11) 1px,transparent 1px); background-size:26px 26px; }}
[data-testid="stAppViewContainer"] {{ background:transparent; }}
.block-container {{ width:min(100%, 1160px); max-width:1160px; padding-top:1.6rem; padding-bottom:2rem; }}
h1,h2,h3,h4,h5 {{ color:var(--ink)!important; letter-spacing:-.02em; }}
.ss-header {{ margin-bottom:1.25rem; }}
.ss-eyebrow {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); margin-bottom:.35rem; }}
.ss-brand {{ display:flex; align-items:center; gap:.65rem; }}
.ss-logo {{ width:50px; height:50px; border:1px solid #2A2A2D; border-radius:10px; object-fit:contain; display:block; }}
.ss-title {{ font-family:Georgia,"Times New Roman",serif; font-size:2.15rem; font-weight:700; line-height:1; margin:0; letter-spacing:-.045em; }}
.ss-subtitle {{ color:var(--muted); font-size:.84rem; margin-top:.25rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:rgba(239,238,226,.52)!important; border:1px solid rgba(105,103,94,.34)!important; border-radius:6px!important; box-shadow:none!important; margin:.45rem 0!important; padding:0!important; }}
.ss-section-title {{ font-family:Georgia,"Times New Roman",serif; font-size:1.12rem; font-weight:700; margin:0 0 .65rem; }}
.ss-muted {{ color:var(--muted); font-size:.78rem; line-height:1.4; }}
.ss-note {{ background:transparent; border-left:2px solid var(--accent); padding:.55rem .75rem; font-size:.76rem; color:var(--muted); margin-top:.55rem; }}
.ss-result-label {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.61rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.ss-big-result {{ font-family:Georgia,"Times New Roman",serif; font-size:1.8rem; font-weight:700; line-height:1; margin-top:.15rem; }}
.ss-confidence {{ font-family:Georgia,"Times New Roman",serif; font-size:1.65rem; line-height:1; margin-top:.15rem; }}
.ss-meaning {{ font-size:.82rem; line-height:1.45; max-width:32rem; }}
.ss-disclaimer {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.6rem; line-height:1.45; color:var(--muted); margin-top:.65rem; }}
.ss-week {{ display:flex; gap:.65rem; align-items:flex-end; margin:.8rem 0 .25rem; }}
.ss-day {{ flex:1; display:flex; flex-direction:column; align-items:center; gap:.2rem; }}
.ss-day-points,.ss-day-label,.ss-day-date {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.ss-day-points {{ font-size:.62rem; color:var(--muted); min-height:.9rem; }}
.ss-bar-track {{ width:100%; max-width:46px; height:92px; background:rgba(0,0,0,.018); display:flex; align-items:flex-end; overflow:hidden; border:1px solid rgba(105,103,94,.28); }}
.ss-bar-fill {{ width:100%; }}
.ss-day-label {{ font-size:.61rem; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); }}
.ss-day-date {{ font-size:.58rem; color:var(--muted); opacity:.8; }}
.ss-day-today .ss-day-label {{ color:var(--accent); font-weight:600; }}
/* Tabs: no top rule, and the active rule is only as wide as the tab text. */
[data-baseweb="tab-list"] {{ gap:0!important; border:0!important; align-items:flex-end; }}
button[data-baseweb="tab"] {{ position:relative!important; width:auto!important; min-width:0!important; flex:0 0 auto!important; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.71rem; color:var(--muted); background:transparent!important; padding:.25rem 0 .42rem!important; margin:0 1.55rem 0 0!important; border:0!important; box-shadow:none!important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--accent); }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display:none!important; background:transparent!important; }}
button[data-baseweb="tab"][aria-selected="true"]::after {{ content:""; position:absolute; left:0; width:100%; right:auto; bottom:0; height:2px; background:var(--accent); }}
/* Compact Streamlit controls. */
div[data-baseweb="input"], div[data-baseweb="select"]>div {{ background-color:rgba(239,238,226,.76)!important; border-color:rgba(105,103,94,.38)!important; border-radius:4px!important; min-height:2.2rem; }}
label {{ color:var(--ink)!important; font-size:.72rem!important; }}
/* "Press Enter to apply" hint: align it under the field instead of Streamlit's default floating position. */
[data-testid="InputInstructions"] {{ position:static!important; display:block; width:100%; text-align:right; font-size:.6rem!important; line-height:1.3; color:var(--muted)!important; opacity:.85; margin-top:.15rem; white-space:normal; }}
div[data-testid="stNumberInput"], div[data-testid="stTextInput"], div[data-testid="stDateInput"] {{ display:flex; flex-direction:column; }}
.stButton>button, .stDownloadButton>button {{ background:var(--ink); color:#F2F0E5; border:1px solid var(--ink); border-radius:3px; padding:.48rem .9rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.66rem; box-shadow:none; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.stButton>button[kind="primary"] {{ background:var(--accent)!important; border-color:var(--accent)!important; color:#fff!important; }}
[data-testid="stExpander"] {{ border:1px solid rgba(105,103,94,.28)!important; border-radius:4px!important; background:transparent!important; }}
[data-testid="stMetric"] {{ border:none!important; background:transparent!important; padding:.1rem 0!important; }}
[data-testid="stMetricLabel"] {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.63rem; color:var(--muted); }}
[data-testid="stMetricValue"] {{ font-family:Georgia,"Times New Roman",serif; color:var(--ink); font-size:1.55rem; }}
.ss-table {{ width:100%; border-collapse:collapse; font-size:.72rem; }}
.ss-table th {{ text-align:left; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.58rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); padding:.5rem .45rem; border-bottom:1px solid var(--line); }}
.ss-table td {{ padding:.5rem .45rem; border-bottom:1px solid rgba(105,103,94,.2); }}
.ss-empty {{ border:1px dashed rgba(105,103,94,.45); padding:.7rem .8rem; color:var(--muted); font-size:.78rem; }}
@media (max-width:850px) {{ .block-container {{ padding:1rem .8rem 2rem; }} .ss-title {{ font-size:1.8rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    metadata_path = os.path.join(MODELS_DIR, "metadata.pkl")
    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path)):
        return None, None, None
    return joblib.load(model_path), joblib.load(preprocessor_path), joblib.load(metadata_path)

model, preprocessor, metadata = load_artifacts()
if model is None:
    st.error("Trained model files were not found. Please run `python train.py` first.")
    st.stop()
FEATURE_COLUMNS = metadata["feature_columns"]

def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def save_entry(entry):
    log = [e for e in load_log() if e["date"] != entry["date"]]
    log.append(entry)
    log.sort(key=lambda e: e["date"])
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def clear_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

def week_dates(reference):
    monday = reference - timedelta(days=reference.weekday())
    return [monday + timedelta(days=i) for i in range(7)]

SCHEDULE_POOR = [
    "Keep a fixed bedtime and wake time, including weekends.",
    "Stop screens 45-60 minutes before bed.",
    "Use a short wind-down routine with dim light or light stretching.",
]
DIET_POOR = [
    "Stop caffeine after early afternoon.",
    "Avoid heavy meals within 3 hours of bedtime.",
    "Stay hydrated through the day and taper fluids near bedtime.",
]
SCHEDULE_AVERAGE = [
    "Keep your wake-up time consistent.",
    "Add a 15-20 minute screen-free wind-down.",
]
DIET_AVERAGE = [
    "Have your last caffeinated drink by mid-afternoon.",
    "Keep dinner balanced and avoid a large meal right before bed.",
]

def render_recommendations(category):
    if category == "Poor":
        schedule, diet = SCHEDULE_POOR, DIET_POOR
    elif category == "Average":
        schedule, diet = SCHEDULE_AVERAGE, DIET_AVERAGE
    else:
        return
    st.markdown('<div class="ss-section-title">Suggested schedule and diet</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Schedule**")
        for item in schedule:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Diet**")
        for item in diet:
            st.markdown(f"- {item}")

def predict_and_save(values):
    input_row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    processed = preprocessor.transform(input_row)
    probabilities = model.predict_proba(processed)[0]
    probability_map = dict(zip(model.classes_, probabilities))
    prediction = model.predict(processed)[0]

    sleep_duration = float(values["Sleep Duration"])
    stress_level = int(values["Stress Level"])
    if sleep_duration <= 4.5 or (sleep_duration <= 5.5 and stress_level >= 8):
        prediction = "Poor"
    elif prediction == "Good" and (sleep_duration < 6.0 or stress_level >= 9):
        prediction = "Average"

    confidence = float(probability_map.get(prediction, max(probabilities)))
    if prediction == "Poor" and confidence < 0.5:
        confidence = max(confidence, 0.75)

    style = CATEGORY_STYLE[prediction]
    entry = {
        "date": values["Date"].isoformat(),
        "gender": values["Gender"], "age": values["Age"], "occupation": values["Occupation"],
        "sleep_duration": sleep_duration, "physical_activity": values["Physical Activity Level"],
        "stress_level": stress_level, "bmi_category": values["BMI Category"],
        "systolic_bp": values["Systolic Blood Pressure"], "diastolic_bp": values["Diastolic Blood Pressure"],
        "heart_rate": values["Heart Rate"], "daily_steps": values["Daily Steps"],
        "category": prediction, "points": style["points"], "confidence": round(confidence, 4),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_entry(entry)
    return prediction, confidence, style

def logo_data_uri():
    with open(os.path.join("assets", "slumbr_logo.webp"), "rb") as f:
        return "data:image/webp;base64," + base64.b64encode(f.read()).decode("ascii")

st.markdown(
    f'''<div class="ss-header"><div class="ss-eyebrow">DAILY SLEEP CHART &amp; WEEKLY LOG</div><div class="ss-brand"><img class="ss-logo" src="{logo_data_uri()}" alt="Slumbr logo"><div><div class="ss-title">slumbr</div><div class="ss-subtitle">Sleep better. Wake smarter.</div></div></div></div>''',
    unsafe_allow_html=True,
)

tab_checkin, tab_week, tab_history = st.tabs(["Today", "Your Week", "History"])

with tab_checkin:
    # All input controls stay in one compact card. The side panel is intentionally narrow but usable.
    with st.container(border=True):
        st.markdown('<div class="ss-section-title">Tonight\'s sleep</div>', unsafe_allow_html=True)
        r1 = st.columns(4, gap="medium")
        with r1[0]: sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 14.0, 7.0, 0.5, format="%.1f")
        with r1[1]: stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        with r1[2]: physical_activity = st.number_input("Activity (min/day)", 0, 300, 45)
        with r1[3]: bmi_category = st.selectbox("BMI Category", metadata["bmi_options"])
        r2 = st.columns(4, gap="medium")
        with r2[0]: age = st.number_input("Age", 10, 100, 30)
        with r2[1]: gender = st.selectbox("Gender", metadata["gender_options"])
        with r2[2]: entry_date = st.date_input("Date", value=date.today(), max_value=date.today())
        with r2[3]: occupation = st.selectbox("Occupation", metadata["occupation_options"])
        with st.expander("Additional health details"):
            e1, e2, e3, e4 = st.columns(4, gap="medium")
            with e1: systolic_bp = st.number_input("Systolic BP", 70, 220, 120)
            with e2: diastolic_bp = st.number_input("Diastolic BP", 40, 140, 80)
            with e3: heart_rate = st.number_input("Heart Rate (bpm)", 30, 220, 72)
            with e4: daily_steps = st.number_input("Daily Steps", 0, 40000, 6000, 100)

    left_action, right_action = st.columns([1, 4])
    with left_action:
        predict_clicked = st.button("Predict Sleep Quality", type="primary", use_container_width=True)
    with right_action:
        reset_clicked = st.button("Reset")
    if reset_clicked:
        st.rerun()

    if predict_clicked:
        values = {
            "Gender": gender, "Age": age, "Occupation": occupation, "Sleep Duration": sleep_duration,
            "Physical Activity Level": physical_activity, "Stress Level": stress_level,
            "BMI Category": bmi_category, "Systolic Blood Pressure": systolic_bp,
            "Diastolic Blood Pressure": diastolic_bp, "Heart Rate": heart_rate,
            "Daily Steps": daily_steps, "Date": entry_date,
        }
        prediction, confidence, style = predict_and_save(values)
        st.session_state["last_result"] = (prediction, confidence, style)

    if "last_result" in st.session_state:
        prediction, confidence, style = st.session_state["last_result"]
        with st.container(border=True):
            st.markdown('<div class="ss-section-title">Prediction result</div>', unsafe_allow_html=True)
            a, b, c = st.columns([1.0, .8, 2.3], gap="large")
            with a:
                st.markdown('<div class="ss-result-label">Predicted Sleep Quality</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ss-big-result" style="color:{style["color"]};">{style["label"]}</div>', unsafe_allow_html=True)
            with b:
                st.markdown('<div class="ss-result-label">Points</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ss-confidence">{style["points"]}</div>', unsafe_allow_html=True)
            with c:
                st.markdown('<div class="ss-result-label">What this means</div>', unsafe_allow_html=True)
                meaning = {
                    "Poor": "Your current inputs point to a sleep pattern that may need attention.",
                    "Average": "You are in the middle range. A steadier schedule and lower evening stress may help.",
                    "Good": "You are likely getting a good balance of rest and recovery. Keep the habits that work.",
                }[prediction]
                st.markdown(f'<div class="ss-meaning">{meaning}</div>', unsafe_allow_html=True)

        if prediction in ("Poor", "Average"):
            with st.expander("Suggestions for tonight", expanded=False):
                render_recommendations(prediction)

    st.markdown('<div class="ss-disclaimer">Disclaimer: This prediction is based on the data you provide and is not a substitute for professional medical advice.</div>', unsafe_allow_html=True)

with tab_week:
    log = load_log()
    by_date = {e["date"]: e for e in log}
    today = date.today()
    days = week_dates(today)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    points = [by_date[d.isoformat()]["points"] for d in days if d.isoformat() in by_date]
    total = sum(points)
    count = len(points)
    avg = total / count if count else 0
    with st.container(border=True):
        st.markdown('<div class="ss-section-title">This week\'s sleep</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Days logged", f"{count} / 7")
        m2.metric("Total points", total)
        m3.metric("Average points/day", f"{avg:.0f}" if count else "-")
        html = '<div class="ss-week">'
        for d, label in zip(days, labels):
            e = by_date.get(d.isoformat())
            pts = e["points"] if e else 0
            color = CATEGORY_STYLE[e["category"]]["color"] if e else "transparent"
            fill = f'<div class="ss-bar-fill" style="height:{max(pts,6)}%;background:{color};"></div>' if e else ""
            today_cls = " ss-day-today" if d == today else ""
            html += f'<div class="ss-day{today_cls}"><div class="ss-day-points">{pts if e else ""}</div><div class="ss-bar-track">{fill}</div><div class="ss-day-label">{label}</div><div class="ss-day-date">{d.day}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    if not log:
        st.markdown('<div class="ss-empty">No check-ins yet. Log today\'s details to start your weekly picture.</div>', unsafe_allow_html=True)

with tab_history:
    log = load_log()
    with st.container(border=True):
        st.markdown('<div class="ss-section-title">Sleep history</div>', unsafe_allow_html=True)
        if log:
            hist = pd.DataFrame(log).sort_values("date", ascending=False)
            hist["Sleep Quality"] = hist["category"].map(lambda c: CATEGORY_STYLE[c]["label"])
            hist["Sleep Duration"] = hist["sleep_duration"].map(lambda x: f"{float(x):.1f} h")
            hist["Activity"] = hist["physical_activity"].map(lambda x: f"{int(x)} min")
            view = hist[["date", "Sleep Quality", "points", "Sleep Duration", "stress_level", "Activity", "bmi_category"]].rename(columns={"date":"Date","points":"Points","stress_level":"Stress","bmi_category":"BMI"})
            st.dataframe(view, use_container_width=True, hide_index=True)
            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(log, indent=2).encode("utf-8")
            b1, b2, b3 = st.columns([1.25, 1.25, 3.5])
            with b1: st.download_button("Download sleep log", csv_bytes, "slumbr_sleep_log.csv", "text/csv", use_container_width=True)
            with b2: st.download_button("Download JSON", json_bytes, "slumbr_sleep_log.json", "application/json", use_container_width=True)
            with b3:
                if st.button("Clear all logged data"):
                    clear_log(); st.rerun()
        else:
            st.markdown('<div class="ss-empty">Nothing logged yet. Your check-ins will build up here day by day.</div>', unsafe_allow_html=True)
