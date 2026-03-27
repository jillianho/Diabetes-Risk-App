import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
from diabetes_proxies import PatientInputs, build_feature_vector, generate_results_content

if "page" not in st.session_state:
    st.session_state["page"] = "input"

if st.session_state["page"] == "results":
    st.markdown("""
    <style>
    .results-wrap { padding-top: 0.25rem; }
    .risk-hero { text-align: center; padding: 1rem 0 0.5rem; }
    .dial-wrap { display: flex; flex-direction: column; align-items: center; margin-bottom: 1rem; }
    .dial {
        --pct: 67;
        --dial-color: #E24B4A;
        width: 220px;
        height: 120px;
        border-top-left-radius: 220px;
        border-top-right-radius: 220px;
        overflow: hidden;
        position: relative;
        background:
            conic-gradient(from 180deg, var(--dial-color) calc(var(--pct) * 1.8deg), #f1efe8 0deg);
    }
    .dial::before {
        content: "";
        position: absolute;
        left: 22px;
        right: 22px;
        top: 22px;
        bottom: -98px;
        background: white;
        border-top-left-radius: 180px;
        border-top-right-radius: 180px;
    }
    .dial::after {
        content: "";
        position: absolute;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #141414;
        left: calc(50% + (var(--pct) - 50) * 1.35px - 8px);
        top: 12px;
        box-shadow: 0 0 0 4px rgba(255,255,255,.96);
    }
    .risk-num { font-size: 56px; font-weight: 600; color: #1f2d46; line-height: 1; }
    .risk-sub { font-size: 13px; color: #667085; margin-top: 6px; }
    .risk-pill { display: inline-block; margin-top: 10px; padding: 4px 14px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .pill-low { background: #e6f9f0; color: #2b8a50; }
    .pill-mod { background: #fff4e5; color: #b86908; }
    .pill-high { background: #ffe7e7; color: #d72e30; }
    .pill-calm { background: #e6f7ff; color: #0d7ea2; }
    .pill-action { background: #fff4e5; color: #b86908; }
    .pill-urgent { background: #ffe7e7; color: #d72e30; }
    .metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #667085; margin-bottom: 6px; }
    .metric-value { font-size: 30px; font-weight: 700; color: #1f2d46; }
    .metric-note { font-size: 12px; color: #667085; margin-top: 4px; }
    .framing-box { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #fff; margin: 0.5rem 0 1rem; }
    .framing-summary { font-size: 14px; color: #1f2d46; line-height: 1.7; margin-bottom: 12px; }
    .framing-pills { display: flex; gap: 8px; flex-wrap: wrap; }
    .bar-row { margin: 12px 0; }
    .bar-head { display: flex; justify-content: space-between; font-size: 13px; color: #1f2d46; margin-bottom: 4px; }
    .bar-track { width: 100%; height: 8px; background: #eef2f7; border-radius: 999px; overflow: hidden; }
    .bar-risk { height: 100%; background: #E24B4A; }
    .bar-protect { height: 100%; background: #639922; }
    .action-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; background: #fff; }
    .action-badge { display: inline-block; margin-bottom: 6px; padding: 3px 8px; border-radius: 6px; font-size: 10px; font-weight: 700; }
    .badge-impact { background: #ffe7e7; color: #d72e30; }
    .badge-fast { background: #e6f9f0; color: #2b8a50; }
    .badge-doctor { background: #fff4e5; color: #b86908; }
    .badge-mod { background: #e6f7ff; color: #0d7ea2; }
    .action-title { font-size: 14px; font-weight: 600; color: #1f2d46; margin-bottom: 4px; }
    .action-desc { font-size: 13px; color: #667085; }
    .action-delta { font-size: 12px; color: #2b8a50; font-weight: 600; margin-top: 6px; }
    .prov-item { display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 13px; color: #1f2d46; }
    .prov-tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }
    .prov-user { background: #e6f9f0; color: #2b8a50; }
    .prov-est { background: #e6f7ff; color: #0d7ea2; }
    </style>
    """, unsafe_allow_html=True)

    if st.button("← Back to calculator"):
        st.session_state["page"] = "input"
        st.rerun()

    if "results_data" in st.session_state:
        data = st.session_state["results_data"]
        risk_pct = data["risk_pct"]
        risk_label_ui = data["risk_label"]
        risk_badge_class = data["risk_badge_class"]
        risk_bar_class = "pill-low" if risk_badge_class == "risk-low" else "pill-mod" if risk_badge_class == "risk-mod" else "pill-high"
        dial_color = "#639922" if risk_badge_class == "risk-low" else "#EF9F27" if risk_badge_class == "risk-mod" else "#E24B4A"

        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
        tab_results, tab_breakdown, tab_whatif, tab_next = st.tabs(["Results", "Risk breakdown", "What if?", "Next steps"])

        with tab_results:
            st.markdown(
                f'<div class="risk-hero"><div class="dial-wrap"><div class="dial" style="--pct:{risk_pct}; --dial-color:{dial_color};"></div></div><div class="risk-num">{risk_pct}%</div><div class="risk-sub">estimated diabetes risk probability</div><span class="risk-pill {risk_bar_class}">{risk_label_ui}</span></div>',
                unsafe_allow_html=True,
            )
            pills_html = "".join(
                f'<span class="risk-pill {pill["class"]}">{pill["text"]}</span>' for pill in data.get("urgency_pills", [])
            )
            st.markdown(
                f'<div class="framing-box"><div class="framing-summary">{data["framing"]}</div><div class="framing-pills">{pills_html}</div></div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Fasting glucose</div><div class="metric-value">{data["glucose_display"]} <span style="font-size:16px;font-weight:500">mg/dL</span></div><div class="metric-note">{"estimated" if data["glucose_estimated"] else "user supplied"}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Insulin resistance</div><div class="metric-value">{data["ir_tier"]}</div><div class="metric-note">HOMA-IR ~{data["homa_range"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">FINDRISC score</div><div class="metric-value">{data["findrisc"]} <span style="font-size:16px;font-weight:500">/ 26</span></div><div class="metric-note">{data["findrisc_label"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f'Estimate confidence: {data["confidence_label"]}')
            st.progress(data["confidence_pct"] / 100)
            st.markdown("#### Ranked action plan")
            for item in data.get("actions", []):
                st.markdown(
                    f'<div class="action-card"><span class="action-badge {item["badge_class"]}">{item["badge"]}</span><div class="action-title">{item["title"]}</div><div class="action-desc">{item["desc"]}</div><div class="action-delta">{item["delta"]}</div></div>',
                    unsafe_allow_html=True,
                )

        with tab_breakdown:
            st.markdown(data["breakdown_headline"])
            st.markdown("##### Risk drivers")
            for factor in data.get("risk_factor_bars", []):
                st.markdown(
                    f'<div class="bar-row"><div class="bar-head"><span>{factor["name"]}</span><span>{factor["pct"]}%</span></div><div class="bar-track"><div class="bar-risk" style="width:{factor["pct"]}%"></div></div><div style="font-size:12px;color:#667085">{factor["note"]}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("##### Protective factors")
            for factor in data.get("protect_factor_bars", []):
                st.markdown(
                    f'<div class="bar-row"><div class="bar-head"><span>{factor["name"]}</span><span>{factor["pct"]}%</span></div><div class="bar-track"><div class="bar-protect" style="width:{factor["pct"]}%"></div></div><div style="font-size:12px;color:#667085">{factor["note"]}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("##### Value provenance")
            for p in data.get("provenance", []):
                source = p.get("src", "estimated") if isinstance(p, dict) else "estimated"
                label = p.get("label", str(p)) if isinstance(p, dict) else str(p)
                tag_class = "prov-user" if source == "user" else "prov-est"
                tag_text = "user" if source == "user" else "estimated"
                st.markdown(
                    f'<div class="prov-item"><span class="prov-tag {tag_class}">{tag_text}</span><span>{label}</span></div>',
                    unsafe_allow_html=True,
                )

        with tab_whatif:
            baseline = data.get("baseline", {})
            sim_waist = st.slider("Waist circumference", 65, 130, int(round(baseline.get("waist", data.get("waist", 96)))), key="sim_waist")
            sim_bmi = st.slider("BMI", 16.0, 45.0, float(baseline.get("bmi", data.get("bmi", 27.5))), 0.1, key="sim_bmi")
            sim_bp = st.slider("Blood pressure (diastolic)", 50, 120, int(round(baseline.get("bp", data.get("bp", 88)))), key="sim_bp")
            sim_gluc = st.slider("Glucose", 70, 250, int(round(baseline.get("gluc", data.get("glucose_raw", 200)))), key="sim_gluc")

            delta = 0
            delta += (baseline.get("waist", sim_waist) - sim_waist) * 0.25
            delta += (baseline.get("bmi", sim_bmi) - sim_bmi) * 1.0
            delta += (baseline.get("bp", sim_bp) - sim_bp) * 0.2
            delta += (baseline.get("gluc", sim_gluc) - sim_gluc) * 0.15
            sim_risk = int(max(1, min(99, round(risk_pct - delta))))
            st.metric("Simulated risk", f"{sim_risk}%", f"{sim_risk - risk_pct:+d}% vs baseline")

        with tab_next:
            st.markdown("#### This week")
            for item in data.get("this_week", []):
                st.markdown(f"- {item}")
            st.markdown("#### What to ask your doctor")
            for item in data.get("ask_doctor", []):
                st.markdown(f"- {item}")
            st.markdown("#### Lab tests")
            for item in data.get("lab_tests", []):
                st.markdown(f"- {item}")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state["page"] = "input"
        st.rerun()

    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
:root, body, .stApp, .root {
    font-family: inherit !important;
}

h1, h2, h3, h4, h5, h6, .main-header, .section-header {
    font-family: inherit !important;
}

.stTextInput input, .stNumberInput input, .stSelectbox select, .stRadio label, .stCheckbox label, .stButton button {
    font-family: 'Source Sans Pro', 'Source Sans 3', sans-serif !important;

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .risk-low {
        color: #27ae60;
        font-weight: 600;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: 600;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Interpretable Diabetes Risk & Intervention Simulator",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Header with better styling
st.markdown('<h1 class="main-header">🩺 Interpretable Diabetes Risk & Intervention Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get personalized insights about your diabetes risk based on your health profile</p>', unsafe_allow_html=True)

with st.expander("ℹ️ What do these measurements mean?"):
    st.markdown("""
    **Glucose**: Blood sugar level from a fasting blood test (normal: 70-99 mg/dL)

    **Blood Pressure**: Diastolic pressure (normal: <80 mmHg) - this is the bottom number in a blood pressure reading like 120/80

    **Insulin**: Fasting insulin level (normal: 2-25 μU/mL)

    **BMI**: Body Mass Index = weight(kg) / height(m)²

    **Waist Circumference**: Measured around the narrowest part of your waist

    *If you don't have recent medical test results, the app will estimate values based on your health profile.*
    """)

st.markdown('<h2 class="section-header">Health Information</h2>', unsafe_allow_html=True)

def synced_slider_number(label, min_value, max_value, value, step=None, format=None, key=None):
    if key is None:
        key = label.replace(" ", "_").lower()

    state_key = f"{key}_value"
    if state_key not in st.session_state:
        st.session_state[state_key] = value

    def _slider_changed():
        st.session_state[state_key] = st.session_state[f"{key}_slider"]
        st.session_state[f"{key}_number"] = st.session_state[state_key]

    def _number_changed():
        st.session_state[state_key] = st.session_state[f"{key}_number"]
        st.session_state[f"{key}_slider"] = st.session_state[state_key]

    col1, col2 = st.columns([3, 1])

    with col1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[state_key],
            step=step,
            format=format,
            key=f"{key}_slider",
            on_change=_slider_changed,
        )
    with col2:
        st.number_input(
            "",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[state_key],
            step=step,
            format=format,
            key=f"{key}_number",
            on_change=_number_changed,
            label_visibility="collapsed"
        )

    return st.session_state[state_key]

# Basic Information
st.markdown("### 📊 Basic Information")
pregnancies = synced_slider_number("Pregnancies", 0, 20, 0, step=1, key="pregnancies")
bmi = synced_slider_number("BMI (kg/m²)", 0.0, 50.0, 0.0, step=0.1, format="%.1f", key="bmi")
age = synced_slider_number("Age (years)", 0, 100, 0, step=1, key="age")
blood_pressure = synced_slider_number("Blood Pressure - Diastolic (mmHg)", 0, 120, 0, step=1, key="blood_pressure")

# Additional Health Factors
st.markdown("### 🏃 Additional Health Factors")
waist_circumference = synced_slider_number("Waist Circumference (cm)", 0, 150, 0, step=1, key="waist")

col1, col2 = st.columns(2)
with col1:
    physical_activity = st.selectbox(
        "Physical Activity Level",
        ["Sedentary", "Moderate", "Active"],
        index=1,
        help="Sedentary: little exercise, Moderate: some exercise, Active: regular exercise"
    )

    diet_quality = st.selectbox(
        "Diet Quality",
        ["Poor", "Average", "Good"],
        index=1,
        help="Poor: high processed foods, Average: mixed diet, Good: mostly whole foods"
    )

with col2:
    family_history = st.selectbox(
        "Family History of Diabetes",
        ["None", "One parent or sibling", "Both parents or early onset"],
        index=0,
        help="Both parents or early onset (<40 years) increases risk significantly"
    )

    prediabetes_diagnosed = st.checkbox("I have been told by a doctor that I have prediabetes")

# Optional Medical Test Results
st.markdown("### 🏥 Optional Medical Test Results")
col1, col2 = st.columns(2)

with col1:
    know_glucose = st.checkbox("I know my glucose value")
    glucose = None
    if know_glucose:
        glucose = synced_slider_number("Glucose (mg/dL)", 0, 300, 0, step=1, key="glucose")

with col2:
    know_insulin = st.checkbox("I know my insulin value")
    insulin = None
    if know_insulin:
        insulin = synced_slider_number("Insulin (µU/mL)", 0, 100, 0, step=1, key="insulin")

# Helper functions used in prediction output. Placed before serialize-time call path.
def compute_findrisc(patient: PatientInputs) -> int:
    score = 0
    # age
    if patient.age <= 44:
        score += 0
    elif patient.age <= 54:
        score += 2
    elif patient.age <= 64:
        score += 3
    else:
        score += 4
    # bmi
    if patient.bmi < 25:
        score += 0
    elif patient.bmi < 30:
        score += 1
    else:
        score += 3
    # waist
    if patient.waist_circumference < 94:
        score += 0
    elif patient.waist_circumference <= 102:
        score += 3
    else:
        score += 4
    # activity
    if patient.physical_activity == "Active":
        score += 0
    elif patient.physical_activity == "Moderate":
        score += 2
    else:
        score += 3
    # diet
    if patient.diet_quality == "Good":
        score += 0
    elif patient.diet_quality == "Average":
        score += 1
    else:
        score += 3
    # blood pressure
    if patient.blood_pressure >= 85:
        score += 2
    # family history
    if patient.family_history == "one parent or sibling":
        score += 1
    elif patient.family_history == "both parents or early onset":
        score += 5
    # prediabetes
    if patient.prediabetes_diagnosed:
        score += 5
    return score


def insulin_resistance_tier(insulin):
    if insulin < 10:
        return "Normal"
    elif insulin < 15:
        return "Borderline"
    else:
        return "Resistant"


def get_homa_ir_range(tier):
    if tier == "Normal":
        return "1.0–1.9"
    elif tier == "Borderline":
        return "1.9–2.9"
    else:
        return "2.9–5.0"


def get_findrisc_label(score):
    if score < 7:
        return "low risk"
    elif score < 12:
        return "slightly elevated"
    elif score < 15:
        return "moderate"
    elif score < 20:
        return "high"
    else:
        return "very high"


def analyze_risk_factors(features: dict) -> list:
    """Analyze which features are contributing most to diabetes risk"""
    risk_factors = []
    normal_ranges = {
        'glucose': (70, 99),
        'blood_pressure': (60, 79),
        'skin_thickness': (10, 25),
        'insulin': (2, 25),
        'bmi': (18.5, 24.9),
        'diabetes_pedigree': (0, 0.5),
        'age': (0, 44),
        'pregnancies': (0, 2)
    }
    risk_descriptions = {
        'glucose': 'Elevated blood sugar',
        'blood_pressure': 'High blood pressure',
        'skin_thickness': 'Higher body fat (estimated)',
        'insulin': 'Insulin resistance',
        'bmi': 'Overweight/Obese',
        'diabetes_pedigree': 'Family history of diabetes',
        'age': 'Age-related risk',
        'pregnancies': 'Multiple pregnancies'
    }
    for feature, value in features.items():
        if feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            if value < min_val:
                if feature in ['bmi']:
                    risk_score = (min_val - value) / min_val
                    risk_factors.append((risk_score, f'Underweight (BMI: {value:.1f})'))
            elif value > max_val:
                if feature == 'glucose':
                    if value > 140:
                        risk_score = (value - max_val) / max_val * 2
                    else:
                        risk_score = (value - max_val) / max_val
                elif feature == 'bmi':
                    if value > 30:
                        risk_score = (value - max_val) / max_val * 1.5
                    else:
                        risk_score = (value - max_val) / max_val
                elif feature == 'diabetes_pedigree':
                    risk_score = value * 2
                elif feature == 'age':
                    risk_score = (value - max_val) / 20
                else:
                    risk_score = (value - max_val) / max_val
                risk_factors.append((risk_score, risk_descriptions.get(feature, f'High {feature}: {value}')))
    risk_factors.sort(reverse=True, key=lambda x: x[0])
    return risk_factors[:5]


if st.button("Predict Risk"):
    # Convert display values to internal format
    family_history_map = {
        "None": "none",
        "One parent or sibling": "one parent or sibling", 
        "Both parents or early onset": "both parents or early onset"
    }
    
    # Create PatientInputs object
    patient = PatientInputs(
        pregnancies=pregnancies,
        bmi=bmi,
        age=age,
        blood_pressure=blood_pressure,
        waist_circumference=waist_circumference,
        physical_activity=physical_activity,
        diet_quality=diet_quality,
        family_history=family_history_map[family_history],
        prediabetes_diagnosed=prediabetes_diagnosed,
        glucose=glucose,
        insulin=insulin
    )

    # Build feature vector using the proxy system
    features = build_feature_vector(patient)

    # Create input data for model (must be in correct order)
    input_data = pd.DataFrame([[
        features['pregnancies'],
        features['glucose'],
        features['blood_pressure'],
        features['skin_thickness'],
        features['insulin'],
        features['bmi'],
        features['diabetes_pedigree'],
        features['age']
    ]], columns=[
        'pregnancies',
        'glucose',
        'blood_pressure',
        'skin_thickness',
        'insulin',
        'bmi',
        'diabetes_pedigree',
        'age'
    ])

    probability = model.predict_proba(input_data)[0][1]
    risk_pct = min(max(int(round(probability * 100)), 0), 100)
    content = generate_results_content(patient, features, risk_pct)
    content["baseline"] = {
        "waist": float(patient.waist_circumference),
        "bmi": float(patient.bmi),
        "bp": float(patient.blood_pressure),
        "gluc": float(features['glucose']),
    }
    st.session_state["results_data"] = content
    st.session_state["page"] = "results"
    st.rerun()
