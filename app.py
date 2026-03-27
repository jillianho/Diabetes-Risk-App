import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from diabetes_proxies import PatientInputs, build_feature_vector, generate_results_content

st.set_page_config(
    page_title="Diabetes Risk & Intervention Simulator",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


def load_css(file_name: str = "styles.css") -> None:
    css_path = Path(__file__).with_name(file_name)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def build_result_summary(data: dict) -> str:
    lines = [
        "Diabetes Risk Summary",
        "=====================",
        f"Estimated risk: {data.get('risk_pct', 'N/A')}% ({data.get('risk_label', 'N/A')})",
        f"Fasting glucose: {data.get('glucose_display', 'N/A')} mg/dL",
        f"Insulin resistance tier: {data.get('ir_tier', 'N/A')} (HOMA-IR {data.get('homa_range', 'N/A')})",
        f"FINDRISC score: {data.get('findrisc', 'N/A')} ({data.get('findrisc_label', 'N/A')})",
        "",
        "Top actions:",
    ]
    for idx, action in enumerate(data.get("actions", [])[:5], start=1):
        lines.append(f"{idx}. {action.get('title', 'N/A')} - {action.get('delta', '')}")
    lines.extend([
        "",
        "Important: This tool is not medical advice and is not a diagnosis.",
        "Confirm decisions with a licensed healthcare professional.",
    ])
    return "\n".join(lines)


load_css()

if "page" not in st.session_state:
    st.session_state["page"] = "input"

if st.session_state["page"] == "results":
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
            st.info("Some inputs (such as glucose/insulin/skin thickness/proxy pedigree) may be estimated using heuristics when not provided directly. Use clinical lab tests for confirmation.")
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
            st.download_button(
                label="Download plain-text result summary",
                data=build_result_summary(data),
                file_name="diabetes_risk_summary.txt",
                mime="text/plain",
            )
            st.warning("This calculator does not provide medical advice or diagnosis. Clinical interpretation by a licensed professional is required.")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state["page"] = "input"
        st.rerun()

    st.stop()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Header with better styling
st.markdown('<h1 class="main-header">🩺 Diabetes Risk & Intervention Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get personalized insights about your diabetes risk based on your health profile</p>', unsafe_allow_html=True)
st.error("Medical disclaimer: This app provides an educational risk estimate only and is not a diagnosis or treatment recommendation. Always consult a licensed healthcare professional.")

with st.expander("ℹ️ What do these measurements mean?"):
    st.markdown("""
    **Glucose**: Blood sugar level from a fasting blood test (normal: 70-99 mg/dL)

    **Blood Pressure**: Diastolic pressure (normal: <80 mmHg) - this is the bottom number in a blood pressure reading like 120/80

    **Insulin**: Fasting insulin level (normal: 2-25 μU/mL)

    **BMI**: Body Mass Index = weight(kg) / height(m)²

    **Waist Circumference**: Measured around the narrowest part of your waist

    *If you don't have recent medical test results, the app will estimate values based on your health profile.*

    **Important limitations:**
    - Several values can be estimated using heuristics (approximations), not lab measurements.
    - Model probabilities may not be perfectly calibrated for every population.
    - This is a screening aid, not medical advice.
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

# Soft sanity warnings for improbable values.
input_warnings = []
if age < 18:
    input_warnings.append("Age under 18: this model was not designed for pediatric diagnosis.")
if bmi < 15:
    input_warnings.append("BMI is very low; confirm your value.")
if blood_pressure < 50:
    input_warnings.append("Diastolic blood pressure looks unusually low; confirm your value.")
if waist_circumference < 55:
    input_warnings.append("Waist circumference looks unusually low; confirm your value.")
if know_glucose and glucose is not None and glucose < 60:
    input_warnings.append("Fasting glucose under 60 mg/dL is uncommon; verify this measurement.")

for warning in input_warnings:
    st.warning(warning)

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
    required_errors = []
    if age <= 0:
        required_errors.append("Age must be greater than 0.")
    if bmi <= 0:
        required_errors.append("BMI must be greater than 0.")
    if blood_pressure <= 0:
        required_errors.append("Diastolic blood pressure must be greater than 0.")
    if waist_circumference <= 0:
        required_errors.append("Waist circumference must be greater than 0.")

    if required_errors:
        st.error("Please fix these values before predicting:\n- " + "\n- ".join(required_errors))
        st.stop()

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
