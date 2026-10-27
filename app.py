
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="LendWise | Loan Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- PROFESSIONAL LIGHT THEME ----------
st.markdown("""
<style>
.stApp {
    background: #F3F6FB;
    color: #17243B;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1250px;
    padding: 2rem 2rem 3rem;
}

/* Main text */
h1, h2, h3, h4, p, label, .stMarkdown {
    color: #17243B;
}
.hero {
    background: linear-gradient(120deg, #142442, #24516B);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.3rem;
    margin-bottom: 8px;
}
.hero p {
    color: #E0EAF5 !important;
    font-size: 1rem;
}
.eyebrow {
    color: #087F80 !important;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 2px;
}
.section-heading {
    color: #17243B;
    font-size: 1.25rem;
    font-weight: 800;
    margin: 8px 0 4px;
}
.section-description {
    color: #64748B !important;
    margin-bottom: 18px;
}

/* White input panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-color: #DFE6F0;
    border-radius: 16px;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label {
    color: #344054 !important;
    font-weight: 600 !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #BCC8D8 !important;
}
div[data-baseweb="select"] * {
    color: #17243B !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #17243B !important;
    border: 1px solid #BCC8D8 !important;
    border-radius: 8px !important;
    -webkit-text-fill-color: #17243B !important;
}
div[data-testid="stNumberInput"] button {
    background: #F0F5FA !important;
    color: #17243B !important;
}

/* Buttons */
div[data-testid="stFormSubmitButton"] button {
    background: #13B8A5 !important;
    color: #082F3A !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 800 !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #42DCC7 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DFE6F0;
    border-radius: 14px;
    padding: 18px;
}
div[data-testid="stMetricLabel"] {
    color: #64748B !important;
}
div[data-testid="stMetricValue"] {
    color: #17243B !important;
    font-weight: 800;
}

/* Result cards */
.result-card {
    padding: 24px;
    border-radius: 16px;
    margin: 12px 0;
}
.approved {
    background: #E8FBF3;
    border: 1px solid #A8E8D0;
}
.rejected {
    background: #FFF0F0;
    border: 1px solid #F5C4C7;
}
.result-card h2 {
    margin-top: 8px;
}
div[data-testid="stProgress"] > div > div {
    background: #13B8A5;
}
hr {
    border-color: #DFE6F0;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD TRAINED MODEL ----------
@st.cache_resource
def load_model():
    bundle = joblib.load("loan_model.pkl")
    return bundle["model"], bundle["feature_names"]


try:
    model, feature_names = load_model()
except Exception as e:
    st.error(f"Model load nahi ho saka: {e}")
    st.stop()


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <div style="color:#5BE6D0;font-weight:800;letter-spacing:2px;">
        AI-POWERED FINANCIAL ANALYSIS
    </div>
    <h1>🏦 LendWise</h1>
    <p>Loan Intelligence Center — make data-driven predictions
    using your trained Machine Learning model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">APPLICATION WORKSPACE</div>',
            unsafe_allow_html=True)
st.markdown('<div class="section-heading">Applicant Information</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="section-description">Complete the details below '
    'to generate a loan prediction.</p>',
    unsafe_allow_html=True
)


# ---------- MAIN SCREEN FORM ----------
with st.form("loan_form"):

    # Personal information
    with st.container(border=True):
        st.markdown("### 👤 01 — Personal Details")

        p1, p2, p3 = st.columns(3)

        with p1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])

        with p2:
            dependents = st.selectbox(
                "Number of Dependents", ["0", "1", "2", "3+"]
            )
            education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"]
            )

        with p3:
            self_employed = st.selectbox(
                "Self Employed", ["No", "Yes"]
            )
            property_area = st.selectbox(
                "Property Area", ["Semiurban", "Urban", "Rural"]
            )

    st.write("")

    # Financial information
    with st.container(border=True):
        st.markdown("### 💳 02 — Financial Details")

        f1, f2 = st.columns(2)

        with f1:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0,
                value=5000,
                step=500
            )
            loan_amount = st.number_input(
                "Loan Amount (dataset units)",
                min_value=0,
                value=150,
                step=10
            )

        with f2:
            coapplicant_income = st.number_input(
                "Co-applicant Income",
                min_value=0,
                value=0,
                step=500
            )
            loan_term = st.selectbox(
                "Loan Term (months)",
                [360, 180, 120, 60, 36, 12]
            )

    st.write("")

    # Credit details
    with st.container(border=True):
        st.markdown("### 📊 03 — Credit Assessment")

        c1, c2 = st.columns(2)

        with c1:
            credit_history = st.selectbox(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x: (
                    "Good / Available (1)"
                    if x == 1.0
                    else "Poor / Not available (0)"
                )
            )

        with c2:
            st.info(
                "Check that the information entered matches "
                "the applicant's records."
            )

    st.write("")
    submitted = st.form_submit_button(
        "✦  ANALYZE LOAN APPLICATION",
        use_container_width=True
    )


# ---------- PREDICTION ----------
if submitted:

    values = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area_Rural": int(property_area == "Rural"),
        "Property_Area_Semiurban": int(property_area == "Semiurban"),
        "Property_Area_Urban": int(property_area == "Urban"),
        "Dependents_1": int(dependents == "1"),
        "Dependents_2": int(dependents == "2"),
        "Dependents_3+": int(dependents == "3+"),
    }

    input_df = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        approval = float(
            probabilities[list(model.classes_).index(1)] * 100
        )
        rejection = 100 - approval

        st.markdown("---")
        st.markdown(
            '<div class="eyebrow">ANALYSIS COMPLETE</div>',
            unsafe_allow_html=True
        )
        st.markdown("## 📈 Prediction Results")

        r1, r2 = st.columns(2)
        r1.metric("Approval Probability", f"{approval:.2f}%")
        r2.metric("Rejection Probability", f"{rejection:.2f}%")

        st.markdown("#### Probability Distribution")
        st.write(f"Approval — {approval:.2f}%")
        st.progress(int(round(approval)))
        st.write(f"Rejection — {rejection:.2f}%")
        st.progress(int(round(rejection)))

        if prediction == 1:
            st.markdown("""
            <div class="result-card approved">
                <div style="color:#08795C;font-weight:800;">
                    MODEL PREDICTION
                </div>
                <h2 style="color:#08795C;">✓ Loan Approved</h2>
                <p style="color:#356B5A;">
                    The model predicts the approval class
                    for this application.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card rejected">
                <div style="color:#B42332;font-weight:800;">
                    MODEL PREDICTION
                </div>
                <h2 style="color:#B42332;">✕ Loan Rejected</h2>
                <p style="color:#8D4B51;">
                    The model predicts the rejection class
                    for this application.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Applicant Summary")

        summary = pd.DataFrame([{
            "Gender": gender,
            "Marital Status": married,
            "Dependents": dependents,
            "Education": education,
            "Self Employed": self_employed,
            "Applicant Income": applicant_income,
            "Co-applicant Income": coapplicant_income,
            "Loan Amount": loan_amount,
            "Loan Term (months)": loan_term,
            "Credit History": credit_history,
            "Property Area": property_area,
            "Approval Probability (%)": round(approval, 2),
            "Rejection Probability (%)": round(rejection, 2),
            "Predicted Class": int(prediction)
        }])

        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download Prediction Report",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="loan_prediction_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction nahi ho saki: {e}")

else:
    st.markdown("")
    st.caption(
        "Fill in the applicant details and click "
        "'Analyze Loan Application' to see the results."
    )


# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "LendWise | Educational Machine Learning Project | "
    "Predictions are estimates, not actual lending decisions."
)
