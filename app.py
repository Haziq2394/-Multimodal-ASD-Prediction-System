import streamlit as st
import torch
import numpy as np
import joblib
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.fusion import FusionModel
from src.standalone_models import CNNClassifier, BiGRUClassifier

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ── Background + Styling ──────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=1920");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.75);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
[data-testid="stSidebar"] {
    background-color: rgba(10, 25, 47, 0.95) !important;
    border-right: 1px solid #1e3a5f;
}
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card h2 { color: #4fc3f7; margin: 0; font-size: 2rem; }
.metric-card p  { color: #aaa; margin: 0; font-size: 0.8rem; letter-spacing: 1px; }
.banner {
    background: linear-gradient(135deg, rgba(13,71,161,0.8), rgba(0,150,136,0.8));
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    text-align: center;
    color: white;
    font-size: 0.95rem;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler   = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    dt_model = joblib.load("models/dt_model.pkl")

    cnn_model = CNNClassifier()
    cnn_model.load_state_dict(torch.load("models/cnn_standalone.pth", map_location="cpu"))
    cnn_model.eval()

    bigru_model = BiGRUClassifier()
    bigru_model.load_state_dict(torch.load("models/bigru_standalone.pth", map_location="cpu"))
    bigru_model.eval()

    fusion = FusionModel(dt_feature_size=len(feature_cols))
    fusion.load_state_dict(torch.load("models/fusion_model.pth", map_location="cpu"))
    fusion.eval()

    comparison = {}
    if os.path.exists("models/comparison_results.json"):
        with open("models/comparison_results.json") as f:
            comparison = json.load(f)

    return scaler, encoders, feature_cols, dt_model, cnn_model, bigru_model, fusion, comparison

scaler, encoders, feature_cols, dt_model, cnn_model, bigru_model, fusion_model, comparison = load_models()

SEQ_COLS = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10_Autism_Spectrum_Quotient']

DEFAULT_FUSION_METRICS = {"accuracy": 0.995, "precision": 1.0, "recall": 0.991, "f1": 0.995}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    page = st.radio("", ["🔍 Predict ASD", "📊 Model Dashboard", "ℹ️ About"])
    st.divider()

    st.markdown("## 📈 Fusion Model Performance")
    fusion_metrics = comparison.get("Fusion", DEFAULT_FUSION_METRICS)
    st.markdown("**Accuracy**")
    st.markdown(f"### {fusion_metrics['accuracy']*100:.1f}%")
    st.markdown("**Precision**")
    st.markdown(f"### {fusion_metrics['precision']*100:.1f}%")
    st.markdown("**Recall**")
    st.markdown(f"### {fusion_metrics['recall']*100:.1f}%")
    st.markdown("**F1 Score**")
    st.markdown(f"### {fusion_metrics['f1']*100:.1f}%")
    st.divider()

    st.markdown("## ⚙️ System Info")
    st.markdown("**Algorithm:** CNN + Bi-GRU + Decision Tree")
    st.markdown("**Dataset:** Kaggle ASD Behavioural")
    st.markdown("**Samples:** 1,985")
    st.markdown("**Features:** 26")
    st.markdown("**Framework:** PyTorch + Scikit-learn")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT ASD
# ══════════════════════════════════════════════════════════════
if page == "🔍 Predict ASD":
    st.title("🧠 Multimodal ASD Prediction System")
    st.markdown("### Autism Spectrum Disorder Detection in Children")
    st.divider()

    # Metric cards (fusion headline numbers)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{fusion_metrics["accuracy"]*100:.1f}%</h2><p>ACCURACY</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{fusion_metrics["f1"]*100:.1f}%</h2><p>F1-SCORE</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{fusion_metrics["precision"]*100:.1f}%</h2><p>PRECISION</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h2>{fusion_metrics["recall"]*100:.1f}%</h2><p>RECALL</p></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card"><h2>26</h2><p>FEATURES</p></div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="metric-card"><h2>1,985</h2><p>SAMPLES</p></div>', unsafe_allow_html=True)

    st.divider()

    # ── Model Selector ──────────────────────────────────────────
    st.subheader("🧬 Choose Prediction Model")
    model_choice = st.selectbox(
        "Select which model to use for this prediction",
        ["Combined (CNN + Bi-GRU + Decision Tree)", "CNN Only", "Bi-GRU Only", "Decision Tree Only"]
    )
    if model_choice != "Combined (CNN + Bi-GRU + Decision Tree)":
        st.info(f"Running prediction using **{model_choice}** — this isolates that branch's contribution for comparison purposes.")

    st.divider()
    st.subheader("📋 Enter Child's Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Behavioural Screening (A1–A10)**")
        A1  = 1 if st.selectbox("A1 - Does not respond to name", ["No", "Yes"]) == "Yes" else 0
        A2  = 1 if st.selectbox("A2 - Poor eye contact", ["No", "Yes"]) == "Yes" else 0
        A3  = 1 if st.selectbox("A3 - Lines up objects", ["No", "Yes"]) == "Yes" else 0
        A4  = 1 if st.selectbox("A4 - Repetitive behaviour", ["No", "Yes"]) == "Yes" else 0
        A5  = 1 if st.selectbox("A5 - Does not point to share interest", ["No", "Yes"]) == "Yes" else 0
        A6  = 1 if st.selectbox("A6 - Does not look where pointed", ["No", "Yes"]) == "Yes" else 0
        A7  = 1 if st.selectbox("A7 - Unusual sensory behaviour", ["No", "Yes"]) == "Yes" else 0
        A8  = 1 if st.selectbox("A8 - Lack of imaginative play", ["No", "Yes"]) == "Yes" else 0
        A9  = 1 if st.selectbox("A9 - Does not follow gaze", ["No", "Yes"]) == "Yes" else 0
        A10 = 1 if st.selectbox("A10 - Autism Spectrum Quotient", ["No", "Yes"]) == "Yes" else 0

    with col2:
        st.markdown("**Clinical Scores**")
        social_resp  = st.number_input("Social Responsiveness Scale", 0.0, 200.0, 50.0)
        age          = st.number_input("Age (Years)", 1, 18, 5)
        qchat        = st.number_input("Qchat-10 Score", 0.0, 10.0, 3.0)
        cars         = st.selectbox("Childhood Autism Rating Scale", [1, 2, 3, 4])
        st.markdown("**Medical History**")
        speech_delay = st.selectbox("Speech Delay / Language Disorder", ["Yes", "No"])
        learning     = st.selectbox("Learning Disorder", ["Yes", "No"])
        genetic      = st.selectbox("Genetic Disorders", ["Yes", "No"])
        depression   = st.selectbox("Depression", ["Yes", "No"])
        global_delay = st.selectbox("Global Developmental Delay", ["Yes", "No"])
        social_beh   = st.selectbox("Social / Behavioural Issues", ["Yes", "No"])
        anxiety      = st.selectbox("Anxiety Disorder", ["Yes", "No"])

    with col3:
        st.markdown("**Demographics**")
        sex        = st.selectbox("Sex", ["M", "F"])
        ethnicity  = st.selectbox("Ethnicity", ["White European", "Asian", "Black",
                                                 "Middle Eastern", "Hispanic", "Others"])
        jaundice   = st.selectbox("Jaundice at Birth", ["Yes", "No"])
        family_asd = st.selectbox("Family Member with ASD", ["Yes", "No"])
        who_test   = st.selectbox("Who Completed the Test", ["Parent", "Family Member",
                                                              "Health Care Professional",
                                                              "Self", "Others"])

    st.divider()

    if st.button("🔍 Predict ASD", use_container_width=True, type="primary"):
        row = {
            "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
            "A6": A6, "A7": A7, "A8": A8, "A9": A9,
            "A10_Autism_Spectrum_Quotient": A10,
            "Social_Responsiveness_Scale": social_resp,
            "Age_Years": age,
            "Qchat_10_Score": qchat,
            "Speech Delay/Language Disorder": speech_delay,
            "Learning disorder": learning,
            "Genetic_Disorders": genetic,
            "Depression": depression,
            "Global developmental delay/intellectual disability": global_delay,
            "Social/Behavioural Issues": social_beh,
            "Childhood Autism Rating Scale": cars,
            "Anxiety_disorder": anxiety,
            "Sex": sex,
            "Ethnicity": ethnicity,
            "Jaundice": jaundice,
            "Family_mem_with_ASD": family_asd,
            "Who_completed_the_test": who_test
        }

        df_input = pd.DataFrame([row])

        # Apply the SAME encoders used at training time
        # (fixes the bug where fit_transform on one row always maps to 0)
        for col, le in encoders.items():
            if col in df_input.columns:
                val = str(df_input[col].iloc[0])
                if val in le.classes_:
                    df_input[col] = le.transform([val])
                else:
                    # unseen category safeguard: fall back to the first known class
                    df_input[col] = le.transform([le.classes_[0]])

        # Enforce exact training column order before scaling
        df_input = df_input[feature_cols]

        X        = df_input.values.astype(np.float32)
        X_scaled = scaler.transform(X).astype(np.float32)
        seq      = X_scaled[0, :10].reshape(1, 10)

        cnn_input = torch.tensor(seq.reshape(1, 2, 5, 1), dtype=torch.float32)
        gru_input = torch.tensor(seq.reshape(1, 10, 1),   dtype=torch.float32)
        dt_input  = torch.tensor(X_scaled,                dtype=torch.float32)

        with torch.no_grad():
            cnn_prob   = cnn_model(cnn_input).item()
            bigru_prob = bigru_model(gru_input).item()
        dt_prob = dt_model.predict_proba(X_scaled)[0][1]

        with torch.no_grad():
            fusion_prob = fusion_model(cnn_input, gru_input, dt_input).item()

        # Pick the probability that matches the selected model
        if model_choice == "CNN Only":
            prob, active_name = cnn_prob, "CNN"
        elif model_choice == "Bi-GRU Only":
            prob, active_name = bigru_prob, "Bi-GRU"
        elif model_choice == "Decision Tree Only":
            prob, active_name = dt_prob, "Decision Tree"
        else:
            prob, active_name = fusion_prob, "Fusion (Combined)"

        prediction = "ASD Traits Detected" if prob > 0.5 else "No ASD Traits Detected"
        confidence = prob if prob > 0.5 else 1 - prob

        st.divider()
        st.caption(f"Prediction made using: **{active_name}**")
        if prob > 0.5:
            st.error(f"## 🔴 {prediction}")
        else:
            st.success(f"## 🟢 {prediction}")

        st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        st.markdown("**All Model Outputs (for comparison):**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("CNN", f"{cnn_prob * 100:.1f}%")
        mc2.metric("Bi-GRU", f"{bigru_prob * 100:.1f}%")
        mc3.metric("Decision Tree", f"{dt_prob * 100:.1f}%")
        mc4.metric("Fusion", f"{fusion_prob * 100:.1f}%")

        st.divider()
        st.subheader("📊 Prediction Visualization")
        col_a, col_b = st.columns(2)

        with col_a:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            fig1.patch.set_alpha(0)
            ax1.set_facecolor("none")
            color = '#e74c3c' if prob > 0.5 else '#2ecc71'
            ax1.barh([active_name], [confidence * 100], color=color, height=0.4)
            ax1.set_xlim(0, 100)
            ax1.set_xlabel("Confidence (%)", color='white')
            ax1.set_title(f"{active_name} — Confidence", color='white')
            ax1.axvline(x=50, color='gray', linestyle='--', linewidth=1)
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_visible(False)
            st.pyplot(fig1)

        with col_b:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            models  = ["CNN", "Bi-GRU", "Decision\nTree", "Fusion"]
            scores  = [cnn_prob * 100, bigru_prob * 100, dt_prob * 100, fusion_prob * 100]
            colors2 = ['#e74c3c' if s > 50 else '#2ecc71' for s in scores]
            ax2.bar(models, scores, color=colors2)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("ASD Probability (%)", color='white')
            ax2.set_title("All Models — This Prediction", color='white')
            ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1)
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_visible(False)
            st.pyplot(fig2)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.title("📊 Model Dashboard")
    st.markdown("Performance overview of the multimodal ASD prediction system.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        models  = ["CNN", "Bi-GRU", "Decision\nTree", "Fusion\nModel"]
        default_accs = {"CNN": 91.2, "Bi-GRU": 90.8, "Decision Tree": 95.97, "Fusion": 99.5}
        accs = [
            comparison.get("CNN", {}).get("accuracy", default_accs["CNN"]/100) * 100,
            comparison.get("Bi-GRU", {}).get("accuracy", default_accs["Bi-GRU"]/100) * 100,
            comparison.get("Decision Tree", {}).get("accuracy", default_accs["Decision Tree"]/100) * 100,
            comparison.get("Fusion", {}).get("accuracy", default_accs["Fusion"]/100) * 100,
        ]
        colors  = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']
        bars    = ax.bar(models, accs, color=colors)
        ax.set_ylim(max(min(accs) - 5, 0), 105)
        ax.set_ylabel("Accuracy (%)", color='white')
        ax.set_title("Individual vs Fusion Model Accuracy", color='white')
        ax.tick_params(colors='white')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', color='white', fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("Fusion Model — Metrics Overview")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values  = [fusion_metrics["accuracy"]*100, fusion_metrics["precision"]*100,
                   fusion_metrics["recall"]*100, fusion_metrics["f1"]*100]
        ax2.bar(metrics, values, color='#4fc3f7')
        ax2.set_ylim(max(min(values) - 5, 0), 101)
        ax2.set_ylabel("Score (%)", color='white')
        ax2.set_title("Fusion Model Metrics", color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_visible(False)
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 Model Architecture Summary")
    st.markdown("""
    | Component | Input | Output | Purpose |
    |---|---|---|---|
    | **CNN** | A1–A10 as 2×5 heatmap | 64 features | Visual pattern detection |
    | **Bi-GRU** | A1–A10 as sequence | 64 features | Sequential behaviour analysis |
    | **Decision Tree** | All 26 features | Probability | Tabular clinical data |
    | **Fusion Layer** | CNN + Bi-GRU + DT | ASD / No ASD | Final prediction |
    """)

    st.divider()
    st.subheader("📦 Dataset Info")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Samples", "1,985")
    d2.metric("ASD Positive", "1,074")
    d3.metric("ASD Negative", "911")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.divider()

    st.markdown("""
    ## 🧠 Multimodal ASD Prediction System

    This system predicts **Autism Spectrum Disorder (ASD) traits in children** using a
    multimodal deep learning approach that combines three different models, and lets
    the user compare each branch individually against the combined fusion prediction:

    ### Models Used
    - **CNN (Convolutional Neural Network)** — Analyses behavioural screening scores
      (A1–A10) as a 2D heatmap image to detect visual patterns
    - **Bi-GRU (Bidirectional Gated Recurrent Unit)** — Processes the same A1–A10 scores
      as a temporal sequence to capture behavioural progression
    - **Decision Tree** — Analyses all 26 clinical and demographic features using
      classical machine learning

    ### Fusion Strategy
    The outputs of all three models are **concatenated and passed through a fusion
    classifier** that makes the final ASD prediction. Users can also select an
    individual model to see its standalone contribution.

    ### Dataset
    - **Source:** Kaggle Autistic Child Behavioural Dataset
    - **Samples:** 1,985 children
    - **Features:** 26 behavioural, clinical, and demographic features
    - **Target:** ASD Traits (Yes / No)

    ### Disclaimer
    > ⚠️ This tool is for **research and educational purposes only**.
    > It does **not** replace professional medical diagnosis.
    > Always consult a qualified healthcare professional for ASD assessment.
    """)