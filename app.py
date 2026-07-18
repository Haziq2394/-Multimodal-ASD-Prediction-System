import streamlit as st
import torch
import numpy as np
import joblib
import sys
import os
import json
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.fusion import FusionModel
from src.standalone_models import CNNClassifier, BiGRUClassifier
from src.resnet_cnn import ResNetMRIClassifier
from src.mri_dataset import ABIDESliceDataset

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
div[data-baseweb="select"] > div {
    border-color: rgba(255,255,255,0.15) !important;
    transition: border-color 0.15s ease;
}
div[data-baseweb="select"] > div:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"]:focus-within > div {
    outline: none !important;
    box-shadow: none !important;
    border-color: rgba(255,255,255,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

DEFAULT_FUSION_METRICS = {"accuracy": 0.9622, "precision": 0.9587, "recall": 0.9721, "f1": 0.9654}

# ── Load Models (Kaggle system) ────────────────────────────────
@st.cache_resource
def load_models():
    scaler   = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    dt_model = joblib.load("models/dt_model_pruned.pkl")

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

    late_fusion_meta = joblib.load("models/late_fusion_meta_classifier.pkl")
    late_fusion_stats = {}
    if os.path.exists("models/late_fusion_statistical_results.json"):
        with open("models/late_fusion_statistical_results.json") as f:
            late_fusion_stats = json.load(f)

    return (scaler, encoders, feature_cols, dt_model, cnn_model, bigru_model,
            fusion, comparison, late_fusion_meta, late_fusion_stats)

(scaler, encoders, feature_cols, dt_model, cnn_model, bigru_model, fusion_model,
 comparison, late_fusion_meta, late_fusion_stats) = load_models()

SEQ_COLS = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10_Autism_Spectrum_Quotient']

# ── Load Models (ABIDE system) — lazy, only when that page is opened ──
ABIDE_FEATURE_COLS = ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ"]
ABIDE_PHENOTYPIC_CSV = "Phenotypic_V1_0b_preprocessed1.csv"

@st.cache_resource
def load_abide_models():
    dt_abide = joblib.load("models/dt_abide_pruned.pkl")
    dt_abide_scaler = joblib.load("models/abide_dt_scaler.pkl")
    dt_abide_feature_cols = joblib.load("models/abide_dt_feature_cols.pkl")

    bigru_pheno = BiGRUClassifier()
    bigru_pheno.load_state_dict(torch.load("models/bigru_phenotypic.pth", map_location="cpu"))
    bigru_pheno.eval()

    cnn_mri = ResNetMRIClassifier(freeze_backbone=False)
    cnn_mri.load_state_dict(torch.load("models/resnet_mri_final.pth", map_location="cpu"))
    cnn_mri.eval()

    meta_clf = joblib.load("models/abide_late_fusion_meta_classifier.pkl")

    pheno_df = pd.read_csv(ABIDE_PHENOTYPIC_CSV)
    iq_cols = ["FIQ", "VIQ", "PIQ"]
    for c in iq_cols:
        pheno_df[c] = pheno_df[c].replace(-9999, np.nan)
    pheno_df = pheno_df.dropna(subset=iq_cols)
    iq_mean = pheno_df[iq_cols].mean()
    iq_std  = pheno_df[iq_cols].std()

    abide_results = {}
    if os.path.exists("models/abide_late_fusion_results.json"):
        with open("models/abide_late_fusion_results.json") as f:
            abide_results = json.load(f)

    return (dt_abide, dt_abide_scaler, dt_abide_feature_cols,
            bigru_pheno, cnn_mri, meta_clf, iq_mean, iq_std, abide_results)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    page = st.radio("", ["🔍 Predict ASD", "🧬 Predict ASD (MRI + IQ, ABIDE)", "📊 Model Dashboard", "ℹ️ About"])
    st.divider()

    st.markdown("## 📈 Late Fusion Performance")
    fusion_metrics = late_fusion_stats.get("model_performance", {}).get("Late Fusion", DEFAULT_FUSION_METRICS)
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
    st.markdown("**Fusion:** Logistic Regression (late fusion)")
    st.markdown("**Dataset:** Kaggle ASD Behavioural")
    st.markdown("**Samples:** 1,985")
    st.markdown("**Features:** 26")
    st.markdown("**Framework:** PyTorch + Scikit-learn")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT ASD (KAGGLE: Behavioural)
# ══════════════════════════════════════════════════════════════
if page == "🔍 Predict ASD":
    # Research topic header
    st.markdown("""
    # MULTIMODAL AI SYSTEM FOR PREDICTING AUTISM SPECTRUM DISORDER IN CHILDREN USING CNN, Bi-GRU AND DECISION TREES
    """)

    # Project name (branded format)
    st.markdown("""
    ## 🧬 NeuroSequence — Behavioral Analysis (Kaggle)
    **Screening Scores + Clinical Features Multimodal Prediction**
    """)

    st.divider()

    # Metric cards (late fusion headline numbers)
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
        ["Combined (Late Fusion)", "CNN Only", "Bi-GRU Only", "Decision Tree Only"],
        key="kaggle_model_choice"
    )
    if model_choice != "Combined (Late Fusion)":
        st.info(f"Running prediction using **{model_choice}** — this isolates that branch's contribution for comparison purposes.")

    st.divider()
    st.subheader("📋 Enter Child's Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Behavioural Screening (A1–A10)**")
        A1  = 1 if st.selectbox("A1 - Does not respond to name", ["No", "Yes"], key="kaggle_a1") == "Yes" else 0
        A2  = 1 if st.selectbox("A2 - Poor eye contact", ["No", "Yes"], key="kaggle_a2") == "Yes" else 0
        A3  = 1 if st.selectbox("A3 - Lines up objects", ["No", "Yes"], key="kaggle_a3") == "Yes" else 0
        A4  = 1 if st.selectbox("A4 - Repetitive behaviour", ["No", "Yes"], key="kaggle_a4") == "Yes" else 0
        A5  = 1 if st.selectbox("A5 - Does not point to share interest", ["No", "Yes"], key="kaggle_a5") == "Yes" else 0
        A6  = 1 if st.selectbox("A6 - Does not look where pointed", ["No", "Yes"], key="kaggle_a6") == "Yes" else 0
        A7  = 1 if st.selectbox("A7 - Unusual sensory behaviour", ["No", "Yes"], key="kaggle_a7") == "Yes" else 0
        A8  = 1 if st.selectbox("A8 - Lack of imaginative play", ["No", "Yes"], key="kaggle_a8") == "Yes" else 0
        A9  = 1 if st.selectbox("A9 - Does not follow gaze", ["No", "Yes"], key="kaggle_a9") == "Yes" else 0
        A10 = 1 if st.selectbox("A10 - Autism Spectrum Quotient", ["No", "Yes"], key="kaggle_a10") == "Yes" else 0

    with col2:
        st.markdown("**Clinical Scores**")
        social_resp  = st.number_input("Social Responsiveness Scale", 0.0, 200.0, 50.0, key="kaggle_social_resp")
        age          = st.number_input("Age (Years)", 1, 18, 5, key="kaggle_age")
        qchat        = st.number_input("Qchat-10 Score", 0.0, 10.0, 3.0, key="kaggle_qchat")
        cars         = st.selectbox("Childhood Autism Rating Scale", [1, 2, 3, 4], key="kaggle_cars")
        st.markdown("**Medical History**")
        speech_delay = st.selectbox("Speech Delay / Language Disorder", ["Yes", "No"], key="kaggle_speech")
        learning     = st.selectbox("Learning Disorder", ["Yes", "No"], key="kaggle_learning")
        genetic      = st.selectbox("Genetic Disorders", ["Yes", "No"], key="kaggle_genetic")
        depression   = st.selectbox("Depression", ["Yes", "No"], key="kaggle_depression")
        global_delay = st.selectbox("Global Developmental Delay", ["Yes", "No"], key="kaggle_global")
        social_beh   = st.selectbox("Social / Behavioural Issues", ["Yes", "No"], key="kaggle_social")
        anxiety      = st.selectbox("Anxiety Disorder", ["Yes", "No"], key="kaggle_anxiety")

    with col3:
        st.markdown("**Demographics**")
        sex        = st.selectbox("Sex", ["M", "F"], key="kaggle_sex")
        ethnicity  = st.selectbox("Ethnicity", ["White European", "Asian", "Black",
                                                 "Middle Eastern", "Hispanic", "Others"], key="kaggle_ethnicity")
        jaundice   = st.selectbox("Jaundice at Birth", ["Yes", "No"], key="kaggle_jaundice")
        family_asd = st.selectbox("Family Member with ASD", ["Yes", "No"], key="kaggle_family_asd")
        who_test   = st.selectbox("Who Completed the Test", ["Parent", "Family Member",
                                                              "Health Care Professional",
                                                              "Self", "Others"], key="kaggle_who_test")

    st.divider()

    if st.button("🔍 Predict ASD", use_container_width=True, type="primary", key="kaggle_predict_btn"):
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
        for col, le in encoders.items():
            if col in df_input.columns:
                val = str(df_input[col].iloc[0])
                if val in le.classes_:
                    df_input[col] = le.transform([val])
                else:
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

        # REAL late fusion: logistic regression over [CNN, Bi-GRU, DT] probabilities
        meta_X = np.array([[cnn_prob, bigru_prob, dt_prob]], dtype=np.float32)
        fusion_prob = late_fusion_meta.predict_proba(meta_X)[0][1]

        # Pick the probability that matches the selected model
        if model_choice == "CNN Only":
            prob, active_name = cnn_prob, "CNN"
        elif model_choice == "Bi-GRU Only":
            prob, active_name = bigru_prob, "Bi-GRU"
        elif model_choice == "Decision Tree Only":
            prob, active_name = dt_prob, "Decision Tree"
        else:
            prob, active_name = fusion_prob, "Late Fusion (Combined)"

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
        mc4.metric("Late Fusion", f"{fusion_prob * 100:.1f}%")

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
            models  = ["CNN", "Bi-GRU", "Decision\nTree", "Late\nFusion"]
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
# PAGE 1B — PREDICT ASD (ABIDE: MRI + PHENOTYPIC)
# ══════════════════════════════════════════════════════════════
elif page == "🧬 Predict ASD (MRI + IQ, ABIDE)":
    # Research topic header
    st.markdown("""
    # MULTIMODAL AI SYSTEM FOR PREDICTING AUTISM SPECTRUM DISORDER IN CHILDREN USING CNN, Bi-GRU AND DECISION TREES
    """)

    # Project name (branded format)
    st.markdown("""
    ## 🧬 NeuroSequence — MRI + Phenotypic (ABIDE)
    **Structural MRI + IQ-based Multimodal Prediction**
    """)

    st.divider()

    if not os.path.exists(ABIDE_PHENOTYPIC_CSV):
        st.error(
            f"Could not find `{ABIDE_PHENOTYPIC_CSV}` in the project root. "
            "This file is needed to reproduce the exact IQ normalization used at training time. "
            "Place it alongside app.py and reload."
        )
        st.stop()

    missing_models = [p for p in [
        "models/dt_abide_pruned.pkl", "models/abide_dt_scaler.pkl", "models/abide_dt_feature_cols.pkl",
        "models/bigru_phenotypic.pth", "models/resnet_mri_final.pth",
        "models/abide_late_fusion_meta_classifier.pkl"
    ] if not os.path.exists(p)]
    if missing_models:
        st.error("Missing required ABIDE model file(s):\n\n" + "\n".join(f"- `{p}`" for p in missing_models))
        st.stop()

    (dt_abide, dt_abide_scaler, dt_abide_feature_cols,
     bigru_pheno, cnn_mri, meta_clf, iq_mean, iq_std, abide_results) = load_abide_models()

    # Reference performance from the held-out ABIDE test set (n=174)
    if abide_results.get("model_performance"):
        st.subheader("📈 Reference Performance")
        perf = abide_results["model_performance"]
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, name in zip([rc1, rc2, rc3, rc4], ["CNN", "Bi-GRU", "Decision Tree", "Late Fusion"]):
            if name in perf:
                col.metric(name, f"{perf[name]['accuracy']*100:.1f}%", help=f"F1: {perf[name]['f1']:.3f}")
            else:
                col.metric(name, "N/A")
    else:
        st.warning(
            "No reference performance found — `models/abide_late_fusion_results.json` is missing "
            "or empty. Run `generate_abide_fusion.py` to produce it."
        )

    st.divider()
    st.subheader("📋 Enter Subject Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demographics**")
        age_scan = st.number_input("Age at Scan (Years)", 5.0, 65.0, 15.0, step=0.5, key="abide_age")
        sex_choice = st.selectbox("Sex", ["Male", "Female"], key="abide_sex")
        sex_val = 1 if sex_choice == "Male" else 2

    with col2:
        st.markdown("**IQ Scores**")
        fiq = st.number_input("Full-Scale IQ (FIQ)", 40.0, 160.0, 100.0, key="abide_fiq")
        viq = st.number_input("Verbal IQ (VIQ)", 40.0, 160.0, 100.0, key="abide_viq")
        piq = st.number_input("Performance IQ (PIQ)", 40.0, 160.0, 100.0, key="abide_piq")

    st.markdown("**Structural MRI Scan**")
    uploaded_mri = st.file_uploader(
        "Upload a preprocessed axial slice (.nii)",
        type=["nii", "gz"],
        help="Expects the same preprocessed 2D axial slice format used in the ABIDE1_Preprocess pipeline.",
        key="abide_mri_upload"
    )

    st.divider()

    if st.button("🧬 Predict ASD (ABIDE)", use_container_width=True, type="primary", key="abide_predict_btn"):
        if uploaded_mri is None:
            st.warning("Please upload an MRI slice file (.nii) before predicting.")
            st.stop()

        # ── DT (clinical: age, sex, FIQ, VIQ, PIQ) — use saved column order ──
        row_lookup = {"AGE_AT_SCAN": age_scan, "SEX": sex_val, "FIQ": fiq, "VIQ": viq, "PIQ": piq}
        dt_row = np.array([[row_lookup[c] for c in dt_abide_feature_cols]], dtype=np.float32)
        dt_scaled = dt_abide_scaler.transform(dt_row).astype(np.float32)
        dt_prob = dt_abide.predict_proba(dt_scaled)[0][1]

        # ── Bi-GRU (phenotypic sequence: FIQ -> VIQ -> PIQ) ─────────
        iq_row = pd.Series({"FIQ": fiq, "VIQ": viq, "PIQ": piq})
        iq_z = ((iq_row - iq_mean) / iq_std).values.astype(np.float32)
        gru_input = torch.tensor(iq_z.reshape(1, 3, 1), dtype=torch.float32)
        with torch.no_grad():
            gru_prob = bigru_pheno(gru_input).item()

        # ── CNN (MRI slice, via the same preprocessing as training) ─
        suffix = ".nii.gz" if uploaded_mri.name.endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_mri.getbuffer())
            tmp_path = tmp.name
        try:
            single_df = pd.DataFrame([{"filepath": tmp_path, "label": 0}])
            mri_dataset = ABIDESliceDataset(single_df, augment=False)
            img_tensor, _ = mri_dataset[0]
            with torch.no_grad():
                cnn_prob = cnn_mri(img_tensor.unsqueeze(0)).item()
        finally:
            os.remove(tmp_path)

        # ── Late fusion (meta-classifier over [cnn, gru, dt] probs) ──
        meta_X = np.array([[cnn_prob, gru_prob, dt_prob]], dtype=np.float32)
        fusion_prob = meta_clf.predict_proba(meta_X)[0][1]

        prediction = "ASD Traits Detected" if fusion_prob > 0.5 else "No ASD Traits Detected"
        confidence = fusion_prob if fusion_prob > 0.5 else 1 - fusion_prob

        st.divider()
        st.caption("Prediction made using: **Late Fusion (CNN + Bi-GRU + Decision Tree)**")
        if fusion_prob > 0.5:
            st.error(f"## 🔴 {prediction}")
        else:
            st.success(f"## 🟢 {prediction}")

        st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        st.markdown("**All Model Outputs (for comparison):**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("CNN (MRI)", f"{cnn_prob * 100:.1f}%")
        mc2.metric("Bi-GRU (IQ sequence)", f"{gru_prob * 100:.1f}%")
        mc3.metric("Decision Tree (clinical)", f"{dt_prob * 100:.1f}%")
        mc4.metric("Late Fusion", f"{fusion_prob * 100:.1f}%")

        st.divider()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        names  = ["CNN\n(MRI)", "Bi-GRU\n(IQ seq)", "Decision\nTree", "Late\nFusion"]
        scores = [cnn_prob * 100, gru_prob * 100, dt_prob * 100, fusion_prob * 100]
        colors = ['#e74c3c' if s > 50 else '#2ecc71' for s in scores]
        ax.bar(names, scores, color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("ASD Probability (%)", color='white')
        ax.set_title("All Models — This Prediction (ABIDE)", color='white')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

        st.caption(
            "⚠️ Note: the CNN and Bi-GRU branches were trained on independently-drawn splits "
            "and reused here for inference only; the Decision Tree and late-fusion meta-classifier "
            "were trained on this exact canonical split. See report for full methodology and honesty notes."
        )


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
        models  = ["CNN", "Bi-GRU", "Decision\nTree", "Late\nFusion"]
        default_accs = {"CNN": 81.36, "Bi-GRU": 81.36, "Decision Tree": 95.97, "Late Fusion": 96.22}
        late_fusion_perf = late_fusion_stats.get("model_performance", {})
        accs = [
            late_fusion_perf.get("CNN", {}).get("accuracy", default_accs["CNN"]/100) * 100,
            late_fusion_perf.get("Bi-GRU", {}).get("accuracy", default_accs["Bi-GRU"]/100) * 100,
            late_fusion_perf.get("Decision Tree", {}).get("accuracy", default_accs["Decision Tree"]/100) * 100,
            late_fusion_perf.get("Late Fusion", {}).get("accuracy", default_accs["Late Fusion"]/100) * 100,
        ]
        colors  = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']
        bars    = ax.bar(models, accs, color=colors)
        ax.set_ylim(max(min(accs) - 5, 0), 105)
        ax.set_ylabel("Accuracy (%)", color='white')
        ax.set_title("Individual vs Late Fusion Accuracy", color='white')
        ax.tick_params(colors='white')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', color='white', fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("Late Fusion — Metrics Overview")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values  = [fusion_metrics["accuracy"]*100, fusion_metrics["precision"]*100,
                   fusion_metrics["recall"]*100, fusion_metrics["f1"]*100]
        ax2.bar(metrics, values, color='#4fc3f7')
        ax2.set_ylim(max(min(values) - 5, 0), 101)
        ax2.set_ylabel("Score (%)", color='white')
        ax2.set_title("Late Fusion Metrics", color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_visible(False)
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 Model Architecture Summary")
    st.markdown("""
    | Component | Input | Output | Purpose |
    |---|---|---|---|
    | **CNN** | A1–A10 as 2×5 heatmap | Probability | Visual pattern detection |
    | **Bi-GRU** | A1–A10 as sequence | Probability | Sequential behaviour analysis |
    | **Decision Tree** | All 26 features (pruned CART) | Probability | Tabular clinical data, interpretable rules |
    | **Late Fusion** | [CNN, Bi-GRU, DT] probabilities | ASD / No ASD | Logistic regression meta-classifier |
    """)

    st.info(
        "ℹ️ **Note on fusion strategy:** an earlier intermediate-fusion variant "
        "(concatenating internal feature embeddings from all three branches) achieved "
        "100% accuracy on the test set — a sign of overfitting caused by feature "
        "redundancy across branches (all three ultimately derive from the same A1–A10 "
        "screening answers). This system now reports the late-fusion logistic regression "
        "model instead, which combines model *probabilities* rather than raw features, "
        "and was validated with McNemar's and DeLong's statistical tests."
    )

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
    multimodal approach that combines three different models via late fusion, and lets
    the user compare each branch individually against the combined prediction:

    ### Models Used
    - **CNN (Convolutional Neural Network)** — Analyses behavioural screening scores
      (A1–A10) as a 2D heatmap image to detect visual patterns
    - **Bi-GRU (Bidirectional Gated Recurrent Unit)** — Processes the same A1–A10 scores
      as a temporal sequence to capture behavioural progression
    - **Decision Tree** — CART with cost-complexity pruning, analysing all 26 clinical
      and demographic features with a fully interpretable if-then rule pathway

    ### Fusion Strategy
    The probability outputs of all three models are combined through a **logistic
    regression late-fusion meta-classifier**, statistically validated against each
    single-modality baseline using McNemar's and DeLong's tests. Users can also select
    an individual model to see its standalone contribution.

    ### Dataset
    - **Source:** Kaggle Autistic Child Behavioural Dataset
    - **Samples:** 1,985 children
    - **Features:** 26 behavioural, clinical, and demographic features
    - **Target:** ASD Traits (Yes / No)

    ---

    ## 🧬 ABIDE MRI + Phenotypic System (separate cohort)

    A second, independently-trained system validated on the **ABIDE I** dataset
    (868 subjects with matched structural MRI, IQ, and clinical data):

    - **CNN (ResNet50)** — structural MRI axial slice
    - **Bi-GRU** — phenotypic IQ sequence (FIQ → VIQ → PIQ)
    - **Decision Tree** — clinical features (age, sex, FIQ, VIQ, PIQ)
    - **Late Fusion** — logistic-regression meta-classifier over the three probabilities

    This system is reported separately from the Kaggle system above because it uses
    a different cohort, different features, and reflects a genuine three-way fusion
    validated on the SAME subjects — required for methodologically sound fusion.

    ### Disclaimer
    > ⚠️ This tool is for **research and educational purposes only**.
    > It does **not** replace professional medical diagnosis.
    > Always consult a qualified healthcare professional for ASD assessment.
    """)