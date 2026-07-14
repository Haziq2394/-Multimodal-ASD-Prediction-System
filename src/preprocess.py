import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Load Dataset ──────────────────────────────────────────────
def load_data(path="data/Kaggle Autistic Child Behavioural Dataset.csv"):
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ── Clean Data ────────────────────────────────────────────────
def clean_data(df):
    # Drop case number column
    df = df.drop(columns=["CASE_NO_PATIENT'S"], errors='ignore')

    # Fill missing values
    df['Social_Responsiveness_Scale'] = df['Social_Responsiveness_Scale'].fillna(df['Social_Responsiveness_Scale'].median())
    df['Qchat_10_Score'] = df['Qchat_10_Score'].fillna(df['Qchat_10_Score'].median())
    df['Social/Behavioural Issues'] = df['Social/Behavioural Issues'].fillna('No')
    df['Depression'] = df['Depression'].fillna('No')

    # Encode target label
    df['ASD_traits'] = df['ASD_traits'].map({'Yes': 1, 'No': 0})

    # Encode all remaining categorical columns — keep the fitted encoders
    # so the Streamlit app can reuse the SAME encoding instead of
    # refitting on a single row (which would always map to 0).
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print("✅ Data cleaned and encoded")
    return df, encoders

# ── Prepare Three Inputs ──────────────────────────────────────
def prepare_inputs(df):
    feature_cols = [c for c in df.columns if c != 'ASD_traits']
    X = df[feature_cols].values
    y = df['ASD_traits'].values

    # Behavioral sequence columns A1-A10
    seq_cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10_Autism_Spectrum_Quotient']
    seq_data = df[seq_cols].values

    # CNN input: reshape A1-A10 as 2D heatmap (N, 2, 5, 1)
    cnn_data = seq_data.reshape(-1, 2, 5, 1).astype(np.float32)

    # Bi-GRU input: (N, 10, 1) — 10 time steps, 1 feature each
    gru_data = seq_data.reshape(-1, 10, 1).astype(np.float32)

    # Decision Tree input: all features
    scaler = StandardScaler()
    dt_data = scaler.fit_transform(X).astype(np.float32)

    print(f"✅ CNN input shape:    {cnn_data.shape}")
    print(f"✅ Bi-GRU input shape: {gru_data.shape}")
    print(f"✅ DT input shape:     {dt_data.shape}")

    return cnn_data, gru_data, dt_data, y, scaler

# ── Train/Test Split ──────────────────────────────────────────
def split_data(cnn_data, gru_data, dt_data, y, test_size=0.2):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                            random_state=42, stratify=y)
    return (cnn_data[train_idx], cnn_data[test_idx],
            gru_data[train_idx], gru_data[test_idx],
            dt_data[train_idx],  dt_data[test_idx],
            y[train_idx],        y[test_idx])

# ── Run as standalone test ────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df, encoders = clean_data(df)
    cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)
    splits = split_data(cnn_data, gru_data, dt_data, y)
    print("✅ Preprocessing complete!")
    print(f"   Train size: {len(splits[6])}, Test size: {len(splits[7])}")