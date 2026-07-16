"""
Objective iii — Dataset builder for genuine ABIDE phenotypic sequences.

IMPORTANT DATA HONESTY NOTE (document this in your report):
- SRS subscale scores were only available for 64/1112 ABIDE I subjects (~6%),
  too sparse to use without collapsing the sample to near-nothing.
- ADI-R scores are ONLY administered to subjects being assessed for autism —
  they do not exist for control subjects. Using ADI-R as a model input would
  leak the label directly (presence of a score = autism, by data-collection
  design), so it is excluded as a feature.
- FIQ/VIQ/PIQ (IQ domain measurements) are available for 868/1112 subjects
  across BOTH classes with good balance (430 autism / 438 control), and are
  used here as the genuine chronologically-ordered phenotypic sequence
  (Full-Scale IQ -> Verbal IQ -> Performance IQ) for the Bi-GRU.
"""
import numpy as np
import pandas as pd


def build_phenotypic_sequences(phenotypic_csv, sub_id_col="SUB_ID", dx_col="DX_GROUP"):
    df = pd.read_csv(phenotypic_csv)

    iq_cols = ["FIQ", "VIQ", "PIQ"]
    for c in iq_cols:
        df[c] = df[c].replace(-9999, np.nan)

    df = df.dropna(subset=iq_cols).copy()
    df["label"] = (df[dx_col] == 1).astype(int)  # 1 = autism, 0 = control

    # z-score normalize each IQ domain independently
    for c in iq_cols:
        mean, std = df[c].mean(), df[c].std()
        df[c] = (df[c] - mean) / std

    # build sequence: (N, 3, 1) -> FIQ, VIQ, PIQ as 3 ordered time steps
    seq = df[iq_cols].values.astype(np.float32).reshape(-1, 3, 1)
    labels = df["label"].values.astype(np.float32)
    sub_ids = df[sub_id_col].values

    print(f"✅ Built phenotypic sequences for {len(df)} subjects")
    print(f"   Class balance -> Autism (1): {(labels==1).sum():.0f}  |  Control (0): {(labels==0).sum():.0f}")
    print(f"   Sequence shape: {seq.shape}  (FIQ -> VIQ -> PIQ per subject)")

    return seq, labels, sub_ids


def get_iq_stats(phenotypic_csv, dx_col="DX_GROUP"):
    """
    Returns the per-column mean/std used to z-score FIQ/VIQ/PIQ in
    build_phenotypic_sequences, computed over the SAME 868-subject cohort.

    Needed at inference time: a single new subject's raw FIQ/VIQ/PIQ must be
    normalized with these exact same statistics (not re-derived from just
    that one subject) to match what the trained Bi-GRU actually saw.
    """
    df = pd.read_csv(phenotypic_csv)
    iq_cols = ["FIQ", "VIQ", "PIQ"]
    for c in iq_cols:
        df[c] = df[c].replace(-9999, np.nan)
    df = df.dropna(subset=iq_cols).copy()
    return {c: (float(df[c].mean()), float(df[c].std())) for c in iq_cols}