import pandas as pd

# ==============================
# Paths
# ==============================
INPUT_CSV = "msl_medical_test_kaggle_fixed.csv"
OUTPUT_CSV = "msl_medical_test_remapped.csv"

# ==========================================================
# Official MSL Label ID  --->  Your MedicalSignFormer Label ID
# ==========================================================
LABEL_MAPPING = {

    # ---------- Exact Matches ----------
    22: 0,    # Asthma -> ASTHMA
    23: 13,   # Cough -> Cough
    24: 43,   # Swollen -> Swell
    25: 29,   # Itchy -> Itch
    26: 46,   # Vomit -> Vomit
    27: 20,   # Flu -> FLU
    28: 22,   # Fever -> Fever
    29: 23,   # Headache -> HEAD ACHE
    32: 37,   # Sore Throat -> SORE THROAT

    # ---------- Approximate Matches ----------
    1: 36,    # Shoulder -> SHOULDER PAIN
    3: 41,    # Fingers -> Sore Finger
    5: 34,    # Skin -> Rash
    9: 45,    # Teeth -> Tooth ache
    15: 3,    # Bones -> BROKEN BONES
    18: 6,    # Blood -> Bleed
    19: 24,   # Heart -> HEART ATTACK
    20: 28,   # Pain -> Injury
    30: 39,   # Stomach -> STOMACH ACHE
    31: 5,    # Waist Pain -> Back Ache
}

# ==============================
# Read CSV
# ==============================
df = pd.read_csv(INPUT_CSV, header=None)
df.columns = ["filename", "label"]

print(f"Original samples : {len(df)}")

# Keep only mapped labels
df = df[df["label"].isin(LABEL_MAPPING.keys())].copy()

print(f"Mapped samples   : {len(df)}")

# Replace labels
df["label"] = df["label"].map(LABEL_MAPPING)

# Save
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)

print("\nClass Distribution")
print(df["label"].value_counts().sort_index())