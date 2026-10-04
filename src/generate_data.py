"""Generate a synthetic student performance dataset (with deliberate messiness).

Run:  python src/generate_data.py
Output: data/student_performance_raw.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

# --- Demographics -----------------------------------------------------------
gender = rng.choice(["Female", "Male"], N, p=[0.52, 0.48])
age = rng.choice([15, 16, 17, 18], N, p=[0.2, 0.3, 0.3, 0.2])
school = rng.choice(["Urban High", "Rural High", "Suburban High"], N, p=[0.45, 0.25, 0.30])
parent_education = rng.choice(
    ["High School", "Diploma", "Bachelor's", "Master's"], N, p=[0.35, 0.25, 0.28, 0.12]
)
internet_access = rng.choice(["Yes", "No"], N, p=[0.82, 0.18])
extracurricular = rng.choice(["Yes", "No"], N, p=[0.55, 0.45])

# --- Behaviour (correlated with background) ---------------------------------
study_hours = np.clip(rng.normal(2.5, 1.2, N) + (internet_access == "Yes") * 0.3, 0, 8)
attendance = np.clip(rng.normal(88, 8, N) - (school == "Rural High") * 3, 50, 100)
sleep_hours = np.clip(rng.normal(7, 1.1, N), 4, 10)
screen_time = np.clip(rng.normal(4, 1.8, N) - study_hours * 0.2, 0, 12)
prev_score = np.clip(rng.normal(62, 14, N), 20, 100)

edu_boost = pd.Series(parent_education).map(
    {"High School": 0, "Diploma": 1.5, "Bachelor's": 3, "Master's": 4.5}
).to_numpy()

# --- Outcome: final exam score ---------------------------------------------
final_score = (
    8
    + 0.55 * prev_score
    + 4.0 * study_hours
    + 0.18 * attendance
    + 1.2 * (sleep_hours - 7)
    - 0.9 * screen_time
    + edu_boost
    + 2.0 * (extracurricular == "Yes")
    + rng.normal(0, 5, N)
)
final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame(
    {
        "student_id": [f"STU{str(i).zfill(4)}" for i in range(1, N + 1)],
        "gender": gender,
        "age": age,
        "school_type": school,
        "parent_education": parent_education,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "study_hours_per_day": study_hours.round(1),
        "attendance_pct": attendance.round(1),
        "sleep_hours": sleep_hours.round(1),
        "screen_time_hours": screen_time.round(1),
        "previous_score": prev_score.round(0),
        "final_score": final_score.round(0),
    }
)

# --- Add realistic mess so there is cleaning practice -----------------------
# 1. Missing values
for col, frac in [("study_hours_per_day", 0.04), ("sleep_hours", 0.03),
                  ("attendance_pct", 0.02), ("parent_education", 0.03)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan

# 2. Inconsistent text formatting
idx = rng.choice(N, 40, replace=False)
df.loc[idx, "gender"] = df.loc[idx, "gender"].str.lower()
idx = rng.choice(N, 30, replace=False)
df.loc[idx, "school_type"] = df.loc[idx, "school_type"].str.upper()
idx = rng.choice(N, 25, replace=False)
df.loc[idx, "internet_access"] = df.loc[idx, "internet_access"].replace({"Yes": "Y", "No": "N"})

# 3. Impossible values / outliers
df.loc[rng.choice(N, 5, replace=False), "study_hours_per_day"] = 24
df.loc[rng.choice(N, 4, replace=False), "attendance_pct"] = 120
df.loc[rng.choice(N, 3, replace=False), "age"] = 150

# 4. Duplicate rows
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("data/student_performance_raw.csv", index=False)
print(df.shape)
