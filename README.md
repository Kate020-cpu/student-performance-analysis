# 🎀 Student Performance Analysis

An exploratory data analysis and modelling project on a **synthetic** student dataset, built to practise the full analyst workflow: inspect, clean, explore, test, model, conclude.

## ❓ Questions
1. Which factors relate most strongly to the final exam score?
2. Does study time still matter once previous performance is accounted for?
3. Do school type, internet access, or parental education relate to outcomes?
4. Can a simple model predict the final score?

## 📂 Project structure
```
├── data/
│   └── student_performance_raw.csv   # 1,215 rows, deliberately messy
├── notebooks/
│   └── 01_student_performance_analysis.ipynb
├── src/
│   ├── generate_data.py              # recreates the dataset (seeded)
│   └── build_notebook.py             # scaffold builder (optional)
├── images/                           # charts saved by the notebook
├── requirements.txt
└── README.md
```

## 🧹 About the data
Synthetic, with realistic relationships between variables. It includes missing values, duplicates, inconsistent text (`Female`/`female`, `Yes`/`Y`), and impossible values (24 study hours, 120% attendance, age 150) for cleaning practice.

| Column | Description |
|---|---|
| student_id | Unique ID |
| gender, age | Demographics |
| school_type | Urban / Rural / Suburban |
| parent_education | Highest parental education |
| internet_access, extracurricular | Yes / No |
| study_hours_per_day, sleep_hours, screen_time_hours | Daily habits |
| attendance_pct | Attendance percentage |
| previous_score, final_score | Exam scores (0-100) |

## 🚀 Run it
```bash
git clone https://github.com/Kate020-cpu/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/01_student_performance_analysis.ipynb
```

## 🔍 Key findings
_To be added after the analysis._

## 🛠️ Tools
Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn

