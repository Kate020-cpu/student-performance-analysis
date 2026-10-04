import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 🎀 Student Performance Analysis

**Author:** Kate · **Dataset:** synthetic (generated with `src/generate_data.py`)

> ⚠️ The data is **synthetic**: it was generated for practice, so findings describe the simulated data, not real students.

## Questions I want to answer
1. Which factors are most strongly related to the final exam score?
2. Does study time still matter once previous performance is accounted for?
3. Do school type, internet access, or parental education relate to outcomes?
4. Can I build a simple model that predicts the final score?

## Notebook outline
1. Setup & loading
2. Data inspection
3. Data cleaning
4. Exploratory data analysis (EDA)
5. Hypothesis testing
6. Modelling
7. Conclusions
""")

md("## 1. Setup & loading")
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pretty pink theme 🌸
pink_palette = ["#FF69B4", "#FFB6D9", "#DDA0DD", "#F8A5C2", "#FF85B3", "#C06C9B"]
sns.set_theme(style="whitegrid", palette=pink_palette, font_scale=1.05)
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.facecolor"] = "#FFF8FB"

pd.set_option("display.max_columns", None)
""")
code("""
df_raw = pd.read_csv("../data/student_performance_raw.csv")
df = df_raw.copy()          # always keep the raw data untouched
df.head()
""")

md("## 2. Data inspection")
code("""
print("Shape:", df.shape)
df.info()
""")
code("""
df.describe().T
""")
code("""
# Missing values
missing = df.isna().sum().to_frame("missing")
missing["pct"] = (missing["missing"] / len(df) * 100).round(2)
missing[missing["missing"] > 0]
""")
code("""
# Duplicates & categorical consistency
print("Duplicate rows:", df.duplicated().sum())
for col in ["gender", "school_type", "parent_education", "internet_access", "extracurricular"]:
    print(f"\\n{col}:\\n", df[col].value_counts(dropna=False))
""")

md("""
### 🔎 Issues spotted
Write down what you find here before cleaning. Things to look for:
- missing values (which columns? how many?)
- duplicated rows
- inconsistent text (e.g. `Female` vs `female`, `Yes` vs `Y`)
- impossible values (e.g. 24 study hours, attendance above 100%, age 150)
""")

md("## 3. Data cleaning")
code("""
# 3.1 Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 3.2 Standardise text columns
df["gender"] = df["gender"].str.strip().str.title()
df["school_type"] = df["school_type"].str.strip().str.title()
df["internet_access"] = df["internet_access"].replace({"Y": "Yes", "N": "No"})

# 3.3 Handle impossible values -> set to NaN, then impute
df.loc[~df["age"].between(10, 25), "age"] = np.nan
df.loc[df["study_hours_per_day"] > 12, "study_hours_per_day"] = np.nan
df.loc[df["attendance_pct"] > 100, "attendance_pct"] = np.nan
""")
code("""
# 3.4 Impute missing values
# TODO: Justify your choices! Median is robust to outliers; mode suits categories.
num_cols = ["age", "study_hours_per_day", "attendance_pct", "sleep_hours"]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

df["parent_education"] = df["parent_education"].fillna(df["parent_education"].mode()[0])

print("Remaining missing values:", df.isna().sum().sum())
print("Rows after cleaning:", len(df))
""")
code("""
# 3.5 Feature engineering
df["score_change"] = df["final_score"] - df["previous_score"]
df["study_group"] = pd.cut(
    df["study_hours_per_day"],
    bins=[0, 1, 2, 3, 4, 12],
    labels=["<1h", "1-2h", "2-3h", "3-4h", "4h+"],
    include_lowest=True,
)
df["passed"] = (df["final_score"] >= 50).astype(int)

df.to_csv("../data/student_performance_clean.csv", index=False)
df.head()
""")

md("## 4. Exploratory data analysis")
md("### 4.1 Distributions")
code("""
num_features = ["study_hours_per_day", "attendance_pct", "sleep_hours",
                "screen_time_hours", "previous_score", "final_score"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flat, num_features):
    sns.histplot(df[col], kde=True, ax=ax, color="#FF69B4")
    ax.set_title(col.replace("_", " ").title())
plt.tight_layout()
plt.savefig("../images/distributions.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("### 4.2 Correlations")
code("""
corr = df[num_features + ["score_change"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", linewidths=0.5)
plt.title("Correlation matrix")
plt.savefig("../images/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""")
md("**📝 Observation:** _What are the strongest relationships with `final_score`? Write it in your own words._")

md("### 4.3 Study time vs. final score")
code("""
plt.figure()
sns.scatterplot(data=df, x="study_hours_per_day", y="final_score", alpha=0.5, color="#FF69B4")
sns.regplot(data=df, x="study_hours_per_day", y="final_score", scatter=False, color="#C06C9B")
plt.title("Study hours vs final score")
plt.savefig("../images/study_vs_score.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("### 4.4 Group comparisons")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["school_type", "parent_education", "internet_access"]):
    sns.boxplot(data=df, x=col, y="final_score", ax=ax, palette=pink_palette)
    ax.set_title(f"Final score by {col.replace('_', ' ')}")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
""")
code("""
# Summary table
(df.groupby("school_type")["final_score"]
   .agg(["count", "mean", "median", "std"])
   .round(2))
""")

md("""
### 4.5 Your turn ✍️
Add your own exploration below. Ideas:
- Average `final_score` by `study_group`: does the effect plateau?
- Compare `score_change` for students with vs. without extracurriculars
- Is the effect of study hours different for rural vs. urban schools?
- Pass rate by sleep-hour bucket
""")
code("""
# TODO: your own analysis here
""")

md("## 5. Hypothesis testing")
md("""
**H₀:** there is no difference in mean final score between students **with** and **without** internet access.  
**H₁:** there is a difference.  
Significance level α = 0.05.
""")
code("""
from scipy import stats

yes = df.loc[df["internet_access"] == "Yes", "final_score"]
no = df.loc[df["internet_access"] == "No", "final_score"]

t, p = stats.ttest_ind(yes, no, equal_var=False)
print(f"Mean (Yes): {yes.mean():.2f} | Mean (No): {no.mean():.2f}")
print(f"t = {t:.3f}, p = {p:.4f}")
print("Reject H0" if p < 0.05 else "Fail to reject H0")
""")
code("""
# TODO: try a one-way ANOVA across school types (stats.f_oneway)
#       and a chi-square test between parent_education and passed
""")

md("## 6. Modelling")
md("A simple baseline: **linear regression** predicting `final_score`.")
code("""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

features = ["study_hours_per_day", "attendance_pct", "sleep_hours",
            "screen_time_hours", "previous_score", "age"]
X = pd.get_dummies(
    df[features + ["school_type", "parent_education", "internet_access", "extracurricular"]],
    drop_first=True,
)
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print(f"R²  : {r2_score(y_test, pred):.3f}")
print(f"MAE : {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
""")
code("""
coefs = pd.Series(model.coef_, index=X.columns).sort_values()
coefs.plot(kind="barh", figsize=(8, 6), color="#FF69B4")
plt.title("Linear regression coefficients")
plt.tight_layout()
plt.savefig("../images/coefficients.png", dpi=150, bbox_inches="tight")
plt.show()
""")
code("""
# TODO: try RandomForestRegressor and compare against this baseline
""")

md("""
## 7. Conclusions
_Fill this in after your analysis._

**Key findings**
1. …
2. …
3. …

**Limitations**
- The data is synthetic, so real-world conclusions can't be drawn.
- Correlation ≠ causation.

**Next steps**
- Try tree-based models and cross-validation
- Build a dashboard (Power BI / Tableau / Streamlit)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/01_student_performance_analysis.ipynb")
