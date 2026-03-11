import pandas as pd
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier


# -----------------------------
# Helper function
# -----------------------------
def group_rare_categories(series, threshold=0.01):
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.replace(rare, "other")


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# -----------------------------
# Basic Cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df.drop_duplicates(inplace=True)
df.drop(columns=["customerID"], inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# -----------------------------
# Feature Engineering
# -----------------------------
df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
df["avg_charge_per_tenure"] = df["avg_charge_per_tenure"].clip(upper=1000)

df["family_flag"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
df["is_long_term"] = (df["tenure"] >= 24).astype(int)


# -----------------------------
# Fix Numerical Instability
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# clip extreme values
df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)


# -----------------------------
# Memory Optimization
# -----------------------------
df = df.astype({col: "category" for col in df.select_dtypes(include=["object"]).columns})
df = df.astype({col: "int16" for col in df.select_dtypes(include=["int64"]).columns})
df = df.astype({col: "float32" for col in df.select_dtypes(include=["float64"]).columns})


# -----------------------------
# Split Features & Target
# -----------------------------
X = df.drop(columns=["Churn"]).copy()
y = df["Churn"].copy()

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()


# -----------------------------
# Preprocessing Pipeline
# -----------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------
# Define Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    ),
}


# -----------------------------
# Train and Compare Models
# -----------------------------
results = []
best_model = None
best_auc = 0
best_model_name = ""

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)

    results.append(
        {
            "Model": name,
            "Accuracy": round(acc, 3),
            "ROC-AUC": round(auc, 3),
        }
    )

    if auc > best_auc:
        best_auc = auc
        best_model = pipeline
        best_model_name = name


# -----------------------------
# Model Comparison Table
# -----------------------------
comparison_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

print("\nMODEL COMPARISON\n")
print(comparison_df.to_string(index=False))


# -----------------------------
# Best Model
# -----------------------------
print("\nBest Model Selected:", best_model_name)
print("Best ROC-AUC:", round(best_auc, 3))


# -----------------------------
# Detailed Evaluation
# -----------------------------
y_pred = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, probs))


# -----------------------------
# Save Model + Feature Names
# -----------------------------
joblib.dump(best_model, "model/best_churn_model.joblib")

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
joblib.dump(feature_names, "model/feature_names.joblib")

print("\nBest model saved to model/best_churn_model.joblib")