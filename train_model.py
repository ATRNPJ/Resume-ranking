import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "dataset/resume_dataset_cleaned (1).csv"
MODEL_FOLDER = "model"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "resume_model.pkl"
)


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# TARGET
# =========================================================

TARGET = "hired"


# =========================================================
# FEATURES
# =========================================================

numeric_features = [
    "age",
    "cgpa",
    "internships",
    "projects",
    "programming_languages",
    "certifications",
    "experience_years",
    "hackathons",
    "research_papers",
    "skills_score",
    "soft_skills_score",
    "resume_length_words"
]


categorical_features = [
    "education_level",
    "university_tier",
    "company_type"
]


all_features = (
    numeric_features
    + categorical_features
)


# =========================================================
# CHECK COLUMNS
# =========================================================

print("\nChecking required columns...")

missing_columns = [
    column
    for column in all_features + [TARGET]
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: These columns are missing:")

    for column in missing_columns:
        print("-", column)

    raise SystemExit


print("All required columns found.")


# =========================================================
# PREPARE X AND y
# =========================================================

X = df[all_features]

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# Remove rows where target could not be converted

valid_rows = y.notna()

X = X[valid_rows]

y = y[valid_rows]


print("\nTarget distribution:")

print(y.value_counts())


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]
)


# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(

    max_iter=1000,

    class_weight=None,

    random_state=42
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "classifier",

            model
        )

    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"Accuracy:            {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:   {balanced_accuracy:.4f}"
)

print(
    f"ROC-AUC:             {roc_auc:.4f}"
)


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "Location:",
    MODEL_PATH
)

print("\nTraining completed successfully!")