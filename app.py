
from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import joblib
import os
import glob
import math

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_FOLDER = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model" / "resume_model.pkl"
MINIMUM_AGE = 18

NUMERIC_FEATURES = [
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
    "resume_length_words",
]

CATEGORICAL_FEATURES = [
    "education_level",
    "university_tier",
    "company_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset():
    """Load the first CSV found in the configured dataset folder."""
    csv_files = sorted(glob.glob(str(DATASET_FOLDER / "*.csv")))

    if not csv_files:
        return None

    try:
        return pd.read_csv(csv_files[0])
    except Exception as exc:
        app.logger.exception("Dataset loading error: %s", exc)
        return None


def load_model():
    """Load the trained scikit-learn pipeline if available."""
    if not MODEL_PATH.is_file():
        app.logger.warning("Trained model not found: %s", MODEL_PATH)
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        app.logger.exception("Model loading error: %s", exc)
        return None


model = load_model()


def as_number(value, label, minimum=0, maximum=None):
    """Validate and convert a numeric form value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Please enter a valid number for {label}.")

    if not math.isfinite(number):
        raise ValueError(f"Please enter a valid number for {label}.")

    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")

    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must not exceed {maximum}.")

    return number


def university_tier_from_score(score):
    """
    Approximate a tier from the form's numeric university score.
    Confirm these thresholds against the dataset's actual encoding.
    """
    if score >= 3:
        return "Tier 1"
    if score >= 2:
        return "Tier 2"
    if score >= 1:
        return "Tier 3"
    return "Tier 3"


def build_candidate():
    """Convert screening form fields into the model's feature schema."""
    age = as_number(request.form.get("age"), "Age", 18, 100)
    cgpa = as_number(request.form.get("cgpa"), "CGPA", 0, 10)

    experience = as_number(
        request.form.get("experience", "0"), "Experience"
    )
    internships = as_number(
        request.form.get("internships", "0"), "Internships"
    )
    projects = as_number(request.form.get("projects", "0"), "Projects")
    hackathons = as_number(
        request.form.get("hackathons", "0"), "Hackathons"
    )
    research = as_number(request.form.get("research", "0"), "Research")
    languages = as_number(
        request.form.get("languages", "0"), "Programming languages"
    )
    certifications = as_number(
        request.form.get("certifications", "0"), "Certifications"
    )
    skills = as_number(
        request.form.get("skills_score", "0"), "Technical skills score", 0, 100
    )
    soft_skills = as_number(
        request.form.get("soft_skills", "0"), "Soft skills score", 0, 100
    )
    resume_pages = as_number(
        request.form.get("resume_length", "1"), "Resume length"
    )
    university_score = as_number(
        request.form.get("university", "0"), "University score"
    )

    education = request.form.get("education", "").strip()
    company_type = request.form.get("company_type", "").strip()

    if not education:
        raise ValueError("Please select an education level.")

    # The model expects university_tier, not a numeric university score.
    # This conversion is only an approximation; verify your dataset values.
    university_tier = university_tier_from_score(university_score)

    # The training feature is measured in words, while the form asks for pages.
    # This estimate assumes approximately 500 words per page.
    resume_words = resume_pages * 500

    features = {
        "age": age,
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "programming_languages": languages,
        "certifications": certifications,
        "experience_years": experience,
        "hackathons": hackathons,
        "research_papers": research,
        "skills_score": skills,
        "soft_skills_score": soft_skills,
        "resume_length_words": resume_words,
        "education_level": education,
        "university_tier": university_tier,
        "company_type": company_type or "Unknown",
    }

    candidate = {
        "age": int(age),
        "education": education,
        "university": university_tier,
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "hackathons": hackathons,
        "research": research,
        "programming_languages": languages,
        "certifications": certifications,
        "skills_score": skills,
        "soft_skills_score": soft_skills,
        "experience": experience,
        "resume_length": resume_pages,
        "company_type": company_type or "Not specified",
    }

    return pd.DataFrame([features], columns=ALL_FEATURES), candidate


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    df = load_dataset()

    if df is None:
        return render_template(
            "dashboard.html",
            error="Dataset could not be loaded.",
            total_candidates=0,
            hired_count=0,
            not_hired_count=0,
            hiring_rate=0,
            column_count=0,
            missing_values=0,
            duplicate_rows=0,
            feature_data=[],
            preview_columns=[],
            preview=[],
        )

    total_candidates = len(df)
    column_count = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    possible_columns = [
        "hired", "Hired", "hire", "Hire", "hired_status",
        "hiring_status", "is_hired", "Is_Hired",
    ]
    hired_column = next(
        (col for col in possible_columns if col in df.columns), None
    )

    hired_count = 0
    not_hired_count = total_candidates
    hiring_rate = 0

    if hired_column:
        values = df[hired_column]

        if pd.api.types.is_numeric_dtype(values):
            numeric_values = pd.to_numeric(values, errors="coerce")
            hired_count = int((numeric_values == 1).sum())
        else:
            hired_count = int(
                values.astype(str).str.strip().str.lower().isin(
                    ["1", "yes", "true", "hired", "selected"]
                ).sum()
            )

        not_hired_count = total_candidates - hired_count

        if total_candidates:
            hiring_rate = round(hired_count / total_candidates * 100, 1)

    feature_data = []

    if hired_column:
        target = pd.to_numeric(df[hired_column], errors="coerce")

        for column in df.select_dtypes(include="number").columns:
            if column == hired_column:
                continue

            try:
                correlation = df[column].corr(target)
                if pd.notna(correlation):
                    feature_data.append({
                        "name": column,
                        "correlation": round(abs(float(correlation)), 3),
                    })
            except (TypeError, ValueError):
                continue

    feature_data.sort(
        key=lambda item: item["correlation"], reverse=True
    )

    preview_columns = list(df.columns[:8])
    preview = df[preview_columns].head(8).fillna("").to_dict(
        orient="records"
    )

    return render_template(
        "dashboard.html",
        total_candidates=total_candidates,
        hired_count=hired_count,
        not_hired_count=not_hired_count,
        hiring_rate=hiring_rate,
        column_count=column_count,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        feature_data=feature_data[:8],
        preview_columns=preview_columns,
        preview=preview,
    )


@app.route("/dataset")
def dataset():
    df = load_dataset()

    if df is None:
        return render_template(
            "dataset.html",
            error="No CSV dataset was found.",
            columns=[],
            rows=[],
            total_rows=0,
            total_columns=0,
        )

    columns = list(df.columns)
    rows = df.head(100).fillna("").to_dict(orient="records")

    return render_template(
        "dataset.html",
        columns=columns,
        rows=rows,
        total_rows=len(df),
        total_columns=len(df.columns),
    )


@app.route("/screening", methods=["GET", "POST"])
def screening():
    if request.method == "GET":
        return render_template("screening.html")

    if model is None:
        return render_template(
            "screening.html",
            error=(
                "The trained model is unavailable. "
                "Please train the model and restart the application."
            ),
        ), 503

    try:
        features, candidate = build_candidate()

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        # Assumes the target classes are 0 and 1.
        positive_index = next(
            (i for i, value in enumerate(classes) if str(value) == "1"),
            None,
        )

        if positive_index is None:
            raise ValueError(
                "The trained model does not have a recognizable positive "
                "class labelled 1. Check the target labels in the dataset."
            )

        probability = float(probabilities[positive_index])
        score = round(probability * 100)

        if str(prediction) == "1":
            recommendation = "Model predicts hired"
        else:
            recommendation = "Model predicts not hired"

        insights = [
            (
                f"The model estimated a {score}% probability for the "
                "positive class in its training labels."
            ),
            (
                "This estimate is experimental: validation showed weak "
                "predictive performance. It must not be used as the sole "
                "basis for a hiring decision."
            ),
        ]

        return render_template(
            "results.html",
            score=score,
            recommendation=recommendation,
            candidate=candidate,
            insights=insights,
        )

    except ValueError as exc:
        return render_template(
            "screening.html",
            error=str(exc),
        ), 400
    except Exception:
        app.logger.exception("Candidate prediction failed")
        return render_template(
            "screening.html",
            error=(
                "Prediction failed. Check that the form values and model "
                "feature categories match the training dataset."
            ),
        ), 500


@app.route("/results")
def results():
    return render_template(
        "results.html",
        score=0,
        recommendation="No candidate analyzed yet.",
        candidate=None,
        insights=[],
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")