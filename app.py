from flask import Flask, render_template, request
import pandas as pd
import os
import glob


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_FOLDER = "dataset"
MINIMUM_AGE = 18


# =========================================================
# DATASET LOADING
# =========================================================

def load_dataset():

    csv_files = glob.glob(
        os.path.join(
            DATASET_FOLDER,
            "*.csv"
        )
    )

    if not csv_files:
        return None

    try:
        return pd.read_csv(csv_files[0])

    except Exception as e:

        print("Dataset loading error:", e)

        return None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

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
            preview=[]
        )


    total_candidates = len(df)

    column_count = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # -----------------------------------------------------
    # FIND HIRING COLUMN
    # -----------------------------------------------------

    hired_column = None

    possible_columns = [
        "hired",
        "Hired",
        "hire",
        "Hire",
        "hired_status",
        "hiring_status",
        "is_hired",
        "Is_Hired"
    ]


    for column in possible_columns:

        if column in df.columns:

            hired_column = column

            break


    # -----------------------------------------------------
    # HIRING STATISTICS
    # -----------------------------------------------------

    hired_count = 0

    not_hired_count = 0

    hiring_rate = 0


    if hired_column:

        values = df[hired_column]


        if values.dtype == "object":

            hired_count = values.astype(
                str
            ).str.lower().isin(
                [
                    "1",
                    "yes",
                    "true",
                    "hired",
                    "selected"
                ]
            ).sum()

        else:

            numeric_values = pd.to_numeric(
                values,
                errors="coerce"
            )

            hired_count = (
                numeric_values == 1
            ).sum()


        not_hired_count = (
            total_candidates - hired_count
        )


        if total_candidates > 0:

            hiring_rate = round(
                (
                    hired_count
                    / total_candidates
                ) * 100,
                1
            )


    # -----------------------------------------------------
    # FEATURE CORRELATION
    # -----------------------------------------------------

    feature_data = []


    if hired_column:

        target = pd.to_numeric(
            df[hired_column],
            errors="coerce"
        )


        numeric_columns = df.select_dtypes(
            include="number"
        ).columns


        for column in numeric_columns:

            if column == hired_column:
                continue


            try:

                correlation = df[column].corr(
                    target
                )


                if pd.notna(correlation):

                    feature_data.append(
                        {
                            "name": column,
                            "correlation": round(
                                abs(correlation),
                                3
                            )
                        }
                    )

            except Exception:

                pass


    feature_data.sort(
        key=lambda x: x["correlation"],
        reverse=True
    )

    feature_data = feature_data[:8]


    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    preview_columns = list(
        df.columns[:8]
    )


    preview = df[
        preview_columns
    ].head(8).to_dict(
        orient="records"
    )


    return render_template(

        "dashboard.html",

        total_candidates=total_candidates,

        hired_count=int(hired_count),

        not_hired_count=int(not_hired_count),

        hiring_rate=hiring_rate,

        column_count=column_count,

        missing_values=missing_values,

        duplicate_rows=duplicate_rows,

        feature_data=feature_data,

        preview_columns=preview_columns,

        preview=preview

    )


# =========================================================
# DATASET PAGE
# =========================================================

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
            total_columns=0
        )


    columns = list(df.columns)

    rows = df.head(100).fillna("").to_dict(
        orient="records"
    )


    return render_template(

        "dataset.html",

        columns=columns,

        rows=rows,

        total_rows=len(df),

        total_columns=len(df.columns)

    )


# =========================================================
# SCREENING
# =========================================================

@app.route(
    "/screening",
    methods=["GET", "POST"]
)
def screening():

    if request.method == "POST":

        return calculate_result()


    return render_template(
        "screening.html"
    )


# =========================================================
# CALCULATE CANDIDATE SCORE
# =========================================================

def calculate_result():

    # -----------------------------------------------------
    # GET FORM VALUES
    # -----------------------------------------------------

    age = request.form.get(
        "age",
        ""
    )

    education = request.form.get(
        "education",
        ""
    )

    university = request.form.get(
        "university",
        ""
    )

    cgpa = request.form.get(
        "cgpa",
        "0"
    )

    internships = request.form.get(
        "internships",
        "0"
    )

    projects = request.form.get(
        "projects",
        "0"
    )

    hackathons = request.form.get(
        "hackathons",
        "0"
    )

    research = request.form.get(
        "research",
        "0"
    )

    languages = request.form.get(
        "languages",
        "0"
    )

    certifications = request.form.get(
        "certifications",
        "0"
    )

    skills_score = request.form.get(
        "skills_score",
        "0"
    )

    soft_skills = request.form.get(
        "soft_skills",
        "0"
    )

    experience = request.form.get(
        "experience",
        "0"
    )

    resume_length = request.form.get(
        "resume_length",
        "0"
    )

    company_type = request.form.get(
        "company_type",
        ""
    )


    # -----------------------------------------------------
    # VALIDATE AGE
    # -----------------------------------------------------

    try:

        age_value = float(age)

    except ValueError:

        return render_template(
            "screening.html",
            error="Please enter a valid age."
        )


    if age_value < MINIMUM_AGE:

        return render_template(
            "screening.html",
            error="Candidates must be at least 18 years old."
        )


    if age_value > 100:

        return render_template(
            "screening.html",
            error="Please enter a valid age."
        )


    # -----------------------------------------------------
    # CONVERT NUMERIC VALUES
    # -----------------------------------------------------

    try:

        cgpa_value = float(cgpa)

        internships_value = float(
            internships
        )

        projects_value = float(
            projects
        )

        hackathons_value = float(
            hackathons
        )

        research_value = float(
            research
        )

        languages_value = float(
            languages
        )

        certifications_value = float(
            certifications
        )

        skills_value = float(
            skills_score
        )

        soft_skills_value = float(
            soft_skills
        )

        experience_value = float(
            experience
        )

        resume_length_value = float(
            resume_length
        )

        university_value = float(
            university
        )

    except ValueError:

        return render_template(
            "screening.html",
            error="Please enter valid numerical values."
        )


    # -----------------------------------------------------
    # LIMIT VALUES
    # -----------------------------------------------------

    cgpa_value = max(
        0,
        min(cgpa_value, 10)
    )

    internships_value = max(
        0,
        internships_value
    )

    projects_value = max(
        0,
        projects_value
    )

    hackathons_value = max(
        0,
        hackathons_value
    )

    research_value = max(
        0,
        research_value
    )

    languages_value = max(
        0,
        languages_value
    )

    certifications_value = max(
        0,
        certifications_value
    )

    skills_value = max(
        0,
        min(skills_value, 100)
    )

    soft_skills_value = max(
        0,
        min(soft_skills_value, 100)
    )

    experience_value = max(
        0,
        experience_value
    )

    resume_length_value = max(
        0,
        resume_length_value
    )


    # =====================================================
    # SUITABILITY SCORE
    # =====================================================

    # Experience - 30%
    experience_score = min(
        experience_value / 5,
        1
    ) * 30


    # Skills - 25%
    skills_component = (
        skills_value / 100
    ) * 25


    # Projects - 15%
    projects_score = min(
        projects_value / 5,
        1
    ) * 15


    # Internships - 10%
    internship_score = min(
        internships_value / 3,
        1
    ) * 10


    # CGPA - 8%
    cgpa_score = (
        cgpa_value / 10
    ) * 8


    # Soft skills - 5%
    soft_skill_score = (
        soft_skills_value / 100
    ) * 5


    # Certifications - 3%
    certification_score = min(
        certifications_value / 3,
        1
    ) * 3


    # Hackathons - 2%
    hackathon_score = min(
        hackathons_value / 3,
        1
    ) * 2


    # Research - 2%
    research_score = min(
        research_value / 2,
        1
    ) * 2


    # -----------------------------------------------------
    # TOTAL SCORE
    # -----------------------------------------------------

    score = (

        experience_score
        + skills_component
        + projects_score
        + internship_score
        + cgpa_score
        + soft_skill_score
        + certification_score
        + hackathon_score
        + research_score

    )


    score = round(
        min(
            max(score, 0),
            100
        )
    )


    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    if score >= 75:

        recommendation = (
            "Strong Candidate"
        )

    elif score >= 60:

        recommendation = (
            "Promising Candidate"
        )

    elif score >= 45:

        recommendation = (
            "Moderate Candidate"
        )

    else:

        recommendation = (
            "Needs Further Review"
        )


    # -----------------------------------------------------
    # INSIGHTS
    # -----------------------------------------------------

    insights = []


    if skills_value >= 75:

        insights.append(
            "The candidate demonstrates strong technical skills."
        )

    elif skills_value >= 50:

        insights.append(
            "The candidate has a moderate technical skill profile."
        )

    else:

        insights.append(
            "Technical skills could be strengthened."
        )


    if experience_value >= 3:

        insights.append(
            "Relevant experience contributes positively to the candidate score."
        )

    elif experience_value > 0:

        insights.append(
            "The candidate has some practical experience."
        )

    else:

        insights.append(
            "Additional practical experience may improve the profile."
        )


    if projects_value >= 3:

        insights.append(
            "The candidate has a good project portfolio."
        )

    elif projects_value > 0:

        insights.append(
            "The candidate has project experience."
        )

    else:

        insights.append(
            "Adding practical projects could strengthen the profile."
        )


    if internships_value >= 2:

        insights.append(
            "Internship experience provides useful industry exposure."
        )


    if cgpa_value >= 8:

        insights.append(
            "The academic performance is strong."
        )

    elif cgpa_value < 6:

        insights.append(
            "Academic performance may require additional review."
        )


    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------

    candidate = {

        "age": int(age_value),

        "education": education,

        "university": university,

        "cgpa": cgpa_value,

        "internships": internships_value,

        "projects": projects_value,

        "hackathons": hackathons_value,

        "research": research_value,

        "programming_languages": languages_value,

        "certifications": certifications_value,

        "skills_score": skills_value,

        "soft_skills_score": soft_skills_value,

        "experience": experience_value,

        "resume_length": resume_length_value,

        "company_type": company_type

    }


    return render_template(

        "results.html",

        score=score,

        recommendation=recommendation,

        candidate=candidate,

        insights=insights

    )


# =========================================================
# RESULTS
# =========================================================

@app.route("/results")
def results():

    return render_template(

        "results.html",

        score=0,

        recommendation="No candidate analyzed yet.",

        candidate=None,

        insights=[]

    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )