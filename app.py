from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path
import os
import joblib
import gdown

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "greatj_best_model_mmap.joblib"

def ensure_model():
    # If already downloaded, skip
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:
        return

    url = os.getenv("MODEL_URL")
    if not url:
        raise FileNotFoundError("MODEL_URL environment variable not set in Render.")

    print("Downloading model from Google Drive using gdown...")
    gdown.download(url, str(MODEL_PATH), quiet=False)
    print("Model downloaded successfully.")

ensure_model()

# ✅ mmap works only with UNCOMPRESSED joblib (this file)
model = joblib.load(MODEL_PATH, mmap_mode="r")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    inputs = {}

    if request.method == "POST":
        try:
            inputs = {
                "person_age": float(request.form["person_age"]),
                "person_income": float(request.form["person_income"]),
                "person_home_ownership": request.form["person_home_ownership"],
                "person_emp_length": float(request.form["person_emp_length"]),
                "loan_intent": request.form["loan_intent"],
                "loan_grade": request.form["loan_grade"],
                "loan_amnt": float(request.form["loan_amnt"]),
                "loan_int_rate": float(request.form["loan_int_rate"]),
                "loan_percent_income": float(request.form["loan_percent_income"]),
                "cb_person_default_on_file": request.form["cb_person_default_on_file"],
                "cb_person_cred_hist_length": float(request.form["cb_person_cred_hist_length"]),
            }

            df = pd.DataFrame([inputs])
            prob = float(model.predict_proba(df)[:, 1][0])
            decision = "APPROVED" if prob >= 0.5 else "REJECTED"

            result = {"probability": prob, "decision": decision}

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, inputs=inputs)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

