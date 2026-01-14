from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("greatj_best_model.pickle", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    inputs = {}

    if request.method == "POST":
        try:
            # Read inputs
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

            result = {
                "probability": prob,
                "decision": decision
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)
