from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():

    data = {

        "age": int(request.form["age"]),
        "job": request.form["job"],
        "marital": request.form["marital"],
        "education": request.form["education"],
        "default": "no",
        "balance": int(request.form["balance"]),
        "housing": request.form["housing"],
        "loan": request.form["loan"],
        "contact": request.form["contact"],
        "day": int(request.form["day"]),
        "month": request.form["month"],
        "campaign": int(request.form["campaign"]),
        "pdays": int(request.form["pdays"]),
        "previous": int(request.form["previous"]),
        "poutcome": request.form["poutcome"]

    }

    df = pd.DataFrame([data])

    pipeline = PredictPipeline()

    prediction = pipeline.predict(df)

    return render_template(
        "index.html",
        prediction=f"Prediction: {prediction[0]}"
    )


if __name__ == "__main__":

    app.run(debug=True)