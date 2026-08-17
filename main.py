from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/sales_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = pd.DataFrame({
        "Order Date": [request.form["order_date"]],
        "Product": [request.form["product"]],
        "Category": [request.form["category"]],
        "Sub-Category": [request.form["sub_category"]],
        "Region": [request.form["region"]]
    })

    prediction = model.predict(data)

    return render_template(
        "index.html",
        result=round(float(prediction[0]), 2)
    )


if __name__ == "__main__":
    app.run(debug=True)