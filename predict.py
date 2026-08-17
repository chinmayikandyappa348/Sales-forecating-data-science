import joblib
import pandas as pd

# Load model
model = joblib.load("model/sales_model.pkl")

# Input data (same columns used during training)
data = pd.DataFrame({
    "Order Date": ["2026-08-01"],
    "Product": ["Laptop"],
    "Category": ["Technology"],
    "Sub-Category": ["Accessories"],
    "Region": ["South"]
})

# Prediction
prediction = model.predict(data)

print("Forecasted Sales:", prediction)