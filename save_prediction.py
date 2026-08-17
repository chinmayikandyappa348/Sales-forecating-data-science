import pandas as pd
import joblib

model = joblib.load("model/sales_model.pkl")

data = pd.DataFrame({
    "Order Date": ["2026-08-01"],
    "Product": ["Laptop"],
    "Category": ["Technology"],
    "Sub-Category": ["Accessories"],
    "Region": ["South"]
})

prediction = model.predict(data)

result = data.copy()
result["Forecasted Sales"] = prediction

result.to_csv("forecast_result.csv", index=False)

print("Forecast saved successfully")