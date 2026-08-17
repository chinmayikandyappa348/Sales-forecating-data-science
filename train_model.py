import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("dataset/sales.csv")

# Features
X = df[["Category", "Sub-Category", "Region"]]

# Target
y = df["Sales"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["Category", "Sub-Category", "Region"])
    ]
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "model/sales_model.pkl")

print("Model saved successfully!")