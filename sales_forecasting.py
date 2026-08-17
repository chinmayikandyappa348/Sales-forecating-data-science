import pandas as pd

# Load dataset
df = pd.read_csv("dataset/sales.csv")

# Display basic information
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create new columns
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# Display first 5 rows
print("\nCleaned Data:")
print(df.head())