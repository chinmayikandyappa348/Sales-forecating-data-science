import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/sales.csv")

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)

# Monthly Sales
monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()

# Plot
monthly_sales.plot(figsize=(10,5))
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()