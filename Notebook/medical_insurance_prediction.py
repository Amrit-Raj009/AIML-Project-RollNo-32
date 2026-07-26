import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("dataset/insurance.csv")

print(df.head())
print(df.info())
print(df.describe())

# ==========================
# DATA CLEANING
# ==========================

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# Standardize text
df["sex"] = df["sex"].str.lower().str.strip()
df["smoker"] = df["smoker"].str.lower().str.strip()
df["region"] = df["region"].str.lower().str.strip()

# Check positive values
print("\nInvalid Age:", (df["age"] <= 0).sum())
print("Invalid BMI:", (df["bmi"] <= 0).sum())
print("Invalid Charges:", (df["charges"] <= 0).sum())

# ==========================
# EDA
# ==========================

# 1. Average charges by smoker
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="smoker", y="charges")
plt.title("Average Charges by Smoking Status")
plt.show()

# 2. BMI vs Charges
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker")
plt.title("BMI vs Charges")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(
    df[["age","bmi","children","charges"]].corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Heatmap")
plt.show()

# 4. Age vs Charges
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="age", y="charges")
plt.title("Age vs Charges")
plt.show()

# 5. Children vs Charges
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="children", y="charges")
plt.title("Children vs Charges")
plt.show()

# 6. Region vs Charges
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="region", y="charges")
plt.title("Average Charges by Region")
plt.show()

# ==========================
# FEATURE ENGINEERING
# ==========================

def bmi_category(x):
    if x < 18.5:
        return "Underweight"
    elif x < 25:
        return "Normal"
    elif x < 30:
        return "Overweight"
    else:
        return "Obese"

df["bmi_category"] = df["bmi"].apply(bmi_category)

# Create binary smoker column BEFORE encoding
df["smoker_binary"] = df["smoker"].map({"no":0,"yes":1})

# Interaction feature
df["smoker_bmi"] = df["smoker_binary"] * df["bmi"]

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["sex","smoker","region","bmi_category"],
    drop_first=True
)

print("\nColumns After Encoding")
print(df.columns)

# ==========================
# MODEL BUILDING
# ==========================

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("R2 Score :", round(r2,4))
print("MAE      :", round(mae,2))
print("RMSE     :", round(rmse,2))

# ==========================
# ACTUAL VS PREDICTED
# ==========================

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()


importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

print("\nFeature Importance")
print(importance[["Feature","Coefficient"]])



print("\nFeature Impact Analysis")

print("""
1. Smoking is the strongest predictor of insurance charges.
2. Smokers have significantly higher predicted medical costs than non-smokers.
3. BMI positively affects insurance charges.
4. The smoker-BMI interaction shows that high BMI has an even greater impact for smokers.
5. Age also increases predicted insurance charges.
6. Number of children and region generally have smaller effects compared with smoking and BMI.
""")
