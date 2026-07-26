import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Linear Regression Model")

# --------------------------------
# LOAD DATA
# --------------------------------
df = pd.read_csv("Dataset/insurance.csv")

# --------------------------------
# DATA CLEANING
# --------------------------------
df.drop_duplicates(inplace=True)

df["sex"] = df["sex"].str.lower().str.strip()
df["smoker"] = df["smoker"].str.lower().str.strip()
df["region"] = df["region"].str.lower().str.strip()

# --------------------------------
# BMI CATEGORY
# --------------------------------
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

df["smoker_binary"] = df["smoker"].map({"no":0,"yes":1})

df["smoker_bmi"] = df["smoker_binary"] * df["bmi"]

# --------------------------------
# SHOW DATA
# --------------------------------
st.header("Dataset")

st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

st.write("Missing Values")

st.write(df.isnull().sum())

# --------------------------------
# EDA
# --------------------------------
st.header("Exploratory Data Analysis")

fig1, ax = plt.subplots(figsize=(6,4))
df.groupby("smoker")["charges"].mean().plot(kind="bar", ax=ax)
ax.set_title("Average Charges by Smoking Status")
st.pyplot(fig1)

fig2, ax = plt.subplots(figsize=(6,4))

colors = df["smoker"].map({"yes":"red","no":"blue"})

ax.scatter(df["bmi"], df["charges"], c=colors)

ax.set_xlabel("BMI")

ax.set_ylabel("Charges")

ax.set_title("BMI vs Charges")

st.pyplot(fig2)

fig3, ax = plt.subplots(figsize=(6,4))

corr = df[["age","bmi","children","charges"]].corr()

im = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

plt.colorbar(im)

st.pyplot(fig3)

# --------------------------------
# ENCODING
# --------------------------------
df = pd.get_dummies(
    df,
    columns=["sex","smoker","region","bmi_category"],
    drop_first=True
)

# --------------------------------
# MODEL
# --------------------------------
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

# --------------------------------
# METRICS
# --------------------------------
st.header("Model Performance")

r2 = r2_score(y_test,y_pred)

mae = mean_absolute_error(y_test,y_pred)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

col1,col2,col3 = st.columns(3)

col1.metric("R² Score",round(r2,4))

col2.metric("MAE",round(mae,2))

col3.metric("RMSE",round(rmse,2))

# --------------------------------
# ACTUAL VS PREDICTED
# --------------------------------
fig4, ax = plt.subplots(figsize=(6,6))

ax.scatter(y_test,y_pred)

ax.set_xlabel("Actual")

ax.set_ylabel("Predicted")

ax.set_title("Actual vs Predicted")

st.pyplot(fig4)

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------
importance = pd.DataFrame({
    "Feature":X.columns,
    "Coefficient":model.coef_
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

st.header("Feature Importance")

st.dataframe(importance[["Feature","Coefficient"]])

# --------------------------------
# PREDICTION
# --------------------------------
st.header("Predict Insurance Charges")

age = st.slider("Age",18,64,30)

bmi = st.slider("BMI",15.0,55.0,28.0)

children = st.slider("Children",0,5,0)

sex = st.selectbox("Sex",["male","female"])

smoker = st.selectbox("Smoker",["no","yes"])

region = st.selectbox(
    "Region",
    ["northeast","northwest","southeast","southwest"]
)

bmi_cat = bmi_category(bmi)

smoker_binary = 1 if smoker=="yes" else 0

smoker_bmi = smoker_binary*bmi

sample = pd.DataFrame({
    "age":[age],
    "bmi":[bmi],
    "children":[children],
    "smoker_binary":[smoker_binary],
    "smoker_bmi":[smoker_bmi]
})

sample = pd.get_dummies(
    sample
)

sample = sample.reindex(columns=X.columns,fill_value=0)

if st.button("Predict"):

    prediction = model.predict(sample)[0]

    st.success(f"Estimated Insurance Charges: ₹{prediction:,.2f}")

# --------------------------------
# ANALYSIS
# --------------------------------
st.header("Feature Impact Analysis")

st.markdown("""
- Smoking is the strongest predictor of insurance charges.
- Smokers have significantly higher predicted medical costs.
- BMI positively affects insurance charges.
- High BMI combined with smoking increases predicted charges further.
- Age also contributes positively to insurance costs.
- Number of children and region have comparatively smaller effects.
""")