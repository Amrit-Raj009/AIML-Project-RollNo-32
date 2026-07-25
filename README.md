# [Medical Insurance Cost Prediction]
 
## Problem Statement
[Medical insurance companies need to estimate a customer's annual medical insurance charges based on personal attributes such as age, BMI, number of children, smoking status, sex, and region. This project builds a machine learning model to accurately predict insurance charges and identify the factors that have the greatest impact on medical costs.]
 
## Dataset
- **Name:** [Medical Cost Personal Dataset]
- **Source:** [Kaggle (mirichoi0218)]
- **Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance]
- **Rows / Columns:** [1338 rows, 7 columns]
 
## Tools Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- 
 
## Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Building (Linear Regression)
6. Evaluation
7. Insights & Recommendations
 
## Results
- **Model:** [ LinearRegression]
- **Key Metric(s):** [R2 Score : 0.8868
MAE      : 2813.07
RMSE     : 4560.55]
- **Top Factors / Drivers:** [Smoking status is the strongest predictor of insurance charges.
Customers who smoke are predicted to have significantly higher insurance charges than non-smokers.
The model shows that smoking has the largest positive impact on the predicted cost, making it the most influential feature.
Body Mass Index (BMI)
Higher BMI is associated with higher insurance charges.
Individuals with obesity generally have higher predicted medical costs.
The interaction feature (Smoker × BMI) indicates that smokers with a higher BMI tend to have even greater insurance charges.
Age
Insurance charges generally increase as age increases.
Older individuals are more likely to have higher predicted medical expenses, making age one of the most important predictors after smoking status.]
 
## Screenshots


### 1. Average Charges: Smokers vs Non-Smokers
[Smoker vs Charges](Images/smokervscharges.png)

### 2. BMI vs Insurance Charges
[BMI vs Charges](Images/bmi vs charges.png)

### 3. Correlation Heatmap
[Correlation Heatmap](Images/ heatmap1.png)

### 4. Predicted vs Actual Charges
[Predicted vs Actual](Images/actual vs predicted.png)
 
## Future Improvements
- [Compare Linear Regression with advanced regression models such as Random Forest Regressor and XGBoost.
Perform hyperparameter tuning to improve prediction accuracy.
Apply feature scaling and cross-validation for better model performance.
Deploy the model as a web application using Flask or Streamlit.]
 
## Author
[Amrit Raj B.Tech CSE] | [linkedin.com/.in/amrit-raj-73127b38]
