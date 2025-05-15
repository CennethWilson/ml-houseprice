# import kaggle
# import zipfile
#
# # kaggle datasets download -d shree1992/housedata
# with zipfile.ZipFile("housedata.zip", "r") as zip_ref:
#      zip_ref.extractall("housedata")

import sklearn                      # Machine Learning
import pandas as pd                 # Data manipulation
import numpy as np                  # Number operations
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Data visualization

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("housedata/data.csv")
df = df.drop(columns=["street", "date", "statezip"])
df = df[df["price"] > 0]
df = df[df["bedrooms"] != 0]

y_plot = df.price
y = np.log1p(df.price)

plt.style.use("ggplot")
backgroundColor = "#F2E9E4"

limit = 2500000
plt.figure(facecolor=backgroundColor)
sns.histplot(y_plot[y_plot < limit], bins=20, kde=True, color="#1f77b4")
plt.xlim(0, limit)
plt.axvline(x = y_plot.median(), linestyle="--", label=f"Median: {y_plot.median():.0f} USD")
plt.axvline(x = y_plot.mean(), linestyle="-", label=f"Mean: {y_plot.mean():.0f} USD", color="#ff9f1c")
plt.legend()
plt.title("Housing Price Analysis", weight="bold")
plt.xlabel("Price (Million USD)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 9), facecolor=backgroundColor)
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f")
plt.title("House Features Correlation", weight="bold")
plt.tight_layout()
plt.show()

# Data Preparation
df["living_lot_ratio"] = df["sqft_lot"] / df["sqft_living"]
df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"]
df["house_age"] = 2025 - df["yr_built"]
df["since_renovated"] = np.where(df["yr_renovated"] == 0, df["house_age"], 2025 - df["yr_renovated"])
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df["bath_to_bed_ratio"] = df["bathrooms"] / df["bedrooms"]
df["renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)
df["has_basement"] = np.where(df["sqft_basement"] > 0, 1, 0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

num_cols = [col for col in df.columns if df[col].dtype in ['float64','int64'] and col != "price"]
cat_cols = [col for col in df.columns if df[col].dtype not in ['float64','int64']]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[cat_cols])
encoded_cols = list(encoder.get_feature_names_out(cat_cols))
df[encoded_cols] = encoder.transform(df[cat_cols])

x_train, x_test, y_train, y_test = train_test_split(df[num_cols + encoded_cols], y, test_size=0.25, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn import svm
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

models = {
    "Linear SVR" : svm.LinearSVR(),
    "SVR" : svm.SVR(),
    "Ridge" : Ridge(),
    "Bayesian Ridge" : BayesianRidge(),
    "Elastic Net" : ElasticNet(),
    "Lasso" : Lasso(),
    # "SGD Regressor" : SGDRegressor(),
    "Linear Regression" : LinearRegression(),
    "XGBoost Regressor" : XGBRegressor(),
    "Random Forest Regressor" : RandomForestRegressor()
}

result = []

for name, model in models.items():
    if name not in ["XGBoost Regressor", "Random Forest Regressor"]:
        fit = model.fit(x_train_scaled, y_train)
        predict = np.expm1(fit.predict(x_test_scaled))
    else:
        fit = model.fit(x_train, y_train)
        predict = np.expm1(fit.predict(x_test))
    trueprice = np.expm1(y_test)
    score = np.sqrt(mean_squared_error(trueprice, predict))
    r2 = r2_score(trueprice, predict)
    result.append({
        "model": name,
        "rmse": score,
        "r2_score": r2
    })

resultdf = pd.DataFrame(result)
print(resultdf)

resultdf = resultdf.sort_values(by=["r2_score"], ascending=False)
plt.figure(figsize=(12, 9), facecolor=backgroundColor)
plt.plot(resultdf["model"], resultdf["r2_score"], marker="o")
for i, (model, r2) in enumerate(zip(resultdf["model"], resultdf["r2_score"])):
    plt.text(i, r2 + 0.03, f"{r2:.2f}", ha='center', va='bottom', fontsize=12, color="black")
plt.title("R2 Score Across Models", weight="bold")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.ylabel("R2 Score")
ymin = resultdf["r2_score"].min()
plt.ylim(ymin - 0.3, 1)
plt.tight_layout()
plt.show()

resultdf = resultdf.sort_values(by=["rmse"], ascending=True)
plt.figure(figsize=(12, 9), facecolor=backgroundColor)
plt.plot(resultdf["model"], resultdf["rmse"], marker="o")
for i, (model, r2) in enumerate(zip(resultdf["model"], resultdf["rmse"])):
    plt.text(i, r2 + 0.03, f"{r2/1e3:.0f} K", ha='center', va='bottom', fontsize=12, color="black")
plt.title("RMSE Across Models", weight="bold")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
