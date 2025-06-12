# House Price Prediction

---

## ℹ️ About the Project

**House Price Prediction** is project that:

- Analyzes Housing Price
- Analyzes House Features Correlation
- Predicts House Price and Evaluates Error Metrics (R2 Score and RMSE), using:
   - Linear SVR (Support Vector Regression)
   - SVR
   - Ridge
   - Bayesian Ridge
   - Elastic Net
   - Lasso
   - Linear Regression
   - XGBoost Regressor
   - Random Forest Regressor

---

## 🛠️ Built With

- [Python](https://www.python.org/) — primary programming language
- [PyCharm](https://www.jetbrains.com/pycharm/) — IDE

- [Scikit-learn](https://scikit-learn.org/stable/) — machine learning
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) — xgboost method
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [NumPy](https://numpy.org/) — number operations
- [Matplotlib](https://matplotlib.org/) — plotting
- [Seaborn](https://seaborn.pydata.org/) — data visualization

---

## 📦 Getting Started

### Prerequisites

To run the project locally, you'll need:

- PyCharm (2025.1.1.1 or newer)

---

### Installation & Setup

1. **Install the necessary libraries:**

   ```bash
   pip install scikit-learn xgboost pandas numpy matplotlib seaborn kaggle

2. **Download dataset through the terminal:**

   ```bash
   kaggle datasets download -d shree1992/housedata

3. **Extract the dataset zip:**

   ```bash
   with zipfile.ZipFile("housedata.zip", "r") as zip_ref:
      zip_ref.extractall("housedata")

4. **Run the code:**

   Run the provided `code.py`

---

## 📊 Results

![Alt text](Figure_1.png?raw=true "Title")
![Alt text](Figure_2.png?raw=true "Title")
![Alt text](Figure_3.png?raw=true "Title")
![Alt text](Figure_4.png?raw=true "Title")

## 📃 License

This project is licensed under the MIT License. See the `LICENSE.txt` file for more information.
