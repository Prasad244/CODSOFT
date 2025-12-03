# 📊 Advertising Sales Prediction

### Linear Regression on Marketing Spend Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![ML](https://img.shields.io/badge/Model-Linear%20Regression-yellow)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20Sklearn%20%7C%20Matplotlib%20%7C%20Seaborn-orange)

A machine learning project that predicts **Sales** using advertising budgets from **TV**, **Radio**, and **Newspaper** channels.
Includes full **EDA**, **model training**, **evaluation**, and **visualization**.

---

## 📁 Project Structure

```
├── advertising.csv
├── script.py  (or .ipynb)
└── README.md
```

---

## 🧰 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📊 Dataset

The dataset (`advertising.csv`) contains:

| Column    | Description                       |
| --------- | --------------------------------- |
| TV        | Budget spent on TV ads            |
| Radio     | Budget spent on Radio ads         |
| Newspaper | Budget spent on Newspaper ads     |
| Sales     | Sales generated (target variable) |

---

## 🔍 Exploratory Data Analysis (EDA)

### Correlation Heatmap

Used to identify feature relationships.

```python
sns.heatmap(df.corr(), annot=True, cmap="BrBG")
```

### Pairplot

Shows feature interactions.

```python
sns.pairplot(df)
```

---

## 🧠 Model Training

```python
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📈 Model Evaluation

```python
RMSE: 1.65
R² Score: 0.90
```

### Baseline Comparison

```python
Baseline RMSE: 4.22
Error Reduction: 60.9%
```

---

## 📊 Visualization

### Actual vs Predicted Sales

```python
plt.scatter(y_test, y_pred)
plt.plot([0, 30], [0, 30], color='red')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
```

---

## 🧾 Model Coefficients

| Feature   | Coefficient |
| --------- | ----------- |
| TV        | ...         |
| Radio     | ...         |
| Newspaper | ...         |

---

## ▶️ How to Run

```bash
python script.py
```

Or open the notebook in **Jupyter / Google Colab**.

---

## 🚀 Future Enhancements

* Regularization (Ridge, Lasso)
* Cross-validation
* Hyperparameter tuning
* Interactive dashboard (Streamlit)

---


