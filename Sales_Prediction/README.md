📊 Advertising Sales Prediction
Linear Regression Model on Marketing Spend Data


A machine-learning project that predicts Sales based on advertising budgets across TV, Radio, and Newspaper.
Includes full EDA, model training, evaluation, and visualization.

📁 Project Structure
├── advertising.csv
├── script.py / notebook.ipynb
└── README.md

🚀 Features

✔ Load & explore dataset
✔ Correlation heatmap and pairplots
✔ Train-test split
✔ Linear Regression model
✔ Model performance metrics (RMSE, R²)
✔ Actual vs Predicted visualization
✔ Baseline model comparison
✔ Extract model coefficients

🧰 Installation

Install required packages:

pip install pandas numpy scikit-learn matplotlib seaborn

📥 Dataset

The dataset (advertising.csv) contains four columns:

Feature	Description
TV	TV advertising budget
Radio	Radio advertising budget
Newspaper	Newspaper advertising budget
Sales	Product sales (target variable)
📌 Exploratory Data Analysis
🔥 Correlation Heatmap

Helps identify strong predictors of Sales.

🔍 Pairplot

Visualizes relationships between features and the target.

sns.heatmap(df.corr(), annot=True, cmap="BrBG")
sns.pairplot(df)

🧠 Model Training

The features used:

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


Train-test split:

train_test_split(X, y, test_size=0.2, random_state=42)


Train model:

model = LinearRegression()
model.fit(X_train, y_train)

📈 Model Evaluation

Metrics computed:

RMSE: 1.65
R² Score: 0.90


Baseline comparison:

Baseline RMSE: 4.22
Error Reduction: 60.9%

📊 Actual vs Predicted Plot

Visualizes how well the model fits the data:

plt.scatter(y_test, y_pred)
plt.plot([0, 30], [0, 30], color='red')

🧾 Model Coefficients
Feature	Coefficient
TV	…
Radio	…
Newspaper	…
▶️ Running the Script
python script.py


or open in Google Colab/Jupyter Notebook.

🧩 Future Improvements

🔹 Add Lasso/Ridge regularization
🔹 Use cross-validation
🔹 Hyperparameter optimization
🔹 Deploy with Streamlit or Flask
