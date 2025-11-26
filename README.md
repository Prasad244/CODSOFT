**Credit Card Fraud Detection using Logistic Regression & SMOTE**

This project focuses on building a machine learning model to identify fraudulent credit card transactions using the popular Kaggle dataset “Credit Card Fraud Detection.”
The dataset is highly imbalanced, with legitimate transactions vastly outnumbering fraudulent ones—making fraud detection a challenging but important task in financial security.

📌 Project Overview

This project demonstrates end-to-end machine learning steps:

✔️ Load and preprocess the dataset

Normalize the Amount and Time features using StandardScaler

Reorganize columns for cleaner processing

Split data into training and testing sets

✔️ Handle class imbalance

The dataset is extremely skewed, so the model must treat fraudulent samples carefully.
We apply:

SMOTE (Synthetic Minority Oversampling Technique)
to balance the training dataset and improve fraud detection recall.

✔️ Train a machine learning model

We train Logistic Regression twice:

Before SMOTE (on the imbalanced dataset)

After SMOTE (on the balanced dataset)

This allows a clear comparison of how oversampling impacts model performance.

✔️ Evaluate the model

We use standard classification metrics:

Precision

Recall

F1-Score

AUPRC (Average Precision-Recall Score)

Confusion Matrix

Visualization is provided to compare before vs after SMOTE performance, showing how balancing the dataset affects fraud detection accuracy.

📊 Key Insights
🔹 Before SMOTE

Model predicts the majority class (legitimate transactions) most of the time

Precision is high, but

Recall is very low (frauds are rarely detected)

🔹 After SMOTE

Model becomes more sensitive to fraud

Recall increases dramatically

Precision typically drops — a normal trade-off when detecting rare events

This demonstrates the importance of handling class imbalance in fraud detection.

📈 Visualizations Included

The project generates:

Class distribution before vs after SMOTE

Performance comparison chart (precision, recall, F1-score, AUPRC)

Confusion matrices via printed output

These help explain why model performance changes when the minority class is oversampled.

🧠 Technologies Used

Python

Pandas

NumPy

scikit-learn

imbalanced-learn (SMOTE)

Matplotlib

🚀 How to Run

Download the dataset from Kaggle:
Credit Card Fraud Detection- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place creditcard.csv in the project directory

Run:

python main.py

📚 Future Improvements

Some potential enhancements:

Try XGBoost, CatBoost, or Random Forests

Use undersampling, SMOTEENN, or SMOTETomek

Tune the decision threshold for better precision-recall balance

Add ROC and Precision-Recall curves

🏁 Conclusion

This project demonstrates how crucial it is to handle class imbalance in fraud detection.
By comparing Logistic Regression performance before and after SMOTE, we clearly see:

Precision may drop

Recall improves significantly

Overall fraud detection becomes more effective

It highlights the real-world trade-offs when detecting rare but critical events like fraudulent transactions.
