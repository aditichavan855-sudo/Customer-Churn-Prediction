# 📊 Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using the IBM Telco Customer Churn dataset.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Real-time Feature Scaling using StandardScaler
- Logistic Regression Model Tuning
- Tree-based Model Comparison (Decision Tree, Random Forest)
- Interactive Streamlit Web Application
- Real-time Churn Probability Prediction

---

## 📂 Dataset

IBM Telco Customer Churn Dataset

- 7,043 customer records
- 19 features used for prediction

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| **Logistic Regression** | **80.45%** |
| Random Forest | 80.45% (or lower based on selection) |

*Note: Logistic Regression was selected as the final production model due to its optimal balance of performance and consistency.*

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
├── data/
│   └── Telco-Customer-Churn.csv
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── models/
│   ├── churn_model.pkl
│   ├── model_columns.pkl
│   └── scaler.pkl
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```
## ▶️ Installation & Setup

Clone the repository, install the dependencies, and launch the dashboard by running the following commands in your terminal:

```bash
# Clone the repository and navigate into it
git clone [https://github.com/yourusername/Customer-Churn-Prediction.git](https://github.com/yourusername/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction

# Install all required Python packages
pip install -r requirements.txt

# Train the models and generate the evaluation plots
python train.py

# Launch the interactive Streamlit dashboard
streamlit run app/app.py
```
## 🎯 Results
```
Predicts whether a telecom customer is likely to churn instantly.

Displays the exact probability of churn using a visual progress bar.

Features dynamic inputs for customer tenure, contract types, and billing rates.

Provides an interactive web interface built with Streamlit.
```

## 👨‍💻 Author
Aditi p Chavan

Data Science Student