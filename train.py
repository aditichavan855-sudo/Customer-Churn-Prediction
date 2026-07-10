import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score
)

# ======================================================
# PROJECT PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

MODEL_DIR = BASE_DIR / "models"

IMAGE_DIR = BASE_DIR / "images"

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_DIR.mkdir(exist_ok=True)

# ======================================================
# LOAD DATASET
# ======================================================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape :", df.shape)

print("\nColumns")

print(df.columns)

# ======================================================
# DATA CLEANING
# ======================================================

df["Monthly Charges"] = pd.to_numeric(
    df["Monthly Charges"],
    errors="coerce"
)

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Remove only rows where Total Charges is missing
df.dropna(subset=["Total Charges"], inplace=True)

print("\nDataset Shape After Cleaning :", df.shape)

print("\nTarget Distribution")

print(df["Churn Value"].value_counts())

# ======================================================
# SELECT FEATURES
# ======================================================

y = df["Churn Value"]

X = df.drop(
    columns=[
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Value",
        "Churn Score",
        "CLTV",
        "Churn Reason"
    ]
)

# ======================================================
# ONE HOT ENCODING
# ======================================================

categorical_columns = X.select_dtypes(
    include="object"
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False
)

model_columns = X.columns.tolist()

print("\nNumber of Features :", len(model_columns))

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ======================================================
# SCALE NUMERICAL FEATURES
# ======================================================

scaler = StandardScaler()

numeric_columns = [

    "Tenure Months",

    "Monthly Charges",

    "Total Charges"

]

X_train[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

print("\nPreprocessing Completed Successfully.")
# ======================================================
# MODEL TRAINING
# ======================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

}

best_model = None
best_model_name = ""
best_accuracy = 0

results = []

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "ROC AUC": auc

    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print(results_df)

print("\nBest Model :", best_model_name)

print("Best Accuracy :", round(best_accuracy * 100, 2), "%")

# ======================================================
# FINAL PREDICTIONS
# ======================================================

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        y_test,

        predictions

    )

)

# ======================================================
# SAVE MODEL
# ======================================================

joblib.dump(

    best_model,

    MODEL_DIR / "churn_model.pkl"

)

joblib.dump(

    model_columns,

    MODEL_DIR / "model_columns.pkl"

)

joblib.dump(

    scaler,

    MODEL_DIR / "scaler.pkl"

)

print("\nModel Saved Successfully!")

print("Saved Files:")

print("✔ churn_model.pkl")

print("✔ model_columns.pkl")

print("✔ scaler.pkl")
# ======================================================
# CONFUSION MATRIX
# ======================================================

print("\nGenerating Confusion Matrix...")

disp = ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig(
    IMAGE_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ confusion_matrix.png saved")

# ======================================================
# ROC CURVE
# ======================================================

print("\nGenerating ROC Curve...")

RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title("ROC Curve")

plt.savefig(
    IMAGE_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ roc_curve.png saved")

# ======================================================
# FEATURE / COEFFICIENT IMPORTANCE
# ======================================================

print("\nGenerating Feature Importance...")

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": model_columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        importance["Feature"][:10],
        importance["Importance"][:10]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 10 Feature Importance")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✔ feature_importance.png saved")

else:

    coefficients = pd.DataFrame({

        "Feature": model_columns,

        "Coefficient": best_model.coef_[0]

    })

    coefficients["Importance"] = coefficients["Coefficient"].abs()

    coefficients = coefficients.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        coefficients["Feature"][:10],
        coefficients["Coefficient"][:10]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 10 Logistic Regression Coefficients")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✔ feature_importance.png saved")

# ======================================================
# SAVE MODEL METRICS
# ======================================================

with open(MODEL_DIR / "model_accuracy.txt", "w") as f:
    f.write(f"{best_accuracy:.2%}")

results_df.to_csv(
    MODEL_DIR / "model_results.csv",
    index=False
)

print("✔ model_accuracy.txt saved")
print("✔ model_results.csv saved")

# ======================================================
# FINAL SUMMARY
# ======================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Best Model      : {best_model_name}")
print(f"Accuracy        : {best_accuracy:.2%}")
print(f"ROC AUC         : {roc_auc_score(y_test, probabilities):.4f}")

print("\nGenerated Files")

print("📁 models/")
print("   ✔ churn_model.pkl")
print("   ✔ model_columns.pkl")
print("   ✔ scaler.pkl")
print("   ✔ model_accuracy.txt")
print("   ✔ model_results.csv")

print("\n📁 images/")
print("   ✔ confusion_matrix.png")
print("   ✔ roc_curve.png")
print("   ✔ feature_importance.png")

print("\nYour Customer Churn Prediction model is ready!")
