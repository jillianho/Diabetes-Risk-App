# File: train.py
# What this does: downloads the data, trains the model, saves it to disk

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ── 1. Load the data ──────────────────────────────────────────────────────────
columns = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

try:
    df = pd.read_csv(url, names=columns)
    print("✓ Data loaded from internet")
except:
    # Fallback: load from scikit-learn if SSL fails
    from sklearn.datasets import fetch_openml
    pima = fetch_openml(name="diabetes", version=1, as_frame=True)
    df = pima.frame
    df['outcome'] = (df['class'] == 'tested_positive').astype(int)
    df = df.drop(columns=['class'])
    df.columns = columns[:8].tolist() + ['outcome']
    print("✓ Data loaded from scikit-learn fallback")

print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ── 2. Split into inputs (X) and answer (y) ───────────────────────────────────
# X = the health measurements (what we feed IN to the model)
# y = the outcome (0 = no diabetes, 1 = diabetes) — what we want to PREDICT
X = df.drop(columns=["outcome"])
y = df["outcome"]

# ── 3. Split into training data and test data ─────────────────────────────────
# We train on 80% of patients, test on the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training on {len(X_train)} patients, testing on {len(X_test)}")

# ── 4. Train the model ────────────────────────────────────────────────────────
# Random Forest = many decision trees voting together
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✓ Model trained")

# ── 5. Check accuracy ─────────────────────────────────────────────────────────
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✓ Accuracy on test data: {accuracy:.1%}")

# ── 6. Show which features matter most ───────────────────────────────────────
feature_names = X.columns.tolist()
importances = model.feature_importances_
print("\nTop risk factors (most → least important):")
for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(score * 40)
    print(f"  {name:<20} {bar} {score:.3f}")

# ── 7. Save the model to disk ─────────────────────────────────────────────────
# We save it so our app can load it without retraining every time
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✓ Model saved to model.pkl")
print("  Ready to build the app!")