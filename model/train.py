import os
import zipfile
import io
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)


def download_bank_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    dest = os.path.join(DATA_DIR, "bank.zip")
    if not os.path.exists(os.path.join(DATA_DIR, "bank-full.csv")):
        print("Downloading Bank Marketing dataset...")
        r = requests.get(url)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(dest) as z:
            z.extractall(DATA_DIR)
        print("Downloaded and extracted dataset.")
    else:
        print("Dataset already present.")


def load_and_preprocess():
    csv_path = os.path.join(DATA_DIR, "bank-full.csv")
    df = pd.read_csv(csv_path, sep=';')
    # target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # separate features
    X = df.drop(columns=['y'])
    y = df['y']

    # one-hot encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, scaler


def train_and_save():
    download_bank_dataset()
    X, y, scaler = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # fallback: use decision function if available
            try:
                y_proba = model.decision_function(X_test)
                # scale to 0-1
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            except Exception:
                y_proba = np.zeros_like(y_pred)

        acc = metrics.accuracy_score(y_test, y_pred)
        try:
            auc = metrics.roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float('nan')
        prec = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc
        })

        # save model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"Saved {name} to {os.path.join(MODEL_DIR, f'{name}.joblib')}")

    # save scaler and feature columns
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_columns.joblib"))

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "results.csv"), index=False)
    print("Saved results to results.csv")


if __name__ == '__main__':
    train_and_save()
