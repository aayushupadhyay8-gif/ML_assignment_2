import streamlit as st
import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

st.title('Bank Marketing Classification — Model Comparison')

# show results table
results_path = os.path.join(MODEL_DIR, 'results.csv')
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    st.header('Model Comparison')
    st.dataframe(df)
else:
    st.warning('No results found. Run the training script first (`python model/train.py`).')

st.header('Predict on new data')
uploaded = st.file_uploader('Upload a CSV with the original Bank dataset columns (bank-full.csv columns)', type=['csv'])

if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    st.write('Preview of uploaded data:')
    st.dataframe(input_df.head())

    # load artifacts
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.joblib'))

    # preprocess uploaded file same way as training
    y_col = None
    if 'y' in input_df.columns:
        y_col = input_df.pop('y')

    cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    input_processed = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    # align to training columns
    for c in feature_cols:
        if c not in input_processed.columns:
            input_processed[c] = 0
    input_processed = input_processed[feature_cols]

    input_scaled = scaler.transform(input_processed)

    model_name = st.selectbox('Choose model', ['LogisticRegression','DecisionTree','KNN','GaussianNB','RandomForest','XGBoost'])
    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.joblib"))

    preds = model.predict(input_scaled)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(input_scaled)[:,1]
    else:
        probs = None

    out = input_df.copy()
    out['prediction'] = preds
    if probs is not None:
        out['probability'] = probs

    st.write('Predictions:')
    st.dataframe(out.head())
    csv = out.to_csv(index=False).encode('utf-8')
    st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv')
