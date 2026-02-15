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
    # log uploaded filename to server logs
    try:
        uploaded_name = uploaded.name
    except Exception:
        uploaded_name = 'uploaded_file'
    print(f"[streamlit] Uploaded file: {uploaded_name}")
    try:
        uploaded.seek(0)
        input_df = pd.read_csv(uploaded, sep=None, engine='python')
    except Exception:
        try:
            uploaded.seek(0)
            input_df = pd.read_csv(uploaded, sep=';')
        except Exception:
            uploaded.seek(0)
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
        print('[streamlit] Detected column `y` in uploaded data')
    else:
        print('[streamlit] No `y` column detected in uploaded data')

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

    # If true labels were provided, show evaluation metrics
    if y_col is not None:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        st.header('Evaluation on uploaded data')

        # Attempt to map common string labels (yes/no,true/false,y/n) to 1/0
        def _map_to_binary(series):
            s = series.astype(str).str.lower().str.strip()
            mapping = {'yes':1, 'y':1, 'true':1, '1':1, 'no':0, 'n':0, 'false':0, '0':0}
            mapped = s.map(mapping)
            unmapped_count = mapped.isna().sum()
            if unmapped_count == 0:
                return mapped.astype(int).values, 0
            # try numeric coercion for remaining
            num = pd.to_numeric(series, errors='coerce')
            if num.notna().all():
                return num.astype(int).values, 0
            return mapped.where(mapped.notna(), series).values, unmapped_count

        y_mapped, unmapped = _map_to_binary(y_col)
        if unmapped == 0:
            print('[streamlit] Successfully mapped uploaded `y` to numeric 0/1')
        else:
            print(f'[streamlit] Warning: {unmapped} `y` values could not be mapped to 0/1; attempting to compute metrics may fail')

        y_true = y_mapped
        y_pred = preds

        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics_df = pd.DataFrame({
            'metric': ['accuracy','precision','recall','f1'],
            'value': [acc, prec, rec, f1]
        })
        st.subheader('Summary metrics')
        st.table(metrics_df.set_index('metric'))

        # Classification report
        st.subheader('Classification report')
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        st.subheader('Confusion matrix')
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
        st.table(cm_df)
        # print summary to server logs as well
        print(f"[streamlit] Eval metrics - accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")
        print(f"[streamlit] Confusion matrix:\n{cm}")
