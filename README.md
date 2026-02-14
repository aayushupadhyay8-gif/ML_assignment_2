# Bank Marketing Classification Assignment

**Problem statement**
- Predict whether a client will subscribe to a term deposit (`y` yes/no) based on attributes from the Bank Marketing dataset (classification problem).

**Dataset description**
- Source: UCI Machine Learning Repository — Bank Marketing (bank-full.csv)
- Instances: 45,211
- Attributes: 16 (mixed categorical and numerical). After one-hot encoding, feature vector > 12 dimensions.

**Models implemented**
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

**Evaluation metrics (calculated for each model)**
- Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)

**Repository structure**
project-folder/
- streamlit_app.py — simple Streamlit UI to display comparison and run predictions
- requirements.txt
- README.md
- model/
  - train.py — training script which downloads dataset, trains all models, saves models and results
  - *.joblib — saved model files (created after running `train.py`)
  - results.csv — metrics table for all models (created after running `train.py`)

**How to run**
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models (this downloads the dataset and saves models into `model/`):

```bash
python model/train.py
```

3. Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

**Deliverables for submission**
- `streamlit_app.py`
- `requirements.txt`
- `README.md` (this file)
- `model/` folder containing trained model files and `results.csv`


**Notes**
- Run `model/train.py` to produce the `results.csv` and saved model files required by the Streamlit app.
- If you want, I can run the training here and attach the generated `results.csv` and model files — tell me to proceed and I will run the training now.