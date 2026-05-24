# Mall Customer Segmentation — Streamlit App

A full ML dashboard for customer segmentation, built on K-Means clustering + Random Forest classifier.

## Features
- 🔍 **Predict** — input customer profile, get segment + confidence
- 📊 **Cluster Analysis** — distribution, scatter plots, boxplots
- 🤖 **Model Comparison** — LR vs RF vs XGBoost vs SVM
- 📋 **Segment Profiles** — business descriptions + marketing strategies

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select your repo, branch, and set `app.py` as the main file
5. Click **Deploy** — done!

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Connecting your own trained model
Replace the `load_model()` function in `app.py` with:
```python
import joblib
clf = joblib.load('model.pkl')
scaler_full = joblib.load('scaler.pkl')
```
Then save your trained objects from the Colab notebook:
```python
import joblib
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler_full, 'scaler.pkl')
```
