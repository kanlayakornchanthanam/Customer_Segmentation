# Mall Customer Segmentation

A complete end-to-end machine learning project that segments mall customers into distinct groups using unsupervised clustering, then builds a supervised classifier to predict segments for new customers — deployed as an interactive Streamlit dashboard.

---

## 🔗 Live Demo

[Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)][(https://customer-segmentation-bykandy.streamlit.app/)

---

## 📁 Project Structure

```
Customer-segmentation/
│
├── notebooks/
│   ├── Part1_Clustering.ipynb       # EDA + K-Means + Hierarchical + DBSCAN
│   └── Part2_Classifier.ipynb       # LR + RF + XGBoost + SVM + Streamlit export
│
├── customer_segmentation_app/streamlit_app/
│   ├── app.py                       # Streamlit dashboard
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml              # Dark theme config
│
├── data/
│   ├── Mall_Customers.csv           # Original dataset (Kaggle)
│   └── clustered_customers.csv      # Labelled dataset exported from Part 1
│
├── models/
│   ├── model.pkl                    # Trained best classifier
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── segment_names.json           # Cluster label → business name
│   └── model_meta.json              # Model name, accuracy, feature info
│
└── README.md
```

---

## 📊 Dataset

**Mall Customer Segmentation** — [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Customer age (18–70) |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Mall-assigned spending score |

- **200 rows**, no missing values
- Clustering features: **Annual Income** × **Spending Score**
- Classifier features: Age, Gender, Annual Income, Spending Score

---

## 🔬 Methodology

### [Part 1 — Unsupervised Clustering (`ML_Customer_segmentation_part1.ipynb`)](https://colab.research.google.com/github/kanlayakornchanthanam/Customer_Segmentation/blob/main/Notebooks/Part1_Clustering.ipynb)

**Goal:** Find the natural customer segments without any labels.

| Step | What happened |
|---|---|
| EDA | Quality check, distributions, boxplots, pairplot |
| Feature scaling | StandardScaler — critical for distance-based models |
| Optimal k | Elbow Method + Silhouette Score → **k = 5** |
| K-Means | Baseline clustering with k-means++ initialisation |
| Hierarchical | Ward linkage dendrogram confirmed k = 5 |
| DBSCAN | k-distance plot tuned eps; handles noise automatically |
| Model comparison | Silhouette, Davies-Bouldin, Calinski-Harabasz → best model selected |
| Export | `clustered_customers.csv` saved for Part 2 |

> **Why not include Gender in clustering?** Gender was tested and reduced the Silhouette Score — it added noise rather than signal. Dropped from clustering but included in the classifier.

---

### [Part 2 — Supervised Classifier (`ML_Customer_segmentation_part2.ipynb`)](https://colab.research.google.com/github/kanlayakornchanthanam/Customer_Segmentation/blob/main/Notebooks/Part2_Classifier.ipynb)

**Goal:** Train a classifier to predict the segment of a brand new customer without re-running clustering.

| Step | What happened |
|---|---|
| Preprocessing | Label-encoded Gender, StandardScaler on all 4 features |
| Stratified split | 80/20 train/test — preserves class proportions |
| Logistic Regression | Linear baseline + GridSearchCV |
| Random Forest | Ensemble bagging + GridSearchCV |
| XGBoost | Ensemble boosting + GridSearchCV |
| SVM | Margin-based + GridSearchCV |
| Evaluation | 5-fold stratified CV + test accuracy |
| Feature importance | RF and XGBoost importance charts |
| Export | `model.pkl`, `scaler.pkl` for Streamlit app |

**All models tuned with GridSearchCV + 5-fold stratified cross-validation.**

---

## 🏆 Results

### Clustering (Part 1)

| Model | n_clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| K-Means | 5 | 0.5547 | 0.5722 | 248.6493 |
| Hierarchical | 5 | 0.5538 | 0.5779 | 244.4103 |
| DBSCAN* | 2 | 0.3876 | 0.7889 | 85.3897 |

✅**Best model: K-Means** — highest Silhouette Score and lowest Davies-Bouldin Index.

### Classification (Part 2)

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistill
c Regression | 0.9625 | 0.975 |
| Random Forest | 0.9688 | 0.975 |
| XGBoost | 0.9438 | 0.975 |
| SVM | 0.9750 | 0.975 |

✅**Best classifier: SVM** — highest CV accuracy (0.9750).

---

## 🧩 Customer Segments

| Cluster | Segment | Income | Spending | Strategy |
|---|---|---|---|---|
| 0 | 💼 Cautious Shoppers | Low | Low | Discounts, loyalty points |
| 1 | 🛒 Impulsive Buyers | Low | High | Trending items, FOMO offers |
| 2 | 👥 Average Customers | Mid | Mid | Consistency, moderate promos |
| 3 | 🏦 Wealthy but Frugal | High | Low | Premium quality, exclusivity |
| 4 | ⭐ VIP Targets | High | High | VIP memberships, personalised experiences |

---

## 📱 Streamlit Dashboard

The app has 4 tabs:

| Tab | Content |
|---|---|
| 🔍 **Predict** | Input customer profile → segment + confidence + scatter position |
| 📊 **Cluster Analysis** | Distribution, Income×Spending scatter, boxplots per cluster |
| 🤖 **Model Comparison** | CV vs Test accuracy for all 4 models + feature importance |
| 📋 **Segment Profiles** | Business descriptions + marketing strategies per segment |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/kanlayakornchanthanam/Customer_Segmentation.git
cd Customer_Segmentation

# 2. Install dependencies
pip install -r app/requirements.txt

# 3. Run the app
streamlit run app/app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Set:
   - **Repository:** `kanlayakornchanthanam/Customer_Segmentation`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
5. Click **Deploy**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| Scikit-learn | Clustering + classification + preprocessing |
| XGBoost | Gradient boosting classifier |
| Streamlit | Interactive web dashboard |
| Google Colab | Notebook environment |
| GitHub | Version control + Streamlit deployment |

---

## 📚 References

- Dataset: [Mall Customer Segmentation Data — Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Scikit-learn documentation: [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html)
- XGBoost documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

---

## 👤 Author

- GitHub: [@kanlayakornchanthanam](https://github.com/kanlayakornchanthanam)

---

*Built as part of a machine learning portfolio project.*
