# 📱 Mobile Product Segmentation & Recommendation System

An end-to-end Data Science project that segments mobile phones into meaningful clusters and recommends similar phones based on user selection.

---

## 🚀 Project Overview

This project analyzes 50,000 Global Mobile Reviews to:
- Identify distinct mobile product segments using **K-Means Clustering**
- Build a **Cosine Similarity** based recommendation system
- Present insights through an interactive **Streamlit** web application

---

## 📁 Project Structure
mobile_segmentation/
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned, encoded, clustered data
├── notebooks/
│   ├── analysis.ipynb              # Data cleaning & preprocessing
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── feature_engineering.ipynb   # Encoding & Scaling
│   └── modeling/
│       ├── clustering.ipynb        # K-Means Clustering
│       └── recommendation.ipynb    # Cosine Similarity
├── app/
│   └── app.py                      # Streamlit Application
├── models/                         # Saved models & scalers
└── README.md

---

## 📊 Dataset

- **Source**: Global Mobile Reviews Dataset
- **Size**: 50,000 reviews × 22 columns
- **Domain**: E-Commerce Analytics

**Key Features**: `brand`, `model`, `price_usd`, `rating`, `battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`, `sentiment`, `country`

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Static visualizations |
| Plotly | Interactive visualizations |
| Scikit-learn | K-Means, Cosine Similarity, Scaling |
| Streamlit | Web application |

---

## 🧹 Data Preprocessing

- Dropped 21 rows with 3+ null values
- Filled `price_usd` nulls with **median per brand**
- Filled `rating` nulls with **median**
- Filled `sentiment` and `source` nulls with **mode**
- Dropped 7 irrelevant columns
- Converted `verified_purchase` to int (0/1)

---

## 🔍 Feature Engineering

| Feature | Formula |
|---------|---------|
| `avg_feature_rating` | Mean of 5 feature ratings |
| `price_segment` | Budget / Mid-Range / Premium |
| `rating_gap` | `rating` - `avg_feature_rating` |

---

## 🤖 Clustering Results (K=4)

| Cluster | Label | Avg Price | Avg Rating | Size |
|---------|-------|-----------|------------|------|
| 0 | Budget Mid-Performers | $490 | 3.24 | 12,033 |
| 1 | Disappointing Mid-Range | $689 | 1.75 | 16,328 |
| 2 | Premium Average | $1,074 | 3.39 | 7,359 |
| 3 | Value Champions | $660 | 4.44 | 14,259 |

**K selected using**: Elbow Method + Silhouette Score

---

## 💡 Key Insights

- All 7 brands are equally represented (~7,000 reviews each)
- `price_usd` has **zero correlation** with ratings — expensive phones don't guarantee satisfaction
- **Value Champions** (Cluster 3) — best bang for buck at ~$660 with 4.44 avg rating
- **Disappointing Mid-Range** (Cluster 1) — customers feel cheated at $689 with only 1.75 rating

---

## 🔁 Recommendation System

- Built using **Cosine Similarity** on phone feature averages
- User selects a phone model → gets top N similar phones
- Similarity based on: price, ratings, battery, camera, performance, design, display

---


## 📦 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
streamlit
jupyter
ipykernel




