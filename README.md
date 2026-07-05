# 📱 Mobile Product Segmentation & Recommendation System

## 📌 Overview

This project implements an end-to-end Machine Learning pipeline for mobile product analysis, customer segmentation, and recommendation using **K-Means Clustering** and **Cosine Similarity**. The application analyzes mobile product data, groups similar products into meaningful market segments, and recommends similar products through an interactive **Streamlit** web application.

---

## 🎯 Features

- 📊 Data preprocessing and cleaning
- 🔍 Exploratory Data Analysis (EDA)
- 🤖 K-Means clustering for product segmentation
- 📈 Cluster evaluation using Silhouette Score and Davies-Bouldin Score
- 💡 Product recommendation system using Cosine Similarity
- 🌐 Interactive Streamlit dashboard
- 📁 Export cleaned datasets and clustering results

---

## 🛠️ Technologies Used

- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```text
Mobile-Product-Segmentation-Recommendation-System/
│
├── app.py
├── Global Mobile Reviews Dataset.csv
├── requirements.txt
├── README.md
│
└── output/
    ├── cleaned_data.csv
    ├── cluster_analysis.csv
    └── cluster_assignments.csv
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Mobile-Product-Segmentation-Recommendation-System.git
cd Mobile-Product-Segmentation-Recommendation-System
```

### 2. Create a Virtual Environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

> **Note:** If the dataset is unavailable, the application automatically generates sample data for demonstration.

---

## 📱 Application Modules

### 🏠 Home
- Project overview
- Key metrics
- Quick navigation

### 📊 Data Overview
- Dataset preview
- Dataset statistics
- Missing value analysis

### 📈 Exploratory Data Analysis
- Brand distribution
- Price distribution
- Rating analysis
- Correlation heatmap
- Interactive visualizations

### 🤖 Product Segmentation
- K-Means clustering
- Adjustable number of clusters (2–10)
- Cluster visualization
- Business insights

### 💡 Recommendation System
- Product-based recommendations
- Brand-based recommendations
- Price-based recommendations
- Cosine Similarity search

### ℹ️ About
- Project description
- Workflow
- Documentation

---

## 🧠 Machine Learning Workflow

1. Load dataset
2. Data preprocessing
3. Handle missing values
4. Remove duplicates
5. Feature engineering
6. Feature scaling using StandardScaler
7. Product segmentation using K-Means Clustering
8. Cluster evaluation
9. Product recommendation using Cosine Similarity
10. Visualization and dashboard deployment

---

## 📊 Dataset Information

**Dataset:** Global Mobile Reviews Dataset

| Feature | Description |
|----------|-------------|
| Product_ID | Unique product ID |
| Brand | Mobile brand |
| Country | Country of origin |
| Price | Price (USD) |
| Rating | Product rating |
| Battery_mAh | Battery capacity |
| RAM_GB | RAM size |
| Storage_GB | Storage capacity |
| Screen_Size | Screen size |
| Camera_MP | Camera resolution |
| Rating_Count | Number of reviews |

---

## 📈 Clustering Evaluation

### Metrics

- **Silhouette Score** – Higher values indicate better cluster separation.
- **Davies-Bouldin Score** – Lower values indicate better clustering quality.

---

## 📊 Market Segments

| Segment | Description |
|----------|-------------|
| Premium | High price, advanced specifications, highly rated |
| Mid-Range | Balanced features and pricing |
| Budget | Affordable devices with essential features |

---

## 📁 Generated Outputs

The project automatically generates:

- `cleaned_data.csv`
- `cluster_analysis.csv`
- `cluster_assignments.csv`

---

## 💻 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📋 Project Deliverables

- ✅ Data Cleaning
- ✅ Exploratory Data Analysis
- ✅ Feature Engineering
- ✅ K-Means Clustering
- ✅ Cluster Evaluation
- ✅ Product Recommendation System
- ✅ Interactive Streamlit Dashboard
- ✅ CSV Output Files
- ✅ Project Documentation

---

## 🔮 Future Enhancements

- Deep Learning recommendation engine
- User authentication
- Product comparison feature
- Real-time database integration
- Cloud deployment
- Advanced filtering options

---



---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
