
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Mobile Product Analytics", layout="wide")

# Title
st.title("📱 Mobile Product Segmentation & Recommendation System")
st.markdown("---")

# Load and process data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('Mobile Reviews Sentiment null.csv', encoding='latin1')
    
    # Clean data
    df = df.drop_duplicates()
    
    num_cols = ['price_usd', 'rating', 'battery_life_rating', 'camera_rating', 
                'performance_rating', 'design_rating', 'display_rating', 'helpful_votes']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].fillna('Neutral')
    
    df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
    df = df.dropna(subset=['price_usd', 'brand', 'model'])
    
    # Feature engineering
    rating_cols = ['rating', 'battery_life_rating', 'camera_rating', 
                   'performance_rating', 'design_rating', 'display_rating']
    df['overall_score'] = df[rating_cols].mean(axis=1)
    
    df['price_segment'] = pd.cut(df['price_usd'], 
                                 bins=[0, 200, 400, 600, 10000], 
                                 labels=['Budget', 'Mid-Range', 'Premium', 'Ultra-Premium'])
    
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                             labels=['Young (18-25)', 'Adult (26-35)', 
                                     'Middle-Aged (36-50)', 'Senior (50+)'])
    
    return df

# Load data
with st.spinner("Loading and processing data..."):
    df = load_and_process_data()

# Prepare clustering
cluster_features = ['price_usd', 'rating', 'overall_score', 'battery_life_rating', 
                    'camera_rating', 'performance_rating', 'design_rating', 'display_rating']

X = df[cluster_features].fillna(df[cluster_features].median())

le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])
X['brand_encoded'] = df['brand_encoded']

df['price_segment_encoded'] = df['price_segment'].astype('category').cat.codes
X['price_segment_encoded'] = df['price_segment_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {0: 'Budget Value', 1: 'Mid-Range Balanced', 
                 2: 'Premium Performance', 3: 'Ultra-Premium Luxury'}
df['cluster_name'] = df['cluster'].map(cluster_names)

# Prepare recommendation system
features_for_sim = ['price_usd', 'rating', 'overall_score', 'battery_life_rating', 
                    'camera_rating', 'performance_rating', 'design_rating', 'display_rating']
rec_features = df[features_for_sim].fillna(df[features_for_sim].median())
scaler_rec = StandardScaler()
rec_features_scaled = scaler_rec.fit_transform(rec_features)
similarity_matrix = cosine_similarity(rec_features_scaled)
model_to_idx = {model: idx for idx, model in enumerate(df['model'])}

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA Dashboard", "🎯 Clustering Analysis", "🔍 Product Recommendations"])

# Tab 1: EDA Dashboard
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Brands")
        brand_counts = df['brand'].value_counts().head(10)
        fig = px.bar(x=brand_counts.index, y=brand_counts.values, 
                     title="Top 10 Mobile Brands",
                     labels={'x': 'Brand', 'y': 'Number of Reviews'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Rating Distribution")
        fig = px.histogram(df, x='rating', nbins=20, 
                          title="Distribution of User Ratings")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Analysis")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Review Sentiment Distribution", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Price vs Rating")
        fig = px.scatter(df, x='price_usd', y='rating', color='sentiment',
                         title="Price vs Rating by Sentiment",
                         labels={'price_usd': 'Price (USD)', 'rating': 'Rating'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Price Distribution by Segment")
        fig = px.box(df, x='price_segment', y='price_usd',
                     title="Price Distribution Across Segments")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Average Rating by Brand")
        brand_rating = df.groupby('brand')['rating'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=brand_rating.index, y=brand_rating.values,
                     title="Top 10 Brands by Average Rating")
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Clustering Analysis
with tab2:
    st.header("Product Segmentation Analysis")
    st.markdown("Using **K-Means Clustering** with 4 segments")
    
    # Cluster summary
    cluster_summary = df.groupby('cluster').agg({
        'price_usd': 'mean',
        'rating': 'mean',
        'overall_score': 'mean',
        'brand': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
        'model': 'count'
    }).round(2)
    cluster_summary.columns = ['Avg Price', 'Avg Rating', 'Avg Score', 'Top Brand', 'Count']
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Cluster interpretation
    st.subheader("🔍 Cluster Interpretation")
    
    for cluster in sorted(df['cluster'].unique()):
        with st.expander(f"{cluster_names[cluster]} (Cluster {cluster})"):
            cluster_data = df[df['cluster'] == cluster]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Price", f"${cluster_data['price_usd'].mean():.2f}")
                st.metric("Average Rating", f"{cluster_data['rating'].mean():.2f}/5.0")
                st.metric("Number of Products", len(cluster_data))
            with col2:
                st.metric("Overall Score", f"{cluster_data['overall_score'].mean():.2f}/5.0")
                st.metric("Top Brand", cluster_data['brand'].mode()[0])
                st.metric("Dominant Sentiment", cluster_data['sentiment'].mode()[0])
    
    # PCA Visualization
    st.subheader("📊 Cluster Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    
    fig = px.scatter(df, x='pca_x', y='pca_y', color=df['cluster'].astype(str),
                     title="Product Segments (PCA Visualization)",
                     hover_data=['brand', 'model', 'price_usd', 'rating'])
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Recommendation System
with tab3:
    st.header("🔍 Similar Product Recommendation Engine")
    st.markdown("Select a mobile phone to get top 5 similar product recommendations")
    
    # Model selection
    model_list = sorted(df['model'].unique())
    selected_model = st.selectbox("Select a Mobile Model:", model_list)
    
    if st.button("Get Recommendations", type="primary"):
        if selected_model in model_to_idx:
            idx = model_to_idx[selected_model]
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
            
            product_indices = [i[0] for i in sim_scores]
            recommendations = df.iloc[product_indices][['brand', 'model', 'price_usd', 
                                                        'rating', 'overall_score', 'cluster_name']].copy()
            recommendations['similarity_score'] = [i[1] for i in sim_scores]
            recommendations['similarity_score'] = recommendations['similarity_score'].apply(lambda x: f"{x:.2%}")
            
            st.success(f"Top 5 recommendations for **{selected_model}**:")
            st.dataframe(recommendations, use_container_width=True)
            
            # Visualization
            original = df[df['model'] == selected_model].iloc[0]
            comparison = pd.DataFrame({
                'Metric': ['Price (USD)', 'Rating', 'Overall Score'],
                'Selected': [original['price_usd'], original['rating'], original['overall_score']],
                'Avg Recommendation': [
                    recommendations['price_usd'].mean(),
                    recommendations['rating'].mean(),
                    recommendations['overall_score'].mean()
                ]
            })
            
            fig = px.bar(comparison, x='Metric', y=['Selected', 'Avg Recommendation'],
                        barmode='group', title="Selected vs Average Recommendation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model not found in database")

# Sidebar
st.sidebar.markdown("## 📱 Project Overview")
st.sidebar.markdown(f"""
**Mobile Product Segmentation & Recommendation System**

- **Total Records**: {df.shape[0]} reviews
- **Unique Brands**: {df['brand'].nunique()}
- **Unique Models**: {df['model'].nunique()}
- **Price Range**: ${df['price_usd'].min():.0f} - ${df['price_usd'].max():.0f}

**Techniques Used**:
- K-Means Clustering (4 Segments)
- Cosine Similarity Recommendations
- PCA for Visualization

**Segments Identified**:
- Budget Value
- Mid-Range Balanced
- Premium Performance
- Ultra-Premium Luxury
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Mobile Product Analytics Project")
st.sidebar.markdown("📅 2026")