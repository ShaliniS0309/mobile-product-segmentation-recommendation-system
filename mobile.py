
# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Product Segmentation & Recommendation",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86C1;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📱 Mobile Analytics")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Overview", "📈 EDA", "🔬 Clustering", "🎯 Recommendations", "📋 About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Project Details:**
    - Domain: E-Commerce Analytics
    - Skills: Python, ML, Streamlit
    """
)

# ============================================================
# DATA LOADING AND CACHING
# ============================================================

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Try to load actual dataset
        df = pd.read_csv('Global Mobile Reviews Dataset.csv')
    except FileNotFoundError:
        # Create sample data if file not found
        np.random.seed(42)
        brands = ['Apple', 'Samsung', 'Xiaomi', 'OnePlus', 'Google', 'Sony', 'Nokia', 'Motorola']
        countries = ['USA', 'China', 'South Korea', 'India', 'Japan', 'UK', 'Germany']
        
        df = pd.DataFrame({
            'Product_ID': [f'P{1000+i}' for i in range(200)],
            'Brand': np.random.choice(brands, 200),
            'Country': np.random.choice(countries, 200),
            'Price': np.random.randint(100, 1500, 200) * 10,
            'Rating': np.round(np.random.uniform(3.0, 5.0, 200), 1),
            'Battery_mAh': np.random.choice([3000, 4000, 4500, 5000, 5500, 6000], 200),
            'RAM_GB': np.random.choice([4, 6, 8, 12, 16], 200),
            'Storage_GB': np.random.choice([64, 128, 256, 512, 1024], 200),
            'Screen_Size': np.round(np.random.uniform(5.5, 7.0, 200), 1),
            'Camera_MP': np.random.choice([12, 16, 20, 24, 48, 64, 108], 200),
            'Rating_Count': np.random.randint(100, 50000, 200)
        })
        df['Price'] = df['Price'] + (df['RAM_GB'] * 50) + (df['Storage_GB'] * 0.5)
    
    return df

@st.cache_data
def preprocess_data(df):
    """Clean and preprocess data"""
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        if not df_clean[col].mode().empty:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

@st.cache_data
def get_features(df):
    """Select features for clustering"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numerical_cols if 'id' not in col.lower()]
    return features[:8]

@st.cache_data
def perform_clustering(df, features, n_clusters=4):
    """Perform K-Means clustering"""
    X = df[features].copy()
    
    # Handle categorical
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    
    return df_clustered, X_scaled, kmeans, silhouette, davies_bouldin

@st.cache_data
def build_recommendation_system(df, features):
    """Build similarity-based recommendation system"""
    X = df[features].copy()
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    similarity_matrix = cosine_similarity(X_scaled)
    
    return similarity_matrix

# ============================================================
# LOAD DATA
# ============================================================

df = load_data()
df_clean = preprocess_data(df)
features = get_features(df_clean)

# ============================================================
# PAGE: HOME
# ============================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📱 Mobile Product Analytics Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Mobile Product Segmentation and Recommendation System**!
    This application helps analyze mobile products, segment them into meaningful categories,
    and provide personalized product recommendations.
    """)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Products", len(df_clean))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'Brand' in df_clean.columns:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Brands", len(df_clean['Brand'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'Price' in df_clean.columns:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Price", f"${df_clean['Price'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if 'Rating' in df_clean.columns:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Rating", f"{df_clean['Rating'].mean():.2f} ⭐")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Data Overview", use_container_width=True):
            st.session_state.page = "📊 Data Overview"
            st.rerun()
    
    with col2:
        if st.button("🔬 Run Clustering", use_container_width=True):
            st.session_state.page = "🔬 Clustering"
            st.rerun()
    
    with col3:
        if st.button("🎯 Get Recommendations", use_container_width=True):
            st.session_state.page = "🎯 Recommendations"
            st.rerun()
    
    # Features
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Exploratory Data Analysis**
        - Product distribution by brand and country
        - Price and rating analysis
        - Correlation analysis
        - Interactive visualizations
        """)
    
    with col2:
        st.markdown("""
        **🔬 Advanced Analytics**
        - Product segmentation using K-Means clustering
        - Cluster analysis and business insights
        - Similarity-based recommendation system
        - Interactive filtering and search
        """)

# ============================================================
# PAGE: DATA OVERVIEW
# ============================================================

elif page == "📊 Data Overview":
    st.markdown('<h1 class="main-header">📊 Data Overview</h1>', unsafe_allow_html=True)
    
    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df_clean.head(100), use_container_width=True)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df_clean.describe(), use_container_width=True)
    
    # Data Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df_clean.columns,
            'Type': df_clean.dtypes.values,
            'Non-Null': df_clean.count().values,
            'Unique': df_clean.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)
    
    with col2:
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df_clean.columns,
            'Missing Count': df_clean.isnull().sum().values,
            'Missing %': (df_clean.isnull().sum() / len(df_clean) * 100).values
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

# ============================================================
# PAGE: EDA
# ============================================================

elif page == "📈 EDA":
    st.markdown('<h1 class="main-header">📈 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    # Brand Distribution
    if 'Brand' in df_clean.columns:
        st.subheader("Brand Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            brand_counts = df_clean['Brand'].value_counts().head(10)
            fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values,
                        title='Top 10 Brands by Count',
                        color=brand_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(df_clean, names='Brand', title='Brand Distribution',
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    
    # Price Analysis
    if 'Price' in df_clean.columns:
        st.subheader("Price Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_clean, x='Price', nbins=30,
                              title='Price Distribution',
                              color_discrete_sequence=['#2E86C1'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_clean, y='Price', title='Price Box Plot',
                        color_discrete_sequence=['#E74C3C'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Rating Analysis
    if 'Rating' in df_clean.columns:
        st.subheader("Rating Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_clean, x='Rating', nbins=20,
                              title='Rating Distribution',
                              color_discrete_sequence=['#28B463'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Price' in df_clean.columns:
                fig = px.scatter(df_clean, x='Price', y='Rating',
                                title='Price vs Rating',
                                color='Brand' if 'Brand' in df_clean.columns else None,
                                hover_data=['Product_ID'] if 'Product_ID' in df_clean.columns else None)
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        fig = px.imshow(df_clean[numerical_cols].corr(),
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: CLUSTERING
# ============================================================

elif page == "🔬 Clustering":
    st.markdown('<h1 class="main-header">🔬 Product Segmentation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This section uses **K-Means Clustering** to segment mobile products into distinct groups.
    Adjust the parameters below to explore different segmentation strategies.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    
    with col2:
        selected_features = st.multiselect(
            "Select Features for Clustering",
            features,
            default=features[:4]
        )
    
    if st.button("🔬 Run Clustering", use_container_width=True, type="primary"):
        if not selected_features:
            st.warning("Please select at least one feature for clustering!")
        else:
            with st.spinner("Performing clustering..."):
                # Perform clustering
                df_clustered, X_scaled, kmeans, silhouette, davies_bouldin = perform_clustering(
                    df_clean, selected_features, n_clusters
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col2:
                    st.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}")
                with col3:
                    st.metric("Clusters", n_clusters)
                
                # Cluster Distribution
                st.subheader("Cluster Distribution")
                fig = px.pie(df_clustered, names='Cluster', title='Product Distribution by Cluster',
                            hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
                # PCA Visualization
                st.subheader("Cluster Visualization (PCA)")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = df_clustered['Cluster']
                
                fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                                title='2D Cluster Visualization',
                                color_continuous_scale='Viridis',
                                hover_data={'Cluster': True})
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster Analysis
                st.subheader("Cluster Analysis")
                cluster_summary = df_clustered.groupby('Cluster')[selected_features].mean()
                st.dataframe(cluster_summary, use_container_width=True)
                
                # Cluster Insights
                st.subheader("Business Insights")
                
                for cluster_id in sorted(df_clustered['Cluster'].unique()):
                    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                    
                    with st.expander(f"📊 Cluster {cluster_id} - {len(cluster_data)} Products"):
                        col1, col2, col3 = st.columns(3)
                        
                        if 'Price' in cluster_data.columns:
                            avg_price = cluster_data['Price'].mean()
                            overall_price = df_clustered['Price'].mean()
                            price_diff = ((avg_price - overall_price) / overall_price * 100)
                            
                            with col1:
                                st.metric("Avg Price", f"${avg_price:.2f}", f"{price_diff:.1f}%")
                        
                        if 'Rating' in cluster_data.columns:
                            avg_rating = cluster_data['Rating'].mean()
                            overall_rating = df_clustered['Rating'].mean()
                            rating_diff = ((avg_rating - overall_rating) / overall_rating * 100)
                            
                            with col2:
                                st.metric("Avg Rating", f"{avg_rating:.2f}⭐", f"{rating_diff:.1f}%")
                        
                        with col3:
                            size_pct = len(cluster_data) / len(df_clustered) * 100
                            st.metric("Market Share", f"{size_pct:.1f}%")
                        
                        st.write("**Feature Averages:**")
                        st.dataframe(cluster_data[selected_features].mean().to_frame('Average'))
                        
                        # Identify segment type
                        if 'Price' in cluster_data.columns:
                            price_q = df_clustered['Price'].quantile([0.25, 0.75])
                            avg_price = cluster_data['Price'].mean()
                            
                            if avg_price > price_q[0.75]:
                                st.success("💰 **Premium Segment** - High-end products")
                            elif avg_price < price_q[0.25]:
                                st.info("💸 **Budget Segment** - Affordable products")
                            else:
                                st.warning("📊 **Mid-Range Segment** - Balanced products")

# ============================================================
# PAGE: RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":
    st.markdown('<h1 class="main-header">🎯 Product Recommendations</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Get personalized product recommendations based on similarity analysis.
    The system finds products with similar features, prices, and ratings.
    """)
    
    # Build similarity matrix
    with st.spinner("Building recommendation system..."):
        similarity_matrix = build_recommendation_system(df_clean, features)
    
    # Recommendation Type
    rec_type = st.radio(
        "Select Recommendation Type",
        ["By Product", "By Price Range", "By Brand"],
        horizontal=True
    )
    
    if rec_type == "By Product":
        st.subheader("Find Similar Products")
        
        # Select product
        product_options = df_clean.index.tolist()
        
        # Create display labels for products
        def format_product_label(idx):
            label = f"Product {idx}"
            if 'Brand' in df_clean.columns:
                label += f" - {df_clean.iloc[idx]['Brand']}"
            if 'Price' in df_clean.columns:
                label += f" - ${df_clean.iloc[idx]['Price']:.2f}"
            if 'Rating' in df_clean.columns:
                label += f" - {df_clean.iloc[idx]['Rating']}⭐"
            return label
        
        selected_idx = st.selectbox(
            "Select a Product",
            product_options,
            format_func=format_product_label
        )
        
        n_recs = st.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("🔍 Find Similar Products", use_container_width=True):
            # Display selected product
            st.subheader("Selected Product")
            st.dataframe(df_clean.iloc[selected_idx:selected_idx+1], use_container_width=True)
            
            # Get recommendations
            scores = list(enumerate(similarity_matrix[selected_idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in scores[1:n_recs+1]]
            top_scores = [i[1] for i in scores[1:n_recs+1]]
            
            recommendations = df_clean.iloc[top_indices].copy()
            recommendations['Similarity_Score'] = top_scores
            
            st.subheader(f"Top {n_recs} Similar Products")
            
            # Display with color coding
            def color_similarity(val):
                if val > 0.8:
                    return 'background-color: #90EE90'
                elif val > 0.6:
                    return 'background-color: #FFD700'
                else:
                    return 'background-color: #FFB6C1'
            
            try:
                styled_recs = recommendations.style.map(color_similarity, subset=['Similarity_Score'])
                st.dataframe(styled_recs, use_container_width=True)
            except AttributeError:
                st.dataframe(recommendations, use_container_width=True)
            
            # Show similarity scores as a bar chart
            st.subheader("Similarity Scores")
            fig = px.bar(
                recommendations,
                x=recommendations.index,
                y='Similarity_Score',
                title='Product Similarity Scores',
                color='Similarity_Score',
                color_continuous_scale='Viridis',
                labels={'index': 'Product Index'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif rec_type == "By Price Range":
        st.subheader("Recommendations by Price Range")
        
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price", value=float(df_clean['Price'].min()))
        with col2:
            max_price = st.number_input("Max Price", value=float(df_clean['Price'].max()))
        
        n_recs = st.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("🔍 Find Products in Range", use_container_width=True):
            # Filter by price range
            filtered = df_clean[(df_clean['Price'] >= min_price) & (df_clean['Price'] <= max_price)]
            
            if len(filtered) == 0:
                st.warning("No products found in this price range!")
            else:
                st.success(f"Found {len(filtered)} products in this price range")
                
                # Get random product from range
                selected_idx = filtered.index[np.random.randint(0, len(filtered))]
                
                st.subheader("Selected Product")
                st.dataframe(df_clean.iloc[selected_idx:selected_idx+1], use_container_width=True)
                
                # Get recommendations
                scores = list(enumerate(similarity_matrix[selected_idx]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                top_indices = [i[0] for i in scores[1:n_recs+1]]
                top_scores = [i[1] for i in scores[1:n_recs+1]]
                
                recommendations = df_clean.iloc[top_indices].copy()
                recommendations['Similarity_Score'] = top_scores
                
                st.subheader(f"Top {n_recs} Similar Products")
                st.dataframe(recommendations, use_container_width=True)
                
                # Visualize recommendations
                if 'Price' in recommendations.columns:
                    fig = px.bar(recommendations, x=recommendations.index, y='Price',
                                title='Price Comparison of Recommendations',
                                color='Similarity_Score' if 'Similarity_Score' in recommendations.columns else None,
                                color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
    
    elif rec_type == "By Brand":
        if 'Brand' in df_clean.columns:
            st.subheader("Recommendations by Brand")
            
            selected_brand = st.selectbox("Select Brand", df_clean['Brand'].unique())
            n_recs = st.slider("Number of Recommendations", 3, 10, 5)
            
            if st.button("🔍 Find Similar Brands", use_container_width=True):
                # Filter by brand
                brand_products = df_clean[df_clean['Brand'] == selected_brand]
                st.success(f"Found {len(brand_products)} products from {selected_brand}")
                
                if len(brand_products) > 0:
                    selected_idx = brand_products.index[np.random.randint(0, len(brand_products))]
                    
                    st.subheader("Selected Product")
                    st.dataframe(df_clean.iloc[selected_idx:selected_idx+1], use_container_width=True)
                    
                    # Get recommendations
                    scores = list(enumerate(similarity_matrix[selected_idx]))
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    top_indices = [i[0] for i in scores[1:n_recs+1]]
                    top_scores = [i[1] for i in scores[1:n_recs+1]]
                    
                    recommendations = df_clean.iloc[top_indices].copy()
                    recommendations['Similarity_Score'] = top_scores
                    
                    st.subheader(f"Top {n_recs} Similar Products")
                    st.dataframe(recommendations, use_container_width=True)

# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "📋 About":
    st.markdown('<h1 class="main-header">📋 About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📱 Mobile Product Segmentation and Recommendation System
    
    This project implements a complete data analytics and machine learning pipeline
    for analyzing mobile products, segmenting them into meaningful groups, and
    providing personalized recommendations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - Clean and preprocess mobile product data
        - Perform Exploratory Data Analysis (EDA)
        - Apply clustering techniques for segmentation
        - Build similarity-based recommendation system
        - Develop interactive Streamlit application
        """)
    
    with col2:
        st.subheader("🛠️ Skills & Technologies")
        st.markdown("""
        - Python & Pandas
        - Scikit-learn (K-Means, PCA)
        - Plotly & Matplotlib
        - Streamlit Web Framework
        - Machine Learning (Unsupervised)
        - Recommendation Systems
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Dataset")
    st.markdown("""
    **Global Mobile Reviews Dataset** containing:
    - Product specifications and features
    - Pricing information
    - User ratings and reviews
    - Brand and country information
    - Technical specifications (RAM, Storage, Battery, Camera)
    """)
    
    st.markdown("---")
    
    st.subheader("📈 Project Pipeline")
    
    pipeline_steps = [
        "1. 📥 Data Collection",
        "2. 🧹 Data Preprocessing",
        "3. 📊 Exploratory Data Analysis",
        "4. 🔧 Feature Engineering",
        "5. 🔬 Clustering (K-Means)",
        "6. 📈 Cluster Analysis",
        "7. 🎯 Recommendation System",
        "8. 🖥️ Streamlit Application"
    ]
    
    for step in pipeline_steps:
        st.markdown(f"- {step}")
    
    st.markdown("---")
    
    st.subheader("📋 Project Deliverables")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ✅ Cleaned dataset (CSV)
        - ✅ Python scripts
        - ✅ Clustering model
        - ✅ Recommendation system
        """)
    
    with col2:
        st.markdown("""
        - ✅ Interactive Streamlit app
        - ✅ Visualizations
        - ✅ Business insights
        - ✅ Project documentation
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Mobile Product Analytics System
    </div>
    """,
    unsafe_allow_html=True
)