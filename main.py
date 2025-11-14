import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px

from functools import lru_cache

@st.cache_data
def load_data():
    df = pd.read_csv("D:\Guvi Projects\Myproject 4\amazon_music_clusters_all_methods.csv")
    return df

@st.cache_data
def run_pca(df):
    from sklearn.decomposition import PCA
    numeric_cols = ['danceability', 'energy', 'valence', 'tempo']
    pca = PCA(n_components=2)
    result = pca.fit_transform(df[numeric_cols])
    return result

@st.cache_data
def compute_metrics(X, labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        return sil, db
    return None, None



# ------------------------------------------------------------
# 1️⃣ PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Amazon Music Cluster Visualizer", layout="wide")

st.title("🎧 Amazon Music Clustering Dashboard")
st.markdown("""
Explore how songs are grouped based on their audio and artist features using  
*K-Means, **DBSCAN, and **Hierarchical Clustering*.
---
""")

# ------------------------------------------------------------
# 2️⃣ LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("D:\Guvi Projects\Myproject 4\amazon_music_clusters_all_methods.csv")
    return df

df = load_data()

st.sidebar.header("⚙ Filter Options")
method = st.sidebar.selectbox("Select Clustering Method", ["K-Means", "DBSCAN", "Hierarchical"])
cluster_col = {
    "K-Means": "cluster",
    "DBSCAN": "cluster_dbscan",
    "Hierarchical": "cluster_hc"
}[method]

st.sidebar.markdown("""
### 💡 Business Use Cases
- Personalized Playlist Curation  
- Improved Song Discovery  
- Artist Analysis  
- Market Segmentation
""")


# Drop NaN clusters (for HC/DBSCAN)
df_vis = df.dropna(subset=[cluster_col]).copy()
df_vis[cluster_col] = df_vis[cluster_col].astype(int)

# ------------------------------------------------------------
# 3️⃣ DATA OVERVIEW
# ------------------------------------------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df_vis.head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Number of Songs", len(df_vis))
col2.metric("Number of Clusters", df_vis[cluster_col].nunique())
col3.metric("Genres", df_vis['genres'].nunique())

# ------------------------------------------------------------
# 4️⃣ CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader(f"🎨 {method} Cluster Distribution")
cluster_counts = df_vis[cluster_col].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette="viridis")
ax.set_xlabel("Cluster ID")
ax.set_ylabel("Number of Songs")
ax.set_title(f"{method} – Cluster Size Distribution")
st.pyplot(fig)

# ------------------------------------------------------------
# 5️⃣ FEATURE COMPARISON PER CLUSTER
# ------------------------------------------------------------
st.subheader("🎼 Average Feature Values per Cluster")

features = ['danceability', 'energy', 'valence', 'tempo']
cluster_profile = df_vis.groupby(cluster_col)[features].mean()

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(cluster_profile, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
ax.set_title(f"{method} – Feature Profile Heatmap")
st.pyplot(fig)


# ------------------------------------------------------------
# 6️⃣ PCA VISUALIZATION (2D)
# ------------------------------------------------------------
st.subheader("🌀 PCA Visualization (2D Projection)")

# Run PCA on numeric features
numeric_cols = ['danceability', 'energy', 'valence', 'tempo']
pca_result = run_pca(df_vis)
 

df_vis['pca1'], df_vis['pca2'] = pca_result[:,0], pca_result[:,1]

fig_pca = px.scatter(
    df_vis, x='pca1', y='pca2',
    color=df_vis[cluster_col].astype(str),
    hover_data=['genres'],
    title=f"{method} – PCA Cluster Visualization",
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig_pca, use_container_width=True)


# ------------------------------------------------------------
# 🔍 5C️⃣ FEATURE CORRELATION ANALYSIS
# ------------------------------------------------------------
st.subheader("🔍 Feature Correlation Heatmap")

corr = df_vis[['danceability', 'energy', 'valence', 'tempo']].corr()

fig_corr, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Feature Correlation Matrix")
st.pyplot(fig_corr)

st.caption("High correlation values indicate similar musical behavior between features.")



# ------------------------------------------------------------
# 7️⃣ GENRE CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader("🎶 Top Genres by Cluster")

top_genres = (
    df_vis.groupby(['genres', cluster_col])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .head(15)
)

fig_genres = px.bar(
    top_genres, 
    x='genres', y='count', color=cluster_col,
    title=f"Top Genres across {method} Clusters",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_genres, use_container_width=True)


# ------------------------------------------------------------
# 🧠 7B️⃣ CLUSTER INTERPRETATION SUMMARY
# ------------------------------------------------------------
st.subheader("🧠 Cluster Interpretation Summary")

summary = df_vis.groupby(cluster_col)[['danceability', 'energy', 'valence', 'tempo']].mean().round(2)

for c_id, row in summary.iterrows():
    desc = []
    if row['energy'] > 0.7: desc.append("high-energy")
    if row['danceability'] > 0.7: desc.append("danceable")
    if row['valence'] > 0.6: desc.append("positive/happy")
    if row['tempo'] > 120: desc.append("fast-tempo")
    if not desc: desc.append("mellow or balanced")
    st.markdown(f"**Cluster {c_id}:** likely contains {', '.join(desc)} tracks.")



# ------------------------------------------------------------
# 8️⃣ DOWNLOAD FILTERED RESULTS
# ------------------------------------------------------------
st.subheader("⬇ Download Filtered Cluster Data")
csv = df_vis.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download CSV",
    csv,
    f"music_clusters_{method.lower()}.csv",
    "text/csv",
    key='download-csv'
)

st.markdown("---")
st.caption("Built with ❤ using Streamlit | Dataset: Amazon Music Single-Genre Artists")


# ------------------------------------------------------------
# 🧾 9️⃣ CLUSTER QUALITY COMPARISON ACROSS METHODS (Optimized)
# ------------------------------------------------------------
st.subheader("📊 Cluster Quality Comparison Across Methods")

@st.cache_data
def compute_quality_summary(df):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    results = []
    for method, col in {
        "K-Means": "cluster",
        "DBSCAN": "cluster_dbscan",
        "Hierarchical": "cluster_hc"
    }.items():
        df_temp = df.dropna(subset=[col])
        X = df_temp[['danceability', 'energy', 'valence', 'tempo']]
        labels = df_temp[col]
        if len(set(labels)) > 1:
            # Subsample to speed up
            if len(X) > 2000:
                X = X.sample(2000, random_state=42)
                labels = labels.loc[X.index]
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            results.append([method, sil, dbi])
    return pd.DataFrame(results, columns=["Method", "Silhouette", "Davies–Bouldin"])

metrics_df = compute_quality_summary(df)

st.dataframe(metrics_df.style.format({
    "Silhouette": "{:.3f}",
    "Davies–Bouldin": "{:.3f}"
}))

st.caption("""
✅ Silhouette Score → Higher = Better  
✅ Davies–Bouldin Index → Lower = Better  
(Note: metrics computed on 2000-song subsample for speed)
""")
