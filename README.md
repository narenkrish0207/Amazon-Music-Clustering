🎵 Amazon Music Clustering — Highlighted README
🧠 Overview

With millions of songs across platforms like Amazon Music, manually tagging genres or moods is nearly impossible.

This project uses unsupervised machine learning to automatically cluster songs based on audio features — revealing hidden sound patterns, assisting in playlist generation, recommendations, and trend analysis.

⚡️ Key Highlights
🔥 What This Project Delivers

⭐ Automatic Song Clustering (K-Means)

⭐ Streamlit Dashboard for interactive exploration

⭐ PCA / t-SNE visualizations (2D & 3D)

⭐ Cluster Quality Metrics: Silhouette, DB Index

⭐ Feature Scaling Options: Standard / MinMax

⭐ Downloadable Results (CSV)

⭐ Visual Tools: Elbow curve, boxplots, heatmaps

🧩 Project Pipeline (Highlighted)
1️⃣ Data Exploration & Cleaning

✔ Load dataset: amazon_music_clusters_all_methods.csv
✔ Handle missing values & duplicates
✔ Pick numerical features
✔ Apply scaling: StandardScaler / MinMaxScaler

2️⃣ Feature Selection (Key Inputs)

Top features used for clustering:
danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms

3️⃣ Dimensionality Reduction (Highlight)

PCA → Understand linear relationships

t-SNE → Capture nonlinear structure

Both used for:
🔹 2D cluster separation
🔹 3D interactive visualization

4️⃣ Clustering Models

🎯 K-Means (primary algorithm)
📌 Optimal k determined via:

Elbow Method

Silhouette Score

📌 Optional extensions:

DBSCAN

Hierarchical Clustering

5️⃣ Evaluation Metrics (Important!)
Metric	Meaning	Goal
Silhouette Score	Cluster separation	⭐ Higher = Better
Davies-Bouldin Index	Intra-cluster similarity	⭐ Lower = Better
Inertia	Compactness	⭐ Lower = Better
📊 Visualization Highlights

📉 Elbow Curve — find optimal k

🌈 PCA 2D & 3D plots — understand separation

🔥 Heatmaps — compare average feature values

📦 Boxplots — analyze distributions per cluster

🧠 Final Analysis (Highlights)

Each track receives a cluster label, enabling interpretation like:

Cluster 0 → 🔊 High energy + loudness → Party / Workout

Cluster 1 → 🎸 High acousticness + valence → Chill / Relaxing

Cluster 2 → 🗣️ High speechiness → Podcasts / Rap

📁 Final dataset exported as:
amazon_music_clustered_data.csv

🖥️ Dashboard Overview (Highlighted)

The Streamlit app offers:

🎛️ Sidebar controls (scaling, cluster count, visualization mode)

📈 Real-time cluster metrics

🌐 Interactive 3D PCA (Plotly)

🧩 Cluster insights & feature comparisons

📥 CSV download

🧮 Tech Stack (Highlight)
Category	Tools / Libraries
Language	Python 3.x
Data Handling	pandas, numpy
ML	scikit-learn
Visualization	matplotlib, seaborn, plotly
App	Streamlit
Methods	PCA, KMeans, Silhouette, DB Index
💡 Business Use Cases (Highlighted)

🎧 Personalized playlists

🔍 Music recommendation engines

🎤 Artist & competitor similarity analysis

📈 Trend and market insights

🧠 Key Insights

✔ Clear, distinct clusters discovered

✔ PCA plots improved interpretability

✔ Strong model performance (high Silhouette)

✔ Streamlit app enhances data exploration
