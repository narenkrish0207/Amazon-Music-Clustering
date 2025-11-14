🎵 Amazon Music Clustering
🧠 Overview

With millions of songs available on platforms like Amazon Music, manually categorizing tracks by genre or mood is both time-consuming and impractical.

This project uses unsupervised machine learning to automatically group songs based on audio features such as tempo, energy, danceability, loudness, and more. These clusters help uncover hidden patterns that reflect genres, moods, or sound styles — all without requiring human-labeled data.

Through clustering, the project enables data-driven insights for playlist creation, music recommendation, listener segmentation, and market trend discovery.

🚀 Key Features

✅ Automated Song Grouping using K-Means Clustering

✅ Interactive Streamlit Dashboard

✅ Dimensionality Reduction with PCA / t-SNE for visualization

✅ Cluster Quality Evaluation (Silhouette & Davies-Bouldin Scores)

✅ Configurable Feature Scaling (StandardScaler / MinMaxScaler)

✅ Downloadable Clustered Dataset (CSV)

✅ Rich Visual Analytics: Elbow Curves, Heatmaps, Boxplots

🧩 Project Pipeline
1️⃣ Data Exploration & Cleaning

Load dataset: amazon_music_clusters_all_methods.csv

Handle missing values & duplicates

Drop irrelevant columns

Identify numerical features for clustering

Apply scaling using StandardScaler or MinMaxScaler

2️⃣ Feature Selection

Selected clustering features include:

danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms

3️⃣ Dimensionality Reduction

PCA for linear dimensionality reduction

t-SNE (optional) for nonlinear structure

Used for both 2D and 3D visualization of cluster separation

4️⃣ Clustering Algorithms

K-Means (primary algorithm)

Optimal k determined via Elbow & Silhouette methods

(Optional) DBSCAN for density-based clustering

(Optional) Hierarchical Clustering with dendrograms

5️⃣ Cluster Evaluation Metrics
Metric	Meaning	Goal
Silhouette Score	Measures separation quality	Higher = Better
Davies-Bouldin Index	Measures cluster similarity	Lower = Better
Inertia	Cluster compactness	Lower = Better
6️⃣ Visualization Tools

📉 Elbow Curve – determines optimal cluster count

🎨 PCA Scatter Plots (2D & 3D) – view cluster separation

🔥 Heatmaps – compare feature averages across clusters

📊 Boxplots – inspect feature distributions within clusters

7️⃣ Final Analysis

Assign final cluster labels to all tracks

Create interpretable cluster profiles such as:

Cluster 0 → High energy + loudness → Party / Workout Tracks

Cluster 1 → High acousticness + valence → Chill / Relaxed Music

Cluster 2 → High speechiness → Podcasts / Rap / Spoken Content

Export final dataset as amazon_music_clustered_data.csv

📈 Example Dashboard

The Streamlit app provides an intuitive interface for exploring clusters.

Key dashboard features:

Sidebar controls for scaling, cluster count, and visualization options

Real-time cluster evaluation metrics

PCA-based 2D and 3D visualizations (Plotly)

Feature comparisons across clusters

Downloadable clustered dataset

🧮 Tech Stack
Category	Tools / Libraries
Language	Python 3.x
Data Handling	pandas, NumPy
Machine Learning	scikit-learn
Visualization	matplotlib, seaborn, plotly
App Framework	Streamlit
Methods	PCA, KMeans, Silhouette Score, Davies-Bouldin Index
💡 Business Use Cases

🎧 Personalized Playlist Generation

🔍 Music Recommendation Systems

🎤 Artist & Competitor Analysis

📈 Market & Trend Insights

🧠 Insights & Results

Identified distinct musical clusters such as:

High-energy, loud tracks → Workout / Party

Acoustic, low-energy tracks → Chill / Relaxing

Speech-heavy tracks → Podcasts / Rap

PCA visualizations improved interpretability

Achieved strong cluster separation with high Silhouette scores

Delivered a fully interactive Streamlit exploration interface
