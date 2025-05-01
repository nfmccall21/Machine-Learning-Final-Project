import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()


def embed_caption(caption):
    """Embed caption using DistilBERT (mean pooling)."""
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def load_features(post_path, json_dir):
    """Load post metadata and BERT embeddings into a DataFrame."""
    df_tsv = pd.read_csv(post_path, sep='\t', header=None)
    json_files = df_tsv[3].dropna().tolist()
    data_points = []
    captions = []
    json_indices = []

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                continue

        edges = data.get("edge_media_to_caption", {}).get("edges", [])
        caption = edges[0]["node"]["text"] if edges else ""
        width = data.get("dimensions", {}).get("width", 1)
        height = data.get("dimensions", {}).get("height", 1)

        try:
            caption_embed = embed_caption(caption)
        except Exception as e:
            print(f"Skipping caption due to embedding error: {e}")
            continue

        base_features = {
            "like_count": data.get("edge_media_preview_like", {}).get("count", 0),
            "comment_count": data.get("edge_media_to_comment", {}).get("count", 0),
            "caption_length": len(caption),
            "hashtag_count": caption.count("#"),
            "mention_count": caption.count("@"),
            "aspect_ratio": width / height if height else 1
        }

        for i, val in enumerate(caption_embed):
            base_features[f"bert_{i}"] = val

        data_points.append(base_features)
        captions.append(caption)
        json_indices.append(json_file)

    df = pd.DataFrame(data_points)
    df["caption"] = captions
    df.index = json_indices
    return df


def train_and_evaluate_model(df):
    """Train and evaluate a linear regression model with feature scaling."""
    df = df[df['like_count'] > 0]
    X = df.drop(columns=['like_count', 'caption', 'caption_cluster'], errors='ignore')
    y = df['like_count']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    coeffs = dict(zip(X.columns, model.coef_))

    return rmse, r2, coeffs


def cluster_caption_embeddings(df, n_clusters=5):
    """Cluster BERT embeddings and add cluster labels to DataFrame."""
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    X_bert = df[bert_cols]

    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(X_bert)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['caption_cluster'] = kmeans.fit_predict(X_reduced)

    return df, X_reduced


def visualize_clusters(X_reduced, labels):
    """Visualize caption clusters in 2D space using t-SNE."""
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_reduced)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title("Caption Embeddings Clustered via KMeans")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster")
    plt.show()


def print_cluster_samples(df):
    for cluster_id in sorted(df['caption_cluster'].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        sample = df[df['caption_cluster'] == cluster_id].sample(3, random_state=42)
        for text in sample['caption']:
            print(f"• {text[:100]}...")


#main

post_path = 'post_info.txt'
json_dir = 'json_data'
# post_path = 'post_info_truncated.tsv'
# json_dir = 'truncated_jsons'

df_features = load_features(post_path, json_dir)
df_features, X_reduced = cluster_caption_embeddings(df_features, n_clusters=5)
visualize_clusters(X_reduced, df_features['caption_cluster'])

rmse, r2, coeffs = train_and_evaluate_model(df_features)

print("\n--- MODEL RESULTS ---")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print("Top 10 Coefficients (by magnitude):")
for key, value in sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"{key}: {value:.4f}")

print("\n--- CLUSTER LIKE STATS ---")
print(df_features.groupby("caption_cluster")["like_count"].describe())

print_cluster_samples(df_features)


#count and proportion of posts per caption cluster
cluster_counts = df_features['caption_cluster'].value_counts().sort_index()
cluster_proportions = df_features['caption_cluster'].value_counts(normalize=True).sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title("Number of Posts per Caption Cluster")
plt.xlabel("Caption Cluster")
plt.ylabel("Post Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 7))
cluster_proportions.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='tab10')
plt.title("Proportion of Posts by Caption Cluster")
plt.ylabel("")
plt.tight_layout()
plt.show()
