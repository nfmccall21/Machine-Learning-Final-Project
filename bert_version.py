import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
from transformers import DistilBertTokenizer, DistilBertModel

# import logging
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

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

    return pd.DataFrame(data_points)


def train_and_evaluate_model(df):
    """Train and evaluate a linear regression model with feature scaling."""
    df = df[df['like_count'] > 0]  #remove 0-like posts...perhaps we want this
    X = df.drop(columns=['like_count'])
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


post_path = 'post_info.txt'   
json_dir = 'json_data'          

df_features = load_features(post_path, json_dir)
rmse, r2, coeffs = train_and_evaluate_model(df_features)

print("\n--- MODEL RESULTS ---")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print("Top 10 Coefficients (by magnitude):")
for key, value in sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"{key}: {value:.4f}")
