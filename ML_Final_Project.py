import os
import shutil
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


post_path = '/Users/nataliemccall/Downloads/Machine Learning/Final Project/post_info_truncated.tsv'
json_dir = '/Users/nataliemccall/Downloads/Machine Learning/Final Project/truncated_jsons'


def copy_over_json(metadata_file):
    source_dir = '/Users/nataliemccall/Downloads/Machine Learning/Final Project/json'
    dest_dir = '/Users/nataliemccall/Downloads/Machine Learning/Final Project/truncated_jsons'
    os.makedirs(dest_dir, exist_ok=True)

    json_files_to_extract = []
    with open(metadata_file, 'r') as file:
        for line in file:
           parts = line.strip().split('\t')
           if len(parts) >= 5:
            #    json_files_to_extract.append(parts[3])
               src_path = os.path.join(source_dir, parts[3])
               shutil.copy2(src_path, os.path.join(dest_dir, parts[3]))


    # for json_file in json_files_to_extract:
    #     src_path = os.path.join(source_dir, json_file)
    #     if os.path.exists(src_path):
    #         shutil.copy2(src_path, os.path.join(dest_dir, json_file))

def load_features(post_path, json_dir):
    df_tsv = pd.read_csv(post_path, sep='\t', header=None)
    json_files = df_tsv[3].dropna().tolist() #should be csv? works like this but investigate
    labels = []

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
        width = data.get("dimensions", {}).get("width", 1) #not sure if these will be needed but extracting for now
        height = data.get("dimensions", {}).get("height", 1)
       

        post = {
            "like_count": data.get("edge_media_preview_like", {}).get("count", 0),
            "comment_count": data.get("edge_media_to_comment", {}).get("count", 0),
            "caption_length": len(caption),
            "hashtag_count": caption.count("#"),
            "mention_count": caption.count("@"),
            "aspect_ratio": width / height if height else 1,
        }
        labels.append(post)

    return pd.DataFrame(labels)

def train_and_evaluate_model(df):
    df = df[df['like_count'] > 0]
    X = df.drop(columns=['like_count'])
    y = df['like_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    coeffs = dict(zip(X.columns, model.coef_))

    return rmse, r2, coeffs


def main():
    # metadata_file = '/Users/nataliemccall/Downloads/Machine Learning/Final Project/post_info_truncated.tsv'
    # copy_over_json(metadata_file=metadata_file)

    df_features = load_features(post_path, json_dir)
    rmse, r2, coeffs = train_and_evaluate_model(df_features)

    print("RMSE:", rmse)
    print("R² Score:", r2)
    print("Coefficients:", coeffs)

if __name__ == "__main__":
    main()