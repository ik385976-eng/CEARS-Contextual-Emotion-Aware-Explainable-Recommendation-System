import pandas as pd
import numpy as np
from model import SVDModel
from evaluation import compute_rmse, precision_at_k
from context import apply_context_adjustment
from emotion import apply_emotion_adjustment
from hybrid import HybridLayer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("review.csv")
df = df[['reviews.username', 'asins', 'name', 'reviews.rating']]
df = df.dropna()
df.columns = ['user_id', 'product_id', 'product_name', 'rating']

# 80/20 split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train baseline SVD
model = SVDModel(n_components=20)
model.train(train_df)

# Baseline RMSE
rmse = compute_rmse(model, test_df)
print("\nBaseline RMSE:", rmse)

# ---------- BASELINE PRECISION ----------
baseline_p5 = precision_at_k(model, train_df, test_df, k=5)
baseline_p10 = precision_at_k(model, train_df, test_df, k=10)

print("Baseline Precision@5:", baseline_p5)
print("Baseline Precision@10:", baseline_p10)

# ---------- CONTEXT EVALUATION ----------
context_selection = {"budget": "Low", "occasion": "Party", "time": "Night"}

def evaluate_layer(layer_name, adjust_function):
    precision_scores_5 = []
    precision_scores_10 = []

    users = test_df['user_id'].unique()

    for user in users:
        relevant = test_df[(test_df['user_id'] == user) &
                           (test_df['rating'] >= 4)]['product_id'].tolist()

        if len(relevant) == 0:
            continue

        candidate_items = train_df['product_id'].unique()
        predictions = []

        for item in candidate_items:
            score = model.predict(user, item)
            predictions.append((item, score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        predictions = predictions[:20]

        # Apply adjustment layer
        adjusted = adjust_function(predictions)

        top5 = [item for item, *_ in adjusted[:5]]
        top10 = [item for item, *_ in adjusted[:10]]

        precision_scores_5.append(len(set(top5) & set(relevant)) / 5)
        precision_scores_10.append(len(set(top10) & set(relevant)) / 10)

    print(f"\n{layer_name} Precision@5:", np.mean(precision_scores_5))
    print(f"{layer_name} Precision@10:", np.mean(precision_scores_10))


# Context Layer
evaluate_layer(
    "After Context",
    lambda preds: apply_context_adjustment(preds, context_selection)
)

# Emotion Layer
evaluate_layer(
    "After Emotion",
    lambda preds: apply_emotion_adjustment(
        apply_context_adjustment(preds, context_selection),
        "I am excited and happy"
    )
)

# Hybrid Layer
hybrid_layer = HybridLayer(df[['product_id', 'product_name']])

evaluate_layer(
    "After Hybrid",
    lambda preds: hybrid_layer.apply_hybrid(
        apply_emotion_adjustment(
            apply_context_adjustment(preds, context_selection),
            "I am excited and happy"
        ),
        train_df['product_id'].iloc[0]
    )
)