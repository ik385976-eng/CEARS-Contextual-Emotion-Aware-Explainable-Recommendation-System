import pandas as pd
from model import SVDModel
from evaluation import compute_rmse, precision_at_k

# Load dataset
df = pd.read_csv("review.csv")

df = df[['reviews.username', 'asins', 'reviews.rating']]
df = df.dropna()
df.columns = ['user_id', 'product_id', 'rating']

# Train model
model = SVDModel(n_components=20)
train_df, test_df = model.train(df)

# Save trained model
model.save_model("model.pkl")

# Baseline Evaluation
rmse = compute_rmse(model, test_df)
precision5 = precision_at_k(model, train_df, test_df, k=5)
precision10 = precision_at_k(model, train_df, test_df, k=10)

print("Baseline RMSE:", rmse)
print("Baseline Precision@5:", precision5)
print("Baseline Precision@10:", precision10)