# import numpy as np
# from sklearn.metrics import mean_squared_error

# def compute_rmse(model, test_df):
#     actual = []
#     predicted = []

#     for _, row in test_df.iterrows():
#         user = row['user_id']
#         item = row['product_id']
#         rating = row['rating']

#         pred = model.predict(user, item)

#         actual.append(rating)
#         predicted.append(pred)

#     rmse = np.sqrt(mean_squared_error(actual, predicted))
#     return rmse


# def precision_at_k(model, train_df, test_df, k=5, threshold=4):
#     precision_scores = []

#     users = test_df['user_id'].unique()

#     for user in users:
#         test_items = test_df[test_df['user_id'] == user]
#         relevant_items = test_items[test_items['rating'] >= threshold]['product_id'].tolist()

#         if len(relevant_items) == 0:
#             continue

#         candidate_items = train_df['product_id'].unique()

#         predictions = []
#         for item in candidate_items:
#             score = model.predict(user, item)
#             predictions.append((item, score))

#         predictions.sort(key=lambda x: x[1], reverse=True)
#         top_k_items = [item for item, _ in predictions[:k]]

#         relevant_in_top_k = len(set(top_k_items) & set(relevant_items))
#         precision = relevant_in_top_k / k
#         precision_scores.append(precision)

#     if len(precision_scores) == 0:
#         return 0

#     return np.mean(precision_scores)




# updated code


import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(model, test_df):
    """Compute RMSE for model predictions on test data"""
    actual = []
    predicted = []

    for _, row in test_df.iterrows():
        user = row['user_id']
        item = row['product_id']
        rating = row['rating']

        pred = model.predict(user, item)

        actual.append(rating)
        predicted.append(pred)

    if len(actual) == 0:
        return float('nan')
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse


def precision_at_k(model, train_df, test_df, k=5, threshold=4):
    """Compute precision@k for model recommendations"""
    precision_scores = []
    
    # Get unique users from test set
    users = test_df['user_id'].unique()
    
    # Get all unique products from training set
    all_products = train_df['product_id'].unique()

    for user in users:
        # Get relevant items (highly rated) for this user in test set
        test_items = test_df[test_df['user_id'] == user]
        relevant_items = test_items[test_items['rating'] >= threshold]['product_id'].tolist()

        if len(relevant_items) == 0:
            continue

        # Predict ratings for all products
        predictions = []
        for item in all_products:
            try:
                score = model.predict(user, item)
                predictions.append((item, score))
            except:
                continue

        # Sort by predicted score and get top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, _ in predictions[:k]]

        # Calculate precision
        relevant_in_top_k = len(set(top_k_items) & set(relevant_items))
        precision = relevant_in_top_k / k
        precision_scores.append(precision)

    if len(precision_scores) == 0:
        return 0.0

    return np.mean(precision_scores)