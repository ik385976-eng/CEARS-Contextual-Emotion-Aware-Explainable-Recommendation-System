# import pandas as pd
# import numpy as np
# from sklearn.decomposition import TruncatedSVD
# from sklearn.model_selection import train_test_split
# import pickle

# class SVDModel:
#     def __init__(self, n_components=20):
#         self.n_components = n_components
#         self.svd = TruncatedSVD(n_components=n_components, random_state=42)
#         self.user_item_matrix = None
#         self.user_ids = None
#         self.item_ids = None
#         self.reconstructed_matrix = None

#     def train(self, df):
#         # 80/20 split
#         train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#         # Create user-item matrix from train only
#         self.user_item_matrix = train_df.pivot_table(
#             index='user_id',
#             columns='product_id',
#             values='rating'
#         ).fillna(0)

#         self.user_ids = self.user_item_matrix.index
#         self.item_ids = self.user_item_matrix.columns

#         # Fit SVD
#         reduced_matrix = self.svd.fit_transform(self.user_item_matrix)

#         # Reconstruct matrix
#         self.reconstructed_matrix = np.dot(reduced_matrix, self.svd.components_)

#         return train_df, test_df

#     def predict(self, user_id, product_id):
#         if user_id in self.user_ids and product_id in self.item_ids:
#             user_idx = self.user_ids.get_loc(user_id)
#             item_idx = self.item_ids.get_loc(product_id)
#             return self.reconstructed_matrix[user_idx, item_idx]
#         return 0

#     def save_model(self, path="model.pkl"):
#         with open(path, "wb") as f:
#             pickle.dump(self, f)

#     @staticmethod
#     def load_model(path="model.pkl"):
#         with open(path, "rb") as f:
#             return pickle.load(f)




# updated code

import pickle
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split

class SVDModel:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.model = SVD(n_factors=n_components, random_state=42)
        self.reader = Reader(rating_scale=(1, 5))
    
    def train(self, df):
        """Train SVD model on user-item rating data"""
        # Prepare data for Surprise
        surprise_data = Dataset.load_from_df(
            df[['user_id', 'product_id', 'rating']], 
            self.reader
        )
        
        # Split into train and test
        trainset, testset = surprise_split(surprise_data, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(trainset)
        
        # Create test dataframe for evaluation
        test_df = pd.DataFrame(testset, columns=['user_id', 'product_id', 'rating'])
        
        # Return train and test dataframes
        train_df = df[~df.index.isin(test_df.index)] if len(test_df) > 0 else df
        return train_df, test_df
    
    def predict(self, user_id, product_id):
        """Predict rating for a user-product pair"""
        try:
            pred = self.model.predict(str(user_id), str(product_id)).est
            return float(pred)
        except:
            return 3.0  # Default neutral rating
    
    def save_model(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)