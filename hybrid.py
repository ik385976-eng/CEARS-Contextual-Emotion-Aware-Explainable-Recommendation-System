# """
# hybrid.py — Content-based similarity engine.
# Returns 5-tuples: (product_id, base_score, context_bonus, emotion_bonus, similarity_score)
# """

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# class HybridLayer:
#     def __init__(self, product_df: pd.DataFrame):
#         """
#         product_df must have at least: product_id, product_name
#         """
#         self.product_df = product_df.copy().drop_duplicates(subset='product_id')
#         self.product_df['product_name'] = self.product_df['product_name'].fillna('')

#         # Build TF-IDF matrix over product names
#         self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.product_df['product_name'])

#         # Map product_id → row index for fast lookup
#         self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_df['product_id'])}

#     def get_content_score(self, pid_a: str, pid_b: str) -> float:
#         """Cosine similarity between two products by name TF-IDF."""
#         idx_a = self.id_to_idx.get(pid_a)
#         idx_b = self.id_to_idx.get(pid_b)
#         if idx_a is None or idx_b is None:
#             return 0.0
#         score = cosine_similarity(
#             self.tfidf_matrix[idx_a],
#             self.tfidf_matrix[idx_b]
#         )[0][0]
#         return float(score)

#     def apply_hybrid(self, adjusted_recs: list, target_pid: str) -> list:
#         """
#         Accepts adjusted_recs — a list of tuples in ANY of these formats:
#             (pid, score)
#             (pid, score, c_bonus)
#             (pid, score, c_bonus, e_bonus)

#         Always returns a list of 5-tuples:
#             (pid, base_score, context_bonus, emotion_bonus, similarity_score)
#         """
#         final = []

#         for rec in adjusted_recs:
#             # Safely unpack regardless of incoming tuple length
#             if len(rec) == 2:
#                 pid, score = rec
#                 c_bonus, e_bonus = 0.0, 0.0
#             elif len(rec) == 3:
#                 pid, score, c_bonus = rec
#                 e_bonus = 0.0
#             elif len(rec) == 4:
#                 pid, score, c_bonus, e_bonus = rec
#             else:
#                 pid, score, c_bonus, e_bonus = rec[0], rec[1], rec[2], rec[3]

#             # Recompute similarity as the final "s_score"
#             s_score = self.get_content_score(pid, target_pid)

#             final.append((pid, float(score), float(c_bonus), float(e_bonus), float(s_score)))

#         # Sort by combined score (base + bonuses + similarity)
#         final.sort(key=lambda x: x[1] + x[2] + x[3] + x[4], reverse=True)
#         return final



# updated code
"""
hybrid.py — Content-based similarity engine.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridLayer:
    def __init__(self, product_df: pd.DataFrame):
        """
        product_df must have at least: product_id, product_name
        """
        self.product_df = product_df.copy().drop_duplicates(subset='product_id')
        self.product_df['product_name'] = self.product_df['product_name'].fillna('')

        # Build TF-IDF matrix over product names
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product_df['product_name'])

        # Map product_id → row index for fast lookup
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_df['product_id'])}

    def get_content_score(self, pid_a: str, pid_b: str) -> float:
        """Cosine similarity between two products by name TF-IDF."""
        idx_a = self.id_to_idx.get(pid_a)
        idx_b = self.id_to_idx.get(pid_b)
        if idx_a is None or idx_b is None:
            return 0.0
        score = cosine_similarity(
            self.tfidf_matrix[idx_a],
            self.tfidf_matrix[idx_b]
        )[0][0]
        return float(score)

    def apply_hybrid(self, adjusted_recs: list, target_pid: str) -> list:
        """
        Accepts adjusted_recs — a list of tuples in ANY of these formats:
            (pid, score)
            (pid, score, c_bonus)
            (pid, score, c_bonus, e_bonus)

        Always returns a list of 5-tuples:
            (pid, base_score, context_bonus, emotion_bonus, similarity_score)
        """
        final = []

        for rec in adjusted_recs:
            # Safely unpack regardless of incoming tuple length
            if len(rec) == 2:
                pid, score = rec
                c_bonus, e_bonus = 0.0, 0.0
            elif len(rec) == 3:
                pid, score, c_bonus = rec
                e_bonus = 0.0
            elif len(rec) >= 4:
                pid, score, c_bonus, e_bonus = rec[0], rec[1], rec[2], rec[3]
            else:
                continue

            # Recompute similarity as the final "s_score"
            s_score = self.get_content_score(pid, target_pid)

            final.append((pid, float(score), float(c_bonus), float(e_bonus), float(s_score)))

        # Sort by combined score (base + bonuses + similarity)
        final.sort(key=lambda x: x[1] + x[2] + x[3] + x[4], reverse=True)
        return final