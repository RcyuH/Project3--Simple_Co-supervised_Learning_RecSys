#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:12:05 2025

@author: rcyuh
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, item_embeddings, ratings_df):
        """
        Khởi tạo hệ thống gợi ý dựa trên nội dung.
        
        Parameters:
        - item_embeddings: dict {item_id: embedding_vector}
        - ratings_df: DataFrame chứa các cột ['userID', 'itemID', 'rating', 'timestamp']
        """
        self.item_embeddings = item_embeddings
        self.ratings_df = ratings_df
        self.user_profiles = self._build_user_profiles()
    
    def _build_user_profiles(self):
        """
        Xây dựng hồ sơ người dùng dựa trên trung bình có trọng số của embeddings các items mà họ đã đánh giá.
        """
        user_profiles = {}
        for user_id in self.ratings_df['userID'].unique():
            user_ratings = self.ratings_df[self.ratings_df['userID'] == user_id]
            vectors = []
            weights = []
            
            for _, row in user_ratings.iterrows():
                item_id, rating = row['itemID'], row['rating']
                if item_id in self.item_embeddings:
                    vectors.append(self.item_embeddings[item_id])
                    weights.append(rating)
            
            if vectors:
                user_profiles[user_id] = np.average(vectors, axis=0, weights=weights)
            else:
                user_profiles[user_id] = np.zeros_like(next(iter(self.item_embeddings.values())))
                # user_profiles[user_id] = np.zeros(len(next(iter(self.item_embeddings.values()))))
        
        return user_profiles
    
    def predict(self, user_id, item_id):
        """
        Dự đoán mức độ yêu thích của user đối với item dựa trên độ tương đồng cosine.
        """
        if user_id not in self.user_profiles or item_id not in self.item_embeddings:
            return np.nan  # Không thể dự đoán nếu thiếu dữ liệu
        
        user_vector = self.user_profiles[user_id].reshape(1, -1)
        item_vector = self.item_embeddings[item_id].reshape(1, -1)
        similarity = cosine_similarity(user_vector, item_vector)[0, 0]
        
        return similarity * 5  # Scale về thang điểm 5
    
    def recommend(self, user_id, top_n=5):
        """
        Gợi ý top N items cho một user dựa trên nội dung.
        """
        if user_id not in self.user_profiles:
            return []
        
        scores = []
        for item_id in self.item_embeddings:
            if item_id not in self.ratings_df[self.ratings_df['userID'] == user_id]['itemID'].values:
                predicted_rating = self.predict(user_id, item_id)
                if not np.isnan(predicted_rating):
                    scores.append((item_id, predicted_rating))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

if __name__ == "__main__":
    pass