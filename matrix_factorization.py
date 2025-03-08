#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:12:31 2025

@author: rcyuh
"""

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from preprocessing import preprocessing_meta_data, preprocessing_matrix

class MatrixFactorizationRecommender:
    def __init__(self, ratings_df, n_factors=50):
        """
        Khởi tạo hệ thống gợi ý dựa trên Matrix Factorization (SVD).
        
        Parameters:
        - ratings_df: DataFrame chứa các cột ['userID', 'itemID', 'rating', 'timestamp']
        - n_factors: Số lượng latent factors trong mô hình SVD.
        """
        self.ratings_df = ratings_df
        self.n_factors = n_factors
        self.model = self._train_model()
    
    def _train_model(self):
        """
        Huấn luyện mô hình SVD trên tập dữ liệu đầu vào.
        """
        reader = Reader(rating_scale=(self.ratings_df['rating'].min(), self.ratings_df['rating'].max()))
        dataset = Dataset.load_from_df(self.ratings_df[['userID', 'itemID', 'rating']], reader)
        trainset = dataset.build_full_trainset()
        
        model = SVD(n_factors=self.n_factors)
        model.fit(trainset)
        
        return model
    
    def predict(self, user_id, item_id):
        """
        Dự đoán mức độ yêu thích của user đối với item.
        """
        return self.model.predict(user_id, item_id).est
    
    def recommend(self, user_id, top_n=5):
        """
        Gợi ý top N items cho một user dựa trên dự đoán của mô hình.
        """
        all_items = set(self.ratings_df['itemID'].unique())
        rated_items = set(self.ratings_df[self.ratings_df['userID'] == user_id]['itemID'].values)
        
        candidates = all_items - rated_items  # Chỉ xét các items chưa được đánh giá
        scores = [(item, self.predict(user_id, item)) for item in candidates]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
if __name__ == "__main__":
    # Constant
    cols_needed = ["problem_id", "sequence_id", "skill", "problem_type", "type", "correct"]
    file_path = "/home/rcyuh/Desktop/1. Đồ án tốt nghiệp/Co-supervised/assistment_2012_2013.csv"
    nrows=1000
       
    # # Test
    # pre = preprocessing_matrix(file_path=file_path)
    # matrix_df = pre.process()
