#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:12:41 2025

@author: rcyuh
"""

import pandas as pd
import numpy as np

class CoTrainingRecommender:
    def __init__(self, content_recommender, mf_recommender, ratings_df, batch_size=1000):
        """
        Khởi tạo hệ thống gợi ý kết hợp (Co-Training).
        
        Parameters:
        - content_recommender: Đối tượng của ContentBasedRecommender
        - mf_recommender: Đối tượng của MatrixFactorizationRecommender
        - ratings_df: DataFrame chứa dữ liệu ['userID', 'itemID', 'rating', 'timestamp']
        - batch_size: Kích thước batch trong quá trình huấn luyện
        """
        self.content_recommender = content_recommender
        self.mf_recommender = mf_recommender
        self.ratings_df = ratings_df
        self.batch_size = batch_size
        self.L, self.U = self._create_labeled_unlabeled_sets()
    
    def _create_labeled_unlabeled_sets(self):
        """
        Tạo tập dữ liệu có nhãn (L) và không có nhãn (U).
        """
        L = self.ratings_df.copy()
        all_user_item_pairs = set(zip(self.ratings_df['userID'], self.ratings_df['itemID']))
        all_users = self.ratings_df['userID'].unique()
        all_items = self.ratings_df['itemID'].unique()
        
        U = pd.DataFrame([(u, i) for u in all_users for i in all_items if (u, i) not in all_user_item_pairs],
                         columns=['userID', 'itemID'])
        return L, U
    
    def co_training(self, max_iter=10, confidence_threshold=0.8):
        """
        Thực hiện thuật toán Co-Training với Mini-Batch.
        
        Parameters:
        - max_iter: Số vòng lặp tối đa.
        - confidence_threshold: Ngưỡng tin cậy để thêm dữ liệu vào tập có nhãn.
        """
        for iteration in range(max_iter):
            new_labels = []
            
            for batch_start in range(0, len(self.U), self.batch_size):
                batch = self.U.iloc[batch_start:batch_start + self.batch_size]
                
                for _, row in batch.iterrows():
                    user_id, item_id = row['userID'], row['itemID']
                    
                    r_cbf = self.content_recommender.predict(user_id, item_id)
                    r_mf = self.mf_recommender.predict(user_id, item_id)
                    
                    if np.isnan(r_cbf) or np.isnan(r_mf):
                        continue
                    
                    confidence = 1 - abs(r_cbf - r_mf) / 5  # Chuẩn hóa tin cậy về khoảng [0,1]
                    
                    if confidence > confidence_threshold:
                        new_labels.append((user_id, item_id, (r_cbf + r_mf) / 2))
            
            if not new_labels:
                print(f"Dừng lại ở vòng {iteration}, không có mẫu mới.")
                break
            
            new_df = pd.DataFrame(new_labels, columns=['userID', 'itemID', 'rating'])
            self.L = pd.concat([self.L, new_df], ignore_index=True)
            self.U = self.U[~self.U.set_index(['userID', 'itemID']).index.isin(new_df.set_index(['userID', 'itemID']).index)]
            
            print(f"Vòng {iteration + 1}: Thêm {len(new_labels)} mẫu mới vào tập có nhãn.")
        
        return self.L
    
if __name__ == "__main__":
    pass