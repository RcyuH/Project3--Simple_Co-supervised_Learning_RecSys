#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:12:18 2025

@author: rcyuh
"""

"""
preprocessing for items meta data
+) B1 - getdata - các columns sẽ giữ: problem_id, sequence_id, skill, problem_type, type, correct_avg (đại diện cho độ khó)
    - Skill: Kỹ năng cần có để làm được problem hay topic của problem
    - Problem_Type: Dạng (về hình thức) của problem
    - Type: Dạng (về hình thức) của sequence
+) B2 - Nhóm theo sequence - Tạo những thông tin cần thiết
    - Topics (Datatype - List): Tên các skill và sắp xếp từ trái sang phải theo thứ tự giảm dần số câu có skill đó trong sequence
    - Problem_types (Datatype - Dict): Tên các type (key) và số câu có type đó (val) 
    - Types: Tên type => Description
    - Correct_avg: Điểm số trung bình của các học sinh đã làm sequence => Độ khó => Chuẩn hóa về thang điểm 1
    - Sequence_id
+) Prompt Pattern:
    topic: topic1 > topic2 > topic3
    type (cần một cái tên chuẩn hơn): Mô tả về type
    problem_type (cần một cái tên chuẩn hơn): Mô tả về problem_type và số câu có problem_type đó 
    difficulty (theo thang điểm 0-1): Correct_avg đã chuẩn hóa
"""

import pandas as pd
from collections import Counter

class preprocessing_meta_data:
    def __init__(self, file_path, nrows, cols_needed):
        self.df = pd.read_csv(file_path, usecols=cols_needed, nrows=nrows)
        self.dict = {}
        
        # Group by problem_id
        self.df = self.df.groupby("problem_id").agg({
            "sequence_id": "first",  # Giữ giá trị đầu tiên
            "skill": "first",
            "problem_type": "first",
            "type": "first",
            "correct": "mean"  # Tính trung bình
        }).reset_index()
        
        self.df.rename(columns={"correct": "correct_avg"}, inplace=True)
        
        self.df["skill"] = self.df["skill"].fillna("Unknown")
        
    def group_by_sequence(self):
        self.df = self.df.groupby("sequence_id").agg({
            "skill": list,
            "problem_type": list,
            "type": "first",
            "correct_avg": "mean",
            "problem_id": "count"
        }).reset_index()
        
        self.df.rename(columns={"problem_id": "amount"}, inplace=True)
        self.df.rename(columns={"correct_avg": "difficulty"}, inplace=True)
        self.df.rename(columns={"skill": "topic"}, inplace=True)
        self.df.rename(columns={"type": "sequence_type"}, inplace=True)
        
        # convert list type -> dict type with amount
        self.df["topic"] = self.df["topic"].apply(lambda x: dict(Counter(x)))
        self.df["problem_type"] = self.df["problem_type"].apply(lambda x: dict(Counter(x)))
        
        # convert dict type -> list type (ordered)
        self.df["topic"] = self.df["topic"].apply(lambda x: sorted(x, key=x.get, reverse=True) if isinstance(x, dict) else x)
        
        # binning diffculty
        self.df["difficulty"] = pd.cut(
            self.df["difficulty"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
            labels=["Very Hard", "Hard", "Normal", "Easy", "Very Easy"],
            include_lowest=True
        )
    
    def convert_df_to_dict(self):
        self.dict = self.df.set_index("sequence_id").to_dict(orient="index")
    
    def process(self):
        self.group_by_sequence()
        self.convert_df_to_dict()
        
        return self.dict
    
    """
    Ngoài ra còn có thể thêm các bước xử lý khác như: 
        - Bắt buộc phải có: Đảo giá trị của correct
        - Lọc (phải chú ý thống nhất giữa metadata và utility matrix)
    """
    
class preprocessing_matrix:
    def __init__(self):
        pass
    
    def generate_utility_matrix(self):
        pass
    
    def filter_matrix(self):
        pass
    
    def reverse_correct(self):
        pass

if __name__ == "__main__":
    # Constant
    cols_needed = ["problem_id", "sequence_id", "skill", "problem_type", "type", "correct"]
    file_path = "/home/rcyuh/Desktop/1. Đồ án tốt nghiệp/Co-supervised/assistment_2012_2013.csv"
    nrows=1000
    
    # Test
    pre = preprocessing_meta_data(file_path, nrows, cols_needed)
    df = pre.process()