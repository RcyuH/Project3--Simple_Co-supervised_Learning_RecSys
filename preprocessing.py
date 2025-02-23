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
    difficulty: Correct_avg đã chuẩn hóa
"""