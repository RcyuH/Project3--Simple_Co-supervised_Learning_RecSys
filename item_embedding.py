#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:13:08 2025

@author: rcyuh
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Set
import os
from pathlib import Path
from preprocessing import preprocessing_meta_data, preprocessing_matrix

os.environ["TOKENIZjsonERS_PARALLELISM"] = "false"

class ItemEmbeddingGenerator:
    def __init__(self, 
                output_dimension: int = 384,  # MiniLM có 384 chiều
                include_fields: Set[str] = None):
        """
        Initialize generator with configurable fields
        
        Args:
            output_dimension: Embedding dimension (default 384 for MiniLM)
            include_fields: Set of fields to include in prompt
                          (topic, sequence_type, problem_type, amount, difficulty)
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.output_dimension = output_dimension
        self.include_fields = include_fields or {'topic', 'amount', 'difficulty', 'sequence_type', 'problem_type'} 

    def create_embedding_input(self, item_data: Dict) -> str:
        """Tạo prompt đầu vào dựa trên các trường được chọn"""
        prompt_parts = []

        if 'topic' in self.include_fields and (cats := item_data.get('topic')):
            category_str = " > ".join(cats) if isinstance(cats, list) else cats
            prompt_parts.append(f"topic: {category_str}")

        if 'difficulty' in self.include_fields and (difficulty := item_data.get('difficulty')):
            prompt_parts.append(f"difficulty: {difficulty}")

        if 'amount' in self.include_fields and (amount := item_data.get('amount')):
            prompt_parts.append(f"number of questions in the exercise: {amount}")
            
        if 'sequence_type' in self.include_fields and (sequence_type := item_data.get('sequence_type')):
            if sequence_type == "LinearSection":
                prompt_parts.append("sequence type: Student completes all problems in a predetermined order")
            if sequence_type == "MasterySection":
                prompt_parts.append("sequence type: Random order, and student must master the problem set by getting a certain number of questions correct in a row before being able to continue")
            if sequence_type == "RandomIterateSection":
                prompt_parts.append("sequence type: Student completes all problems, but each student is presented with the problems in a different random order")

        if 'problem_type' in self.include_fields and (problem_type := item_data.get('problem_type')):
            desc = str()
            tmp = str()
            for key, val in problem_type.items():
                if key == "fill_in_1":
                    desc += str(val) + " questions" + " Simple string-compared answer (text box); "
                elif key == "open_response":
                    desc += str(val) + " questions" + " Open response - Records student answer, but their response is always marked correct; "
                elif key == "algebra":
                    desc += str(val) + " questions" + " Math evaluated string (text box); "
                elif key == "choose_1":
                    desc += str(val) + " questions" + " Multiple choice (radio buttons) - only one correct answer; "
                elif key == "choose_n":
                    desc += str(val) + " questions" + " Multiple choice (radio buttons) - many correct answers; "
                else:
                    desc += str(val) + " questions" + " ranking; "
            prompt_parts.append(f"problem type: {desc}")

        return "\n".join(prompt_parts)

    def generate_item_embeddings(self, items: Dict) -> Dict[str, np.ndarray]:
        """Tạo embedding từ danh sách sản phẩm"""
        embeddings = {}
        texts = {item_id: self.create_embedding_input(data) for item_id, data in items.items()}

        # Encode tất cả các văn bản cùng lúc để nhanh hơn
        encoded_vectors = self.model.encode(list(texts.values()))

        for item_id, vector in zip(texts.keys(), encoded_vectors):
            embeddings[item_id] = np.array(vector)

        return embeddings
    
    def save_embeddings(self, embeddings, save_dir='data_save/embeddings'):
        """Save embeddings to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        items = sorted(embeddings.keys())
        embedding_array = np.stack([embeddings[item] for item in items])
        np.save(f'{save_dir}/embeddings.npy', embedding_array)
        np.save(f'{save_dir}/items.npy', np.array(items))
        print(f"Saved embeddings to {save_dir}")
        
    def load_embeddings(self, load_dir='data_save/embeddings'):
        """Load embeddings and create item mapping"""
        try:
            embedding_array = np.load(f'{load_dir}/embeddings.npy')
            items = np.load(f'{load_dir}/items.npy')
            embeddings = {item: emb for item, emb in zip(items, embedding_array)}
            item_to_idx = {item: idx for idx, item in enumerate(items)}
            print(f"Loaded embeddings for {len(embeddings)} items")
            
            return embeddings, item_to_idx
        
        except FileNotFoundError:
            print("No saved embeddings found")
            
            return None, None
    
    def debug_prompt(self, items: Dict, num_samples: int = 3):
        """Hiển thị ví dụ prompt"""
        print("\nSample prompts:")
        for item_id in list(items.keys())[:num_samples]:
            print(f"\nItem ID: {item_id}")
            print("-" * 40)
            print(self.create_embedding_input(items[item_id]))
            print("=" * 80)

if __name__ == "__main__":
    # Ví dụ sử dụng
    # Constant
    cols_needed = ["problem_id", "sequence_id", "skill", "problem_type", "type", "correct"]
    file_path = "/home/rcyuh/Desktop/1. Đồ án tốt nghiệp/Co-supervised/assistment_2012_2013.csv"
    nrows=1000
    
    pre = preprocessing_matrix(file_path=file_path)
    pre.reverse_correct()
    pre.filter_matrix()
    matrix_df = pre.df
    unique_values = pre.extract_sequence_id()
    
    pre_meta = preprocessing_meta_data(file_path=file_path, sequence_id_list=unique_values)
    items_dict = pre_meta.process()
    
    generator = ItemEmbeddingGenerator()
    item_embeddings = generator.generate_item_embeddings(items_dict)
    generator.save_embeddings(embeddings=item_embeddings)