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
                          (title, description, category, brand, price, sales_rank)
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.output_dimension = output_dimension
        self.include_fields = include_fields or {'title', 'description', 'category', 'brand', 'price', 'sales_rank'}

    def create_embedding_input(self, item_data: Dict) -> str:
        """Tạo prompt đầu vào dựa trên các trường được chọn"""
        prompt_parts = []

        if 'title' in self.include_fields and (title := item_data.get('title')):
            prompt_parts.append(f"title: {title}")

        if 'description' in self.include_fields and (desc := item_data.get('description')):
            prompt_parts.append(f"description: {desc}")

        if 'category' in self.include_fields and (cats := item_data.get('category')):
            category_str = " > ".join(cats) if isinstance(cats, list) else cats
            prompt_parts.append(f"category: {category_str}")

        if 'price' in self.include_fields and (price := item_data.get('price')):
            prompt_parts.append(f"price: {price}")

        if 'brand' in self.include_fields and (brand := item_data.get('brand')):
            prompt_parts.append(f"brand: {brand}")

        if 'sales_rank' in self.include_fields and (rank := item_data.get('rank')):
            prompt_parts.append(f"sales rank: {rank}")

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
    
    # def save_embeddings(self, embeddings, file_path="data_save/embeddings.json"):
    #     # Chuyển `ndarray` thành list để JSON có thể lưu trữ
    #     formatted_embeddings = {str(k): v.tolist() for k, v in embeddings.items()}
    #     with open(file_path, 'w') as f:
    #         json.dump(formatted_embeddings, f)
    
    # def load_embeddings(self, file_path="data_save/embeddings.json"):
    #     with open(file_path, 'r') as f:
    #         data = json.load(f)
    #         # Chuyển list về `ndarray`
            
    #         return {int(k): np.array(v) for k, v in data.items()}
    
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
    items = {
        "item1": {"title": "iPhone 15 Pro", "description": "Titanium frame, powerful chip", "category": ["Electronics", "Mobile Phones"], "brand": "Apple", "price": 999, "rank": 1},
        "item2": {"title": "Samsung Galaxy S24", "description": "Latest AI-powered smartphone", "category": ["Electronics", "Mobile Phones"], "brand": "Samsung", "price": 899, "rank": 2}
    }
    
    generator = ItemEmbeddingGenerator()
    generator.debug_prompt(items)

    embeddings = generator.generate_item_embeddings(items)
    print("\nEmbedding size:", embeddings["item1"].shape)  # Output: (384,)