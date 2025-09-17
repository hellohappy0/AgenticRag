import numpy as np
from typing import List
from src.model.embedding_model import BaseEmbeddingModel


class MockEmbeddingModel(BaseEmbeddingModel):
    """
    模拟嵌入模型，用于测试和演示
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        初始化模拟嵌入模型
        
        @param embedding_dim: 嵌入向量的维度
        """
        self.embedding_dim = embedding_dim
        print(f"初始化模拟嵌入模型，维度: {embedding_dim}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        为文本生成模拟嵌入向量
        
        @param text: 输入文本
        @return: 模拟嵌入向量
        """
        # 使用简单的随机向量作为模拟嵌入
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        为多个文本批量生成模拟嵌入向量
        
        @param texts: 输入文本列表
        @return: 模拟嵌入向量数组
        """
        num_texts = len(texts)
        print(f"正在生成{num_texts}个文本的模拟嵌入...")
        
        # 为每个文本生成随机嵌入
        embeddings = np.random.randn(num_texts, self.embedding_dim).astype(np.float32)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        @return: 嵌入向量维度
        """
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """
        检查模拟嵌入模型是否可用
        
        @return: 模型是否可用
        """
        return True  # 模拟模型始终可用


# 导出类
__all__ = ["MockEmbeddingModel"]