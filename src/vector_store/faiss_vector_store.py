import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from src.model.embedding_model import EmbeddingModelFactory


class FAISSVectorStore:
    """
    基于FAISS的向量存储实现
    """
    
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, embedding_source: str = "ollama", 
                 embedding_model_name: str = "bge-m3", embedding_dim: int = 1024):
        """
        初始化FAISS向量存储
        
        @param documents: 文档列表，每个文档应包含"content"字段
        @param embedding_source: 嵌入来源，可选值："ollama"
        @param embedding_model_name: 用于生成嵌入的模型名称，默认为Ollama上的bge-m3模型
        @param embedding_dim: 嵌入向量的维度（当无法加载模型时使用）
        """
        # 使用嵌入模型工厂创建嵌入模型
        self.embedding_model = EmbeddingModelFactory.create_embedding_model(
            source=embedding_source,
            model_name=embedding_model_name
        )
        
        # 存储文档和索引
        self.documents = []
        
        # 获取嵌入维度
        if self.embedding_model.is_available():
            self.embedding_dim = self.embedding_model.get_embedding_dimension()
        else:
            self.embedding_dim = embedding_dim
        
        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # 使用L2距离的平面索引
        
        # 如果提供了文档，则添加它们
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向向量存储中添加文档
        
        @param documents: 文档列表，每个文档应包含"content"字段
        """
        # 提取文档内容
        contents = [doc["content"] for doc in documents]
        num_docs = len(documents)
        
        # 生成向量嵌入
        embeddings = self.embedding_model.get_embeddings(contents)
        
        # 确保嵌入是float32格式
        embeddings = embeddings.astype(np.float32)
        
        # 添加到FAISS索引
        self.index.add(embeddings)
        
        # 存储文档
        self.documents.extend(documents)
        print(f"已成功添加{num_docs}个文档到向量存储")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索与查询最相关的文档
        
        @param query: 查询文本
        @param top_k: 返回的结果数量
        @return: 搜索结果列表
        """
        # 为查询生成嵌入
        query_embedding = self.embedding_model.get_embedding(query).reshape(1, -1)
        
        # 执行FAISS搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 准备结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):  # 确保索引有效
                # 复制文档并添加分数
                result_doc = self.documents[idx].copy()
                result_doc["score"] = float(1.0 / (1.0 + distances[0][i]))  # 将距离转换为相似度分数
                results.append(result_doc)
        
        print(f"搜索 '{query}' 返回了 {len(results)} 个结果")
        return results


# 导出类
__all__ = ["FAISSVectorStore"]