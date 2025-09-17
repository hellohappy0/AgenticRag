from typing import List, Dict, Any, Optional
import numpy as np


class MockFAISSVectorStore:
    """
    模拟FAISS向量存储，用于测试和演示
    """
    
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """
        初始化模拟FAISS向量存储
        
        @param documents: 文档列表，每个文档应包含"content"字段
        @param kwargs: 其他参数（与真实实现保持接口一致）
        """
        # 存储文档
        self.documents = []
        
        # 模拟FAISS索引（实际上我们不使用FAISS）
        self.embedding_dim = 384  # 默认嵌入维度
        self.index = None  # 模拟索引对象
        
        print("初始化模拟FAISS向量存储")
        
        # 如果提供了文档，则添加它们
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向模拟向量存储中添加文档
        
        @param documents: 文档列表，每个文档应包含"content"字段
        """
        num_docs = len(documents)
        
        # 提取文档内容
        contents = [doc["content"] for doc in documents]
        
        # 模拟生成嵌入向量
        print(f"正在生成{num_docs}个文档的模拟嵌入...")
        embeddings = np.random.randn(num_docs, self.embedding_dim).astype(np.float32)
        
        # 模拟添加到索引
        print(f"模拟添加{num_docs}个文档到向量索引")
        
        # 存储文档
        self.documents.extend(documents)
        print(f"已成功添加{num_docs}个文档到模拟向量存储")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        模拟搜索功能
        
        @param query: 查询文本
        @param top_k: 返回的结果数量
        @return: 搜索结果列表
        """
        # 模拟为查询生成嵌入
        print(f"为查询 '{query}' 生成模拟嵌入...")
        query_embedding = np.random.randn(1, self.embedding_dim).astype(np.float32)
        
        # 简单的基于字符串匹配的搜索（作为模拟）
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                results.append(doc.copy())
            
            if len(results) >= top_k:
                break
        
        # 如果没有匹配结果，返回前top_k个文档
        if not results and len(self.documents) > 0:
            results = [doc.copy() for doc in self.documents[:min(top_k, len(self.documents))]]
        
        # 为结果添加模拟分数
        for i, result in enumerate(results):
            result["score"] = 1.0 - (i * 0.1)  # 简单的模拟分数递减
        
        print(f"搜索 '{query}' 返回了 {len(results)} 个结果")
        return results


# 导出类
__all__ = ["MockFAISSVectorStore"]