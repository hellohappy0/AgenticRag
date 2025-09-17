from typing import List, Dict, Any


class MockVectorStore:
    """
    模拟向量存储，用于演示检索功能
    """
    
    def __init__(self, documents: List[Dict[str, Any]]):
        """
        初始化模拟向量存储
        
        @param documents: 文档列表
        """
        self.documents = documents
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        模拟搜索功能
        
        @param query: 查询文本
        @param top_k: 返回的结果数量
        @return: 搜索结果列表
        """
        # 简单的基于字符串匹配的搜索
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                results.append(doc)
            
            if len(results) >= top_k:
                break
        
        # 如果没有匹配结果，返回前top_k个文档
        if not results and len(self.documents) > 0:
            results = self.documents[:min(top_k, len(self.documents))]
        
        return results