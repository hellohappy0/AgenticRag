import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FAISSVectorStore")

from src.model.embedding_model import EmbeddingModelFactory


class FAISSVectorStore:
    """
    基于FAISS的向量存储实现
    """
    
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, embedding_source: str = "ollama", 
                 embedding_model_name: str = "bge-m3:latest", embedding_dim: int = 1024):
        """
        初始化FAISS向量存储
        
        @param documents: 文档列表，每个文档应包含"content"字段
        @param embedding_source: 嵌入来源，可选值："ollama"
        @param embedding_model_name: 用于生成嵌入的模型名称，默认为Ollama上的bge-m3模型
        @param embedding_dim: 嵌入向量的维度（当无法加载模型时使用）
        """
        try:
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
            
            logger.info("FAISS向量存储初始化成功")
        except Exception as e:
            logger.error(f"FAISS向量存储初始化失败: {str(e)}")
            # 即使初始化失败也继续，后面会处理可用性检查
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向向量存储中添加文档
        
        @param documents: 文档列表，每个文档应包含"content"字段
        """
        try:
            if not documents:
                logger.warning("没有文档可添加")
                return
            
            # 检查嵌入模型可用性
            if not self.embedding_model.is_available():
                logger.warning("嵌入模型不可用，无法添加文档")
                return
            
            # 提取文档内容
            contents = [doc["content"] for doc in documents]
            num_docs = len(documents)
            
            # 生成向量嵌入
            embeddings = self.embedding_model.get_embeddings(contents)
            
            # 确保嵌入是float32格式
            embeddings = embeddings.astype(np.float32)
            
            # 添加到FAISS索引
            if embeddings.size > 0:
                self.index.add(embeddings)
                # 存储文档
                self.documents.extend(documents)
                logger.info(f"成功添加{num_docs}个文档到向量存储")
            else:
                logger.warning("未获取到有效嵌入向量，文档添加失败")
        except Exception as e:
            logger.error(f"添加文档时出错: {str(e)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索与查询最相关的文档
        
        @param query: 查询文本
        @param top_k: 返回的结果数量
        @return: 搜索结果列表
        """
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("向量存储为空或未初始化，无法执行搜索")
                return []
            
            # 检查嵌入模型可用性
            if not self.embedding_model.is_available():
                logger.warning("嵌入模型不可用，无法执行搜索")
                return []
            
            # 为查询生成嵌入
            query_embedding = self.embedding_model.get_embedding(query).reshape(1, -1)
            
            # 检查嵌入向量是否有效
            if query_embedding is None or len(query_embedding) == 0:
                logger.warning("获取查询嵌入失败，返回空结果")
                return []
            
            # 执行FAISS搜索
            distances, indices = self.index.search(query_embedding, top_k)
            
            # 准备结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):  # 确保索引有效
                    try:
                        # 复制文档并添加分数
                        result_doc = self.documents[idx].copy()
                        result_doc["score"] = float(1.0 / (1.0 + distances[0][i]))  # 将距离转换为相似度分数
                        results.append(result_doc)
                    except Exception as inner_e:
                        logger.error(f"处理文档结果时出错: {str(inner_e)}")
                        continue
            
            logger.info(f"搜索查询'{query}'返回{len(results)}个结果")
            return results
        except Exception as e:
            logger.error(f"搜索过程中出错: {str(e)}")
            return []


    def is_available(self) -> bool:
        """
        检查向量存储是否可用
        
        @return: 是否可用
        """
        try:
            # 检查嵌入模型和索引是否都可用
            model_available = hasattr(self, 'embedding_model') and self.embedding_model.is_available()
            index_available = self.index is not None and len(self.documents) > 0
            return model_available and index_available
        except Exception as e:
            logger.error(f"检查向量存储可用性时出错: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储的统计信息
        
        @return: 统计信息字典
        """
        try:
            stats = {
                "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
                "embedding_dimension": self.embedding_dim if hasattr(self, 'embedding_dim') else None,
                "is_index_initialized": self.index is not None,
                "is_embedding_model_available": self.embedding_model.is_available() if hasattr(self, 'embedding_model') else False
            }
            return stats
        except Exception as e:
            logger.error(f"获取向量存储统计信息时出错: {str(e)}")
            return {"error": str(e)}


# 导出类
__all__ = ["FAISSVectorStore"]