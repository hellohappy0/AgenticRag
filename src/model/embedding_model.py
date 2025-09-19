import numpy as np
import requests
import json
from abc import ABC, abstractmethod
from typing import List, Optional
from src.config import get_config


class BaseEmbeddingModel(ABC):
    """
    嵌入模型基类，定义了生成文本嵌入的基本接口
    """
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        为单个文本生成嵌入向量
        
        @param text: 输入文本
        @return: 嵌入向量
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        为多个文本批量生成嵌入向量
        
        @param texts: 输入文本列表
        @return: 嵌入向量数组
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        @return: 嵌入向量维度
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查嵌入模型是否可用
        
        @return: 模型是否可用
        """
        pass


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    基于Ollama的嵌入模型实现
    """
    
    def __init__(self, model_name: str = "bge-m3:latest", base_url: Optional[str] = None):
        """
        初始化Ollama嵌入模型
        
        @param model_name: 模型名称
        @param base_url: Ollama API的基础URL
        """
        self.model_name = model_name
        # 从配置获取Ollama连接信息，如果没有提供的话
        self.base_url = base_url or get_config("model.ollama.base_url", "http://localhost:11434")
        
        # 模型是否可用的标志
        self._available = None
        # 检查服务健康状态
        self.check_health()
        
        # 默认嵌入维度，当无法获取真实维度时使用
        self.default_embedding_dim = 1024  # bge-m3模型的默认维度
    
    def check_health(self) -> bool:
        """
        检查Ollama服务的健康状态
        
        @return: 服务是否健康
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                # 检查模型是否已安装
                self._available = self.model_name in model_names
                return self._available
            else:
                self._available = False
                return False
        except Exception as e:
            self._available = False
            return False
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        从Ollama获取单个文本的嵌入
        
        @param text: 输入文本
        @return: 嵌入向量
        """
        if not self.is_available():
            # 返回随机向量作为备用
            return np.random.randn(self.default_embedding_dim).astype(np.float32)
        
        try:
            url = f"{self.base_url}/api/embeddings"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"从Ollama获取嵌入时出错: {str(e)}")
            # 返回随机向量作为备用
            return np.random.randn(self.default_embedding_dim).astype(np.float32)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        从Ollama批量获取文本嵌入
        
        @param texts: 输入文本列表
        @return: 嵌入向量数组
        """
        # 批量获取嵌入向量
        embeddings = [self.get_embedding(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        @return: 嵌入向量维度
        """
        # 对于Ollama，我们假设是默认维度
        # 实际应用中可能需要调用一次API来确定真实维度
        return self.default_embedding_dim
    
    def is_available(self) -> bool:
        """
        检查嵌入模型是否可用
        
        @return: 模型是否可用
        """
        if self._available is None:
            self.check_health()
        return self._available





class EmbeddingModelFactory:
    """
    嵌入模型工厂类，用于创建不同类型的嵌入模型
    """
    
    @staticmethod
    def create_embedding_model(source: str = "ollama", model_name: str = None, **kwargs) -> BaseEmbeddingModel:
        """
        创建嵌入模型实例
        
        @param source: 嵌入来源，可选值："ollama"
        @param model_name: 模型名称
        @param kwargs: 其他参数
        @return: 嵌入模型实例
        """
        source = source.lower()
        
        if source == "ollama":
            # 如果没有提供模型名称，使用默认值
            if model_name is None:
                model_name = "bge-m3:latest"
            return OllamaEmbeddingModel(model_name=model_name, **kwargs)
        else:
            print(f"Warning: 不支持的嵌入来源: {source}")
            # 默认返回Ollama嵌入模型
            if model_name is None:
                model_name = "bge-m3:latest"
            return OllamaEmbeddingModel(model_name=model_name, **kwargs)


# 导出类
__all__ = ["BaseEmbeddingModel", "OllamaEmbeddingModel", "EmbeddingModelFactory"]