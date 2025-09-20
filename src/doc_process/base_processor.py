from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod


class DocumentProcessor(ABC):
    """
    文档处理器基类，定义了文档处理的基本接口
    """
    
    @abstractmethod
    def process(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        处理文档内容
        
        @param content: 文档内容字符串
        @param kwargs: 处理参数
        @return: 处理后的文档块列表，每个块包含id、内容等信息
        """
        pass
    
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        从文件加载并处理文档
        
        @param file_path: 文件路径
        @param kwargs: 处理参数
        @return: 处理后的文档块列表
        """
        pass


class DocumentSplitter(ABC):
    """
    文档分割器基类，定义了文档分割的基本接口
    """
    
    @abstractmethod
    def split(self, text: str, **kwargs) -> List[str]:
        """
        分割文本为多个片段
        
        @param text: 要分割的文本
        @param kwargs: 分割参数
        @return: 分割后的文本片段列表
        """
        pass