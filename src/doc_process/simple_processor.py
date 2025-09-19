from typing import List, Optional
from pathlib import Path
from .base_processor import DocumentProcessor, DocumentSplitter
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text


class SimpleTextSplitter(DocumentSplitter):
    """
    简单的文本分割器，使用unstructured包进行文本分割
    """
    
    def split(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """将文本分割成指定大小的块"""
        # 参数验证
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap不能为负数")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        
        # 如果文本为空或长度小于等于chunk_size，直接返回原文本
        if not text or len(text) <= chunk_size:
            return [text]
        
        # 使用unstructured的chunk_by_title进行分割
        # 首先创建一个临时的Text元素
        elements = [Text(text=text)]
        
        # 进行分块，使用chunk_size和chunk_overlap参数
        # 注意：unstructured的chunk_by_title方法不直接支持overlap参数
        # 我们会在分块后手动处理重叠逻辑
        chunks = chunk_by_title(
            elements=elements,
            max_characters=chunk_size,
            combine_text_under_n_chars=int(chunk_size * 0.8),  # 合并小于chunk_size 80%的块
        )
        
        # 提取文本内容
        text_chunks = [chunk.text for chunk in chunks]
        
        # 手动处理重叠逻辑
        if chunk_overlap > 0 and len(text_chunks) > 1:
            overlapped_chunks = []
            for i in range(len(text_chunks)):
                if i > 0:
                    # 为当前块添加前一个块的末尾重叠部分
                    prev_chunk = text_chunks[i-1]
                    overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) >= chunk_overlap else prev_chunk
                    overlapped_chunks.append(overlap_text + text_chunks[i])
                else:
                    # 第一个块保持不变
                    overlapped_chunks.append(text_chunks[i])
            text_chunks = overlapped_chunks
        
        # 清理空块
        text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
        
        return text_chunks


class SimpleDocumentProcessor(DocumentProcessor):
    """
    简单的文档处理器，使用unstructured包处理文档
    目前仅支持.md和.txt格式
    """
    
    def __init__(self, splitter: Optional[DocumentSplitter] = None):
        """
        初始化文档处理器
        
        Args:
            splitter: 文档分割器，如果为None，则使用默认的SimpleTextSplitter
        """
        self.splitter = splitter or SimpleTextSplitter()
    
    def load(self, file_path: str) -> str:
        """
        加载文档内容
        
        Args:
            file_path: 文档路径
        
        Returns:
            文档内容
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # 支持的文件类型
        if file_ext == '.txt':
            elements = partition_text(str(file_path))
        elif file_ext == '.md':
            elements = partition_md(str(file_path))
        else:
            raise ValueError(f"目前仅支持.md和.txt格式的文件，不支持: {file_ext}")
        
        # 提取文本内容
        text = "\n".join([element.text for element in elements])
        return text
    
    def process(self, file_path_or_content: str, chunk_size: int = 500, chunk_overlap: int = 50, is_content: bool = False) -> List[str]:
        """
        处理文档，包括加载和分割
        
        Args:
            file_path_or_content: 文档路径或文档内容
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            is_content: 如果为True，则file_path_or_content被视为内容而非文件路径
        
        Returns:
            分割后的文档块列表
        """
        if is_content:
            text = file_path_or_content
        else:
            text = self.load(file_path_or_content)
        return self.splitter.split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)