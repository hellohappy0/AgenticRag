from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import docx
import os
from .base_processor import DocumentProcessor, DocumentSplitter


class SimpleTextSplitter(DocumentSplitter):
    """
    简单的文本分割器，基于固定的字符数分割文本
    """
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100, **kwargs) -> List[str]:
        """
        按固定字符数分割文本
        
        @param text: 要分割的文本
        @param chunk_size: 每个片段的最大字符数
        @param chunk_overlap: 相邻片段之间的重叠字符数
        @param kwargs: 其他参数
        @return: 分割后的文本片段列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            # 如果不是最后一个块，确保在句子结束处分割
            if end < text_length:
                # 寻找最近的句号、问号或感叹号
                punctuation_positions = [text.rfind(p, start, end) for p in ['.', '?', '!']]
                valid_positions = [pos for pos in punctuation_positions if pos > start + chunk_size * 0.5]
                
                if valid_positions:
                    end = max(valid_positions) + 1
            
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
            
            # 防止无限循环
            if start >= text_length or start >= end:
                break
        
        return chunks


class SimpleDocumentProcessor(DocumentProcessor):
    """
    简单的文档处理器，支持处理多种类型的文档
    """
    
    def __init__(self, splitter: Optional[DocumentSplitter] = None):
        """
        初始化文档处理器
        
        @param splitter: 文档分割器，默认为SimpleTextSplitter
        """
        self.splitter = splitter or SimpleTextSplitter()
    
    def process(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        处理文档内容
        
        @param content: 文档内容字符串
        @param kwargs: 处理参数，会传递给分割器
        @return: 处理后的文档块列表
        """
        if not content:
            return []
        
        # 使用分割器分割内容
        chunks = self.splitter.split(content, **kwargs)
        
        # 为每个块创建元数据
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                processed_chunks.append({
                    "id": f"chunk_{i}",
                    "content": chunk,
                    "metadata": {
                        "chunk_index": i,
                        "chunk_size": len(chunk)
                    }
                })
        
        return processed_chunks
    
    def load(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        从文件加载并处理文档
        
        @param file_path: 文件路径
        @param kwargs: 处理参数，会传递给process方法
        @return: 处理后的文档块列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = file_path.suffix.lower()
        content = ""
        
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_ext == '.pdf':
                content = self._read_pdf(file_path)
            elif file_ext == '.docx':
                content = self._read_docx(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 处理文档内容
            return self.process(content, **kwargs)
        except Exception as e:
            raise RuntimeError(f"加载文件时出错: {str(e)}")
    
    def _read_pdf(self, file_path: Path) -> str:
        """
        读取PDF文件内容
        
        @param file_path: PDF文件路径
        @return: 提取的文本内容
        """
        content = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                content.append(page.extract_text())
        
        return '\n'.join(content)
    
    def _read_docx(self, file_path: Path) -> str:
        """
        读取DOCX文件内容
        
        @param file_path: DOCX文件路径
        @return: 提取的文本内容
        """
        doc = docx.Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)
        
        return '\n'.join(content)