import unittest
import numpy as np
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.vector_store.faiss_vector_store import FAISSVectorStore
from src.doc_process.simple_processor import SimpleDocumentProcessor

class TestFAISSVectorStore(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 设置测试文档路径
        cls.docs_dir = Path(__file__).resolve().parent.parent.parent / 'docs'
        # 加载docs目录下的所有md文件
        cls.doc_files = list(cls.docs_dir.glob('*.md'))
        cls.processor = SimpleDocumentProcessor()
        
        # 加载并处理文档
        cls.documents = []
        for doc_file in cls.doc_files:
            try:
                # 使用真实的文档处理器处理文件
                processed_docs = cls.processor.process(str(doc_file), chunk_size=500, chunk_overlap=50)
                
                # 将处理后的文档转换为FAISSVectorStore所需的格式
                for chunk in processed_docs:
                    doc = {
                        'content': chunk,
                        'metadata': {'source': str(doc_file.name)}
                    }
                    cls.documents.append(doc)
            except Exception as e:
                print(f"处理文件 {doc_file.name} 时出错: {str(e)}")
                
        # 如果没有找到文档，创建一些模拟文档
        if not cls.documents:
            print("未找到有效的文档，创建模拟文档用于测试...")
            mock_docs = [
                {'content': '这是模拟文档内容1', 'metadata': {'source': 'mock_doc1.md'}},
                {'content': '这是模拟文档内容2', 'metadata': {'source': 'mock_doc2.md'}},
                {'content': '这是模拟文档内容3', 'metadata': {'source': 'mock_doc3.md'}}
            ]
            cls.documents = mock_docs
        
    def setUp(self):
        # 使用实际的嵌入模型，但限制文档数量以加速测试
        # 为了避免测试依赖外部服务，我们仍然使用模拟的嵌入模型
        self.mock_embedding_model = MagicMock()
        self.mock_embedding_model.is_available.return_value = True
        self.mock_embedding_model.get_embedding_dimension.return_value = 128
        self.mock_embedding_model.get_embeddings.return_value = np.random.randn(len(self.documents), 128).astype(np.float32)
        self.mock_embedding_model.get_embedding.return_value = np.random.randn(128).astype(np.float32)
        
    @patch('src.vector_store.faiss_vector_store.EmbeddingModelFactory')
    def test_initialization(self, mock_factory):
        """测试向量存储初始化"""
        mock_factory.create_embedding_model.return_value = self.mock_embedding_model
        
        # 初始化向量存储
        vector_store = FAISSVectorStore()
        
        # 验证初始化是否成功
        self.assertIsNotNone(vector_store)
        self.assertTrue(hasattr(vector_store, 'index'))
        self.assertTrue(hasattr(vector_store, 'documents'))
        self.assertEqual(len(vector_store.documents), 0)
    
    @patch('src.vector_store.faiss_vector_store.EmbeddingModelFactory')
    def test_add_documents(self, mock_factory):
        """测试添加文档功能"""
        mock_factory.create_embedding_model.return_value = self.mock_embedding_model
        
        # 使用前几个文档进行测试，确保使用真实的文档处理结果
        test_docs = self.documents[:3] if len(self.documents) >= 3 else self.documents
        
        # 初始化并添加文档
        vector_store = FAISSVectorStore(documents=test_docs)
        
        # 验证文档是否成功添加
        self.assertEqual(len(vector_store.documents), len(test_docs))
        self.mock_embedding_model.get_embeddings.assert_called_once()
    
    @patch('src.vector_store.faiss_vector_store.EmbeddingModelFactory')
    def test_search(self, mock_factory):
        """测试搜索功能"""
        mock_factory.create_embedding_model.return_value = self.mock_embedding_model
        
        # 使用前5个文档进行测试，确保使用真实的文档处理结果
        test_docs = self.documents[:5] if len(self.documents) >= 5 else self.documents
        
        # 初始化并添加文档
        vector_store = FAISSVectorStore(documents=test_docs)
        
        # 模拟FAISS搜索结果
        with patch.object(vector_store.index, 'search') as mock_search:
            # 模拟返回前3个结果
            mock_search.return_value = (
                np.array([[0.1, 0.2, 0.3]]),  # 距离
                np.array([[0, 1, 2]])          # 索引
            )
            
            # 执行搜索
            results = vector_store.search("配置指南", top_k=3)
            
            # 验证搜索结果
            self.assertEqual(len(results), 3)
            self.mock_embedding_model.get_embedding.assert_called_once_with("配置指南")
            mock_search.assert_called_once()
    
    @patch('src.vector_store.faiss_vector_store.EmbeddingModelFactory')
    def test_is_available(self, mock_factory):
        """测试向量存储可用性检查"""
        mock_factory.create_embedding_model.return_value = self.mock_embedding_model
        
        # 初始化空的向量存储
        vector_store_empty = FAISSVectorStore()
        self.assertFalse(vector_store_empty.is_available())
        
        # 初始化有文档的向量存储
        if self.documents:
            vector_store_with_docs = FAISSVectorStore(documents=[self.documents[0]])
            self.assertTrue(vector_store_with_docs.is_available())
    
    @patch('src.vector_store.faiss_vector_store.EmbeddingModelFactory')
    def test_get_stats(self, mock_factory):
        """测试获取统计信息功能"""
        mock_factory.create_embedding_model.return_value = self.mock_embedding_model
        
        # 使用前3个文档进行测试，确保使用真实的文档处理结果
        test_docs = self.documents[:3] if len(self.documents) >= 3 else self.documents
        
        # 初始化并添加文档
        vector_store = FAISSVectorStore(documents=test_docs)
        
        # 获取统计信息
        stats = vector_store.get_stats()
        
        # 验证统计信息
        self.assertEqual(stats["document_count"], len(test_docs))
        self.assertEqual(stats["embedding_dimension"], 128)
        self.assertTrue(stats["is_index_initialized"])
        self.assertTrue(stats["is_embedding_model_available"])

if __name__ == '__main__':
    unittest.main()