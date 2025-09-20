import unittest
from unittest.mock import Mock, patch
from src.tools.retrieval_tool import RetrievalTool, ToolError

class TestRetrievalTool(unittest.TestCase):
    
    def setUp(self):
        # 创建一个模拟的向量存储对象
        self.mock_vector_store = Mock()
        # 创建检索工具实例
        self.retrieval_tool = RetrievalTool(self.mock_vector_store)
    
    def test_tool_initialization(self):
        """测试检索工具的初始化是否正确"""
        self.assertEqual(self.retrieval_tool.name, "retrieve_documents")
        self.assertIn("从本地知识库中获取准确、可靠的已有信息", self.retrieval_tool.description)
        self.assertEqual(self.retrieval_tool.vector_store, self.mock_vector_store)
    
    def test_run_with_valid_query(self):
        """测试使用有效查询运行检索工具"""
        # 设置模拟对象的返回值
        mock_results = [
            {"content": "测试文档1", "metadata": {"source": "文档源1"}, "score": 0.9},
            {"content": "测试文档2", "metadata": {"source": "文档源2"}, "score": 0.8}
        ]
        self.mock_vector_store.search.return_value = mock_results
        
        # 执行检索
        result = self.retrieval_tool("测试查询", top_k=2)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["query"], "测试查询")
        self.assertEqual(result["results"], mock_results)
        self.assertEqual(result["total_results"], 2)
        
        # 验证向量存储的search方法被正确调用
        self.mock_vector_store.search.assert_called_once_with("测试查询", top_k=2)
    
    def test_run_with_default_top_k(self):
        """测试使用默认的top_k参数"""
        # 设置模拟对象的返回值
        mock_results = [{"content": "测试文档", "metadata": {"source": "文档源"}, "score": 0.9}]
        self.mock_vector_store.search.return_value = mock_results
        
        # 执行检索，不指定top_k
        result = self.retrieval_tool("测试查询")
        
        # 验证向量存储的search方法被调用时使用了默认的top_k=3
        self.mock_vector_store.search.assert_called_once_with("测试查询", top_k=3)
    
    def test_run_with_empty_query(self):
        """测试使用空查询时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.retrieval_tool("")
    
    def test_run_with_invalid_query_type(self):
        """测试使用非字符串类型的查询时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.retrieval_tool(123)
    
    def test_run_with_invalid_top_k(self):
        """测试使用无效的top_k值时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.retrieval_tool("测试查询", top_k=0)
        
        with self.assertRaises(ToolError):
            self.retrieval_tool("测试查询", top_k=-1)
        
        with self.assertRaises(ToolError):
            self.retrieval_tool("测试查询", top_k="invalid")
    
    def test_run_with_vector_store_error(self):
        """测试当向量存储出现错误时应该正确处理异常"""
        # 设置模拟对象抛出异常
        self.mock_vector_store.search.side_effect = Exception("向量存储错误")
        
        with self.assertRaises(ToolError) as context:
            self.retrieval_tool("测试查询")
        
        # 验证异常消息包含原始错误信息
        self.assertIn("向量存储错误", str(context.exception))

if __name__ == "__main__":
    unittest.main()