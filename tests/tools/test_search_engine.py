import unittest
from unittest.mock import Mock, patch
from src.tools import SearchTool
from src.tools.search_engine import DuckDuckGoSearchTool
from src.tools.base_tool import ToolError

class TestSearchEngine(unittest.TestCase):
    """
    测试搜索相关工具的功能
    """
    
    def setUp(self):
        """测试前的设置"""
        # 创建模拟的搜索引擎对象用于SearchTool测试
        self.mock_search_engine = Mock()
        # 创建SearchTool实例
        self.search_tool = SearchTool(self.mock_search_engine)
        # 测试查询
        self.test_query = "人工智能最新进展"
        # 模拟搜索结果
        self.mock_results = [
            {"title": "测试结果1", "url": "https://example.com/1", "snippet": "这是第一个测试结果的摘要"},
            {"title": "测试结果2", "url": "https://example.com/2", "snippet": "这是第二个测试结果的摘要"}
        ]
    
    def test_search_tool_initialization(self):
        """测试SearchTool的初始化是否正确"""
        self.assertEqual(self.search_tool.name, "web_search")
        self.assertIn("当本地知识库无法提供所需信息", self.search_tool.description)
        self.assertEqual(self.search_tool.search_engine, self.mock_search_engine)
    
    def test_run_with_valid_query(self):
        """测试SearchTool使用有效查询运行"""
        # 设置模拟对象的返回值
        self.mock_search_engine.search.return_value = self.mock_results
        
        # 执行搜索
        result = self.search_tool(self.test_query, max_results=2)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["query"], self.test_query)
        self.assertEqual(result["results"], self.mock_results)
        self.assertEqual(result["total_results"], 2)
        
        # 验证搜索引擎的search方法被正确调用
        self.mock_search_engine.search.assert_called_once_with(self.test_query, max_results=2)
    
    def test_run_with_default_max_results(self):
        """测试SearchTool使用默认的max_results参数"""
        # 设置模拟对象的返回值
        self.mock_search_engine.search.return_value = self.mock_results
        
        # 执行搜索，不指定max_results
        result = self.search_tool(self.test_query)
        
        # 验证搜索引擎的search方法被调用时使用了默认的max_results=3
        self.mock_search_engine.search.assert_called_once_with(self.test_query, max_results=3)
    
    def test_run_with_empty_query(self):
        """测试SearchTool使用空查询时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.search_tool("")
    
    def test_run_with_invalid_query_type(self):
        """测试SearchTool使用非字符串类型的查询时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.search_tool(123)
    
    def test_run_with_invalid_max_results(self):
        """测试SearchTool使用无效的max_results值时应该抛出异常"""
        with self.assertRaises(ToolError):
            self.search_tool(self.test_query, max_results=0)
        
        with self.assertRaises(ToolError):
            self.search_tool(self.test_query, max_results=-1)
        
        with self.assertRaises(ToolError):
            self.search_tool(self.test_query, max_results="invalid")
    
    def test_run_with_search_engine_error(self):
        """测试当搜索引擎出现错误时应该正确处理异常"""
        # 设置模拟对象抛出异常
        self.mock_search_engine.search.side_effect = Exception("搜索引擎错误")
        
        with self.assertRaises(ToolError) as context:
            self.search_tool(self.test_query)
        
        # 验证异常消息包含原始错误信息
        self.assertIn("搜索引擎错误", str(context.exception))

if __name__ == "__main__":
    unittest.main()