import unittest
from src.tools.base_tool import BaseTool, ToolError
from abc import ABC

class TestBaseTool(unittest.TestCase):
    
    def test_base_tool_is_abstract(self):
        """测试BaseTool是否为抽象基类"""
        # 验证BaseTool是ABC的子类
        self.assertTrue(issubclass(BaseTool, ABC))
        
        # 验证尝试实例化BaseTool会抛出TypeError（因为它是抽象类）
        with self.assertRaises(TypeError):
            BaseTool("test_tool", "test_description")
    
    def test_concrete_tool_implementation(self):
        """测试实现BaseTool的具体工具类"""
        # 创建一个具体的工具实现
        class ConcreteTool(BaseTool):
            def run(self, **kwargs):
                return {"result": "success", "kwargs": kwargs}
        
        # 实例化具体工具
        tool = ConcreteTool("test_tool", "test_description")
        
        # 验证初始化参数
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "test_description")
        
        # 验证get_tool_info方法
        tool_info = tool.get_tool_info()
        self.assertEqual(tool_info["name"], "test_tool")
        self.assertEqual(tool_info["description"], "test_description")
        
        # 验证run方法
        result = tool.run(param1="value1", param2="value2")
        self.assertEqual(result["result"], "success")
        self.assertEqual(result["kwargs"], {"param1": "value1", "param2": "value2"})
    
    def test_tool_error(self):
        """测试ToolError异常类"""
        # 创建一个会抛出ToolError的工具
        class ErrorTool(BaseTool):
            def run(self, **kwargs):
                raise ToolError("测试错误")
        
        tool = ErrorTool("error_tool", "error_description")
        
        # 验证ToolError可以被抛出和捕获
        with self.assertRaises(ToolError) as context:
            tool.run()
        
        # 验证异常消息
        self.assertEqual(str(context.exception), "测试错误")
        
        # 验证ToolError是Exception的子类
        self.assertTrue(issubclass(ToolError, Exception))

if __name__ == "__main__":
    unittest.main()