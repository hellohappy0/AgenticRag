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
            BaseTool()
    
    def test_concrete_tool_implementation(self):
        """测试实现BaseTool的具体工具类"""
        # 创建一个具体的工具实现
        class ConcreteTool(BaseTool):
            def __init__(self):
                # 直接设置属性
                self.name = "test_tool"
                self.description = "test_description"
            
            def forward(self, **kwargs):
                return {"result": "success", "kwargs": kwargs}
            
            def __call__(self, **kwargs):
                return self.forward(**kwargs)
        
        # 实例化具体工具
        tool = ConcreteTool()
        
        # 验证属性设置
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "test_description")
        
        # 验证forward方法
        result = tool.forward(param1="value1", param2="value2")
        self.assertEqual(result["result"], "success")
        self.assertEqual(result["kwargs"], {"param1": "value1", "param2": "value2"})
        
        # 验证__call__方法
        result = tool(param1="value1", param2="value2")
        self.assertEqual(result["result"], "success")
        self.assertEqual(result["kwargs"], {"param1": "value1", "param2": "value2"})
    
    def test_tool_error(self):
        """测试ToolError异常类"""
        # 创建一个会抛出ToolError的工具
        class ErrorTool(BaseTool):
            def __init__(self):
                # 直接设置属性
                self.name = "error_tool"
                self.description = "error_description"
            
            def forward(self, **kwargs):
                raise ToolError("测试错误")
            
            def __call__(self, **kwargs):
                return self.forward(**kwargs)
        
        tool = ErrorTool()
        
        # 验证ToolError可以被抛出和捕获
        with self.assertRaises(ToolError) as context:
            tool.forward()
        
        # 验证异常消息
        self.assertEqual(str(context.exception), "测试错误")
        
        # 验证ToolError是Exception的子类
        self.assertTrue(issubclass(ToolError, Exception))

if __name__ == "__main__":
    unittest.main()