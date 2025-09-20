from smolagents.tools import BaseTool
from smolagents.utils import AgentToolExecutionError

class SimpleLogger:
    """简单的logger实现，满足smolagents的接口要求"""
    def log_error(self, message: str):
        pass

# 创建一个全局的简单logger实例
simple_logger = SimpleLogger()

# 定义ToolError类，自动提供logger参数
class ToolError(AgentToolExecutionError):
    def __init__(self, message: str):
        super().__init__(message, simple_logger)

# 导出必要的类
export = [BaseTool, ToolError]