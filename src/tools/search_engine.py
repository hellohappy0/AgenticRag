"""
搜索引擎模块，直接使用smolagents框架的搜索引擎工具
"""
from smolagents.default_tools import DuckDuckGoSearchTool
from .base_tool import ToolError

# 直接使用smolagents的DuckDuckGoSearchTool，无需适配器
# 保持原有导出名称以维持兼容性

# 导出类
__all__ = ["DuckDuckGoSearchTool"]