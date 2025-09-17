from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """
    工具基类，定义了所有工具的基本接口
    """
    
    def __init__(self, name: str, description: str):
        """
        初始化工具
        
        @param name: 工具名称
        @param description: 工具描述，用于提示Agent如何使用该工具
        """
        self.name = name
        self.description = description
        
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        运行工具的抽象方法
        
        @param kwargs: 工具所需的参数
        @return: 工具运行的结果，以字典形式返回
        """
        pass
    
    def get_tool_info(self) -> Dict[str, str]:
        """
        获取工具的基本信息
        
        @return: 包含工具名称和描述的字典
        """
        return {
            "name": self.name,
            "description": self.description
        }


class ToolError(Exception):
    """\工具执行过程中出现的异常"""
    pass