from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Type

T = TypeVar('T')


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
        
    def validate_param(self, param: Any, param_name: str, param_type: Type[T], 
                      required: bool = True, validator: Optional[callable] = None) -> T:
        """
        参数验证辅助函数
        
        @param param: 待验证的参数值
        @param param_name: 参数名称
        @param param_type: 参数类型
        @param required: 是否为必需参数
        @param validator: 自定义验证器函数
        @return: 验证通过的参数值
        @raises ValueError: 当参数验证失败时抛出异常
        """
        # 检查是否为必需参数且为None
        if required and param is None:
            raise ValueError(f"参数 '{param_name}' 是必需的")
        
        # 如果参数为None且不是必需的，则直接返回
        if param is None:
            return param
        
        # 检查参数类型
        if not isinstance(param, param_type):
            raise ValueError(f"参数 '{param_name}' 必须是 {param_type.__name__} 类型")
        
        # 应用自定义验证器
        if validator and not validator(param):
            raise ValueError(f"参数 '{param_name}' 验证失败")
        
        return param


class ToolError(Exception):
    """工具执行过程中出现的异常"""
    pass