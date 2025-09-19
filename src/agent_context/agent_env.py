from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class AgentEnvironment(ABC):
    """
    代理环境基类，定义了代理运行环境的基本接口
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态
        
        @return: 环境状态字典
        """
        pass
    
    @abstractmethod
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新环境状态
        
        @param updates: 要更新的状态字段
        @return: 更新后的环境状态
        """
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        重置环境状态
        
        @return: 重置后的环境状态
        """
        pass


class SimpleAgentEnvironment(AgentEnvironment):
    """
    简单的代理环境实现
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        初始化代理环境
        
        @param initial_state: 初始环境状态
        """
        self._state = initial_state or {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态
        
        @return: 环境状态字典的深拷贝
        """
        # 返回状态的深拷贝，避免外部直接修改内部状态
        return self._state.copy()
    
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新环境状态
        
        @param updates: 要更新的状态字段
        @return: 更新后的环境状态
        """
        if not isinstance(updates, dict):
            raise TypeError("updates必须是字典类型")
        
        # 更新状态
        self._state.update(updates)
        
        # 返回更新后的状态
        return self.get_state()
    
    def reset(self) -> Dict[str, Any]:
        """
        重置环境状态
        
        @return: 重置后的环境状态
        """
        self._state = {}
        return self.get_state()