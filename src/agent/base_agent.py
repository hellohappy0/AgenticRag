from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """
    代理基类，定义了代理的基本接口
    """
    
    @abstractmethod
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行代理处理查询
        
        @param query: 用户查询
        @param kwargs: 其他参数
        @return: 代理处理结果
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        重置代理状态
        """
        pass