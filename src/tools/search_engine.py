"""
搜索引擎模块，提供各种搜索引擎的实现
"""
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入ddgs库
try:
    from ddgs import DDGS
except ImportError:
    logger.warning("ddgs模块未安装，网络搜索功能将不可用")
    DDGS = None


class DuckDuckGoSearchEngine:
    """
    基于DuckDuckGo的真实搜索引擎实现
    """
    
    def __init__(self):
        """
        初始化DuckDuckGo搜索引擎
        """
        if DDGS is None:
            raise ImportError("ddgs模块未安装，无法初始化DuckDuckGo搜索引擎")
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        执行真实的网络搜索
        
        @param query: 搜索查询
        @param max_results: 返回的最大结果数量
        @return: 搜索结果列表
        """
        try:
            # 参数验证
            if not query or not isinstance(query, str):
                raise ValueError("搜索查询必须是非空字符串")
            
            if not isinstance(max_results, int) or max_results <= 0:
                raise ValueError("max_results必须是正整数")
                
            # 使用ddgs执行搜索
            results = list(self.ddgs.text(query, max_results=max_results))
            
            # 格式化结果为统一格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
            
            return formatted_results
        except ValueError as ve:
            logger.error(f"参数错误: {str(ve)}")
            return []
        except Exception as e:
            logger.error(f"搜索过程中出错: {str(e)}")
            return []


# 导出类
__all__ = ["DuckDuckGoSearchEngine"]