"""
搜索引擎模块，提供各种搜索引擎的实现
"""
from typing import Dict, List, Any

# 尝试导入ddgs库
try:
    from ddgs import DDGS
except ImportError:
    print("警告: ddgs模块未安装，网络搜索功能将不可用")
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
        except Exception as e:
            print(f"搜索过程中出错: {str(e)}")
            # 出错时返回空结果
            return []


# 导出类
__all__ = ["DuckDuckGoSearchEngine"]