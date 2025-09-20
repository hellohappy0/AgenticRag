from typing import Dict, Any
from smolagents.tools import BaseTool
from .base_tool import ToolError


class RetrievalTool(BaseTool):
    """
    检索工具，用于从知识库中检索相关文档
    """
    
    def __init__(self, vector_store):
        """
        初始化检索工具
        
        @param vector_store: 向量存储对象，用于执行实际的检索操作
        """
        # 直接设置属性，不调用父类带参数的__init__
        self.name = "retrieve_documents"
        self.description = "当需要从本地知识库中获取准确、可靠的已有信息时使用此工具。适用于事实性问题、已有文档中存在的知识查询、需要详细解释的专业概念等场景。如果问题涉及的信息可能存在于已有的知识库中，请优先使用此工具。"
        self.vector_store = vector_store
    
    def forward(self, query: str, top_k: int = 3, **kwargs) -> Dict[str, Any]:
        """
        执行检索操作
        
        @param query: 查询文本
        @param top_k: 返回的相关文档数量，默认为3
        @return: 包含检索结果的字典
        @raises ToolError: 当检索失败时抛出异常
        """
        try:
            # 执行检索
            results = self.vector_store.search(query, top_k=top_k)
            
            # 格式化结果
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            raise ToolError(f"检索过程中出错: {str(e)}")
    
    def get_tool_info(self) -> Dict[str, str]:
        """
        获取工具信息
        
        @return: 包含工具名称和描述的字典
        """
        return {
            "name": self.name,
            "description": self.description
        }
    
    def get_tool_info(self) -> Dict[str, str]:
        """
        获取工具信息
        
        @return: 包含工具名称和描述的字典
        """
        return {
            "name": self.name,
            "description": self.description
        }
    
    # 保持向后兼容性
    __call__ = forward
    run = forward


class SearchTool(BaseTool):
    """
    搜索工具，用于执行网络搜索或其他外部搜索
    """
    
    def __init__(self, search_engine):
        """
        初始化搜索工具
        
        @param search_engine: 搜索引擎对象，用于执行实际的搜索操作
        """
        # 直接设置属性，不调用父类带参数的__init__
        self.name = "web_search"
        self.description = "当本地知识库无法提供所需信息，或需要获取最新事件、实时数据、外部资源或最新研究进展时使用此工具。适用于时效性问题、本地知识未覆盖的领域、需要最新发展动态的查询等场景。在确定本地知识库无法满足需求时使用此工具。"
        self.search_engine = search_engine
    
    def forward(self, query: str, max_results: int = 3, **kwargs) -> Dict[str, Any]:
        """
        执行搜索操作
        
        @param query: 搜索查询
        @param max_results: 返回的最大结果数量，默认为3
        @return: 包含搜索结果的字典
        @raises ToolError: 当搜索失败时抛出异常
        """
        try:
            # 执行搜索
            results = self.search_engine.search(query, max_results=max_results)
            
            # 格式化结果
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            raise ToolError(f"搜索过程中出错: {str(e)}")
    
    # 保持向后兼容性
    __call__ = forward


# 导出类
__all__ = ["RetrievalTool", "SearchTool"]