from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolError


class RetrievalTool(BaseTool):
    """
    检索工具，用于从知识库中检索相关文档
    """
    
    def __init__(self, vector_store):
        """
        初始化检索工具
        
        @param vector_store: 向量存储对象，用于执行实际的检索操作
        """
        super().__init__(
            name="retrieve_documents",
            description="当需要从本地知识库中获取准确、可靠的已有信息时使用此工具。适用于事实性问题、已有文档中存在的知识查询、需要详细解释的专业概念等场景。如果问题涉及的信息可能存在于已有的知识库中，请优先使用此工具。"
        )
        self.vector_store = vector_store
    
    def run(self, query: str, top_k: int = 3, **kwargs) -> Dict[str, Any]:
        """
        运行检索操作
        
        @param query: 查询文本
        @param top_k: 返回的相关文档数量，默认为3
        @param kwargs: 其他参数
        @return: 包含检索结果的字典
        @raises ToolError: 当检索失败时抛出异常
        """
        try:
            # 参数验证
            if not query or not isinstance(query, str):
                raise ValueError("查询文本必须是非空字符串")
            
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k必须是正整数")
            
            # 执行检索
            results = self.vector_store.search(query, top_k=top_k)
            
            # 格式化结果
            formatted_results = {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results)
            }
            
            return formatted_results
        except Exception as e:
            raise ToolError(f"检索过程中出错: {str(e)}")


class SearchTool(BaseTool):
    """
    搜索工具，用于执行网络搜索或其他外部搜索
    """
    
    def __init__(self, search_engine):
        """
        初始化搜索工具
        
        @param search_engine: 搜索引擎对象，用于执行实际的搜索操作
        """
        super().__init__(
            name="web_search",
            description="当本地知识库无法提供所需信息，或需要获取最新事件、实时数据、外部资源或最新研究进展时使用此工具。适用于时效性问题、本地知识未覆盖的领域、需要最新发展动态的查询等场景。在确定本地知识库无法满足需求时使用此工具。"
        )
        self.search_engine = search_engine
    
    def run(self, query: str, max_results: int = 3, **kwargs) -> Dict[str, Any]:
        """
        运行搜索操作
        
        @param query: 搜索查询
        @param max_results: 返回的最大结果数量，默认为3
        @param kwargs: 其他参数
        @return: 包含搜索结果的字典
        @raises ToolError: 当搜索失败时抛出异常
        """
        try:
            # 参数验证
            if not query or not isinstance(query, str):
                raise ValueError("搜索查询必须是非空字符串")
            
            if not isinstance(max_results, int) or max_results <= 0:
                raise ValueError("max_results必须是正整数")
            
            # 执行搜索
            results = self.search_engine.search(query, max_results=max_results)
            
            # 打印调试信息
            print(f"\n调试信息 - web_search工具调用结果: {results}\n")
            
            # 格式化结果
            formatted_results = {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results)
            }
            
            return formatted_results
        except Exception as e:
            raise ToolError(f"搜索过程中出错: {str(e)}")