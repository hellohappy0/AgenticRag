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
            description="从知识库中检索与查询相关的文档片段，用于回答问题或获取相关信息"
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
            description="执行网络搜索以获取最新信息或外部知识"
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