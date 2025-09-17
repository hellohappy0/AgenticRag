from typing import List, Dict, Any


class MockSearchEngine:
    """
    模拟搜索引擎，用于演示网络搜索功能
    """
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        模拟网络搜索
        
        @param query: 搜索查询
        @param max_results: 返回的最大结果数量
        @return: 搜索结果列表
        """
        # 模拟搜索结果
        mock_results = [
            {
                "title": f"关于{query}的搜索结果1",
                "url": f"https://example.com/search1?query={query}",
                "snippet": f"这是关于{query}的搜索结果摘要1。包含了相关的信息和内容。"
            },
            {
                "title": f"关于{query}的搜索结果2",
                "url": f"https://example.com/search2?query={query}",
                "snippet": f"这是关于{query}的搜索结果摘要2。提供了更多详细的背景信息。"
            },
            {
                "title": f"关于{query}的搜索结果3",
                "url": f"https://example.com/search3?query={query}",
                "snippet": f"这是关于{query}的搜索结果摘要3。包含了最新的研究进展。"
            }
        ]
        
        return mock_results[:max_results]