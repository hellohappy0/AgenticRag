from src.tool import SearchTool
from main import DuckDuckGoSearchEngine

"""
测试DuckDuckGoSearchEngine和SearchTool的功能
"""

if __name__ == "__main__":
    print("测试DuckDuckGo搜索引擎功能...")
    
    # 创建搜索引擎实例
    search_engine = DuckDuckGoSearchEngine()
    
    # 执行直接搜索测试
    print("\n1. 直接使用搜索引擎测试:")
    query = "人工智能最新进展"
    print(f"搜索查询: {query}")
    
    try:
        results = search_engine.search(query, max_results=3)
        print(f"找到 {len(results)} 条结果:")
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"标题: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"摘要: {result['snippet'][:100]}...")  # 只显示前100个字符
    except Exception as e:
        print(f"直接搜索测试失败: {str(e)}")
    
    # 执行通过SearchTool测试
    print("\n2. 通过SearchTool测试:")
    search_tool = SearchTool(search_engine)
    
    try:
        tool_result = search_tool.run(query, max_results=3)
        print(f"工具调用状态: {tool_result['status']}")
        print(f"工具返回结果数: {tool_result['total_results']}")
        
        if tool_result['status'] == 'success' and tool_result['results']:
            print("工具返回的第一条结果:")
            first_result = tool_result['results'][0]
            print(f"标题: {first_result['title']}")
            print(f"URL: {first_result['url']}")
            print(f"摘要: {first_result['snippet'][:100]}...")
    except Exception as e:
        print(f"工具调用测试失败: {str(e)}")
    
    print("\n测试完成！")