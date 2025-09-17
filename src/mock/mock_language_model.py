from typing import Dict, List, Any


class MockLanguageModel:
    """
    模拟语言模型，用于在真实模型不可用时进行演示
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        模拟生成响应
        
        @param prompt: 输入提示
        @param kwargs: 模型参数
        @return: 模拟的响应
        """
        if "人工智能" in prompt or "AI" in prompt:
            return "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。它包括机器学习、自然语言处理等多个研究领域。"
        elif "机器学习" in prompt:
            return "机器学习是人工智能的一个子集，专注于开发能够从数据中学习的算法。常见方法包括监督学习、无监督学习和强化学习。"
        elif "大型语言模型" in prompt or "LLM" in prompt:
            return "大型语言模型是一类基于深度学习的模型，能够理解和生成人类语言。例如GPT-4、Claude和LLaMA等。"
        else:
            return "这是一个基于模拟模型的响应。在实际应用中，这里会返回真实语言模型生成的内容。"
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        模拟生成带工具调用的响应
        
        @param prompt: 输入提示
        @param tools: 可用工具列表
        @param kwargs: 模型参数
        @return: 模拟的响应
        """
        # 检查是否包含查询词
        query_keywords = ["人工智能", "AI", "机器学习", "大型语言模型", "LLM"]
        
        # 检查是否有检索工具
        has_retrieve_tool = any(tool.get("name") == "retrieve_documents" for tool in tools)
        has_search_tool = any(tool.get("name") == "web_search" for tool in tools)
        
        for keyword in query_keywords:
            if keyword in prompt:
                # 优先使用检索工具，如果有
                if has_retrieve_tool:
                    return {
                        "response": "<|FunctionCallBegin|>[{\"name\": \"retrieve_documents\", \"parameters\": {\"query\": \"" + keyword + "\", \"top_k\": 3}}]<|FunctionCallEnd|>",
                        "tool_calls": [{"name": "retrieve_documents", "parameters": {"query": keyword, "top_k": 3}}],
                        "raw_output": ""
                    }
                # 否则使用搜索工具
                elif has_search_tool:
                    return {
                        "response": "<|FunctionCallBegin|>[{\"name\": \"web_search\", \"parameters\": {\"query\": \"" + prompt + "\", \"max_results\": 3}}]<|FunctionCallEnd|>",
                        "tool_calls": [{"name": "web_search", "parameters": {"query": prompt, "max_results": 3}}],
                        "raw_output": ""
                    }
        
        # 不使用工具，直接回答
        return {
            "response": self.generate(prompt, **kwargs),
            "tool_calls": [],
            "raw_output": ""
        }