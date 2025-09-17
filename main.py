from typing import Dict, Any, List
from src.agent import AgenticRAG
from src.tool import BaseTool, RetrievalTool, SearchTool
from src.doc_process import SimpleDocumentProcessor
from src.agent_context import SimpleAgentEnvironment, MemoryManager
from src.prompt import AgentPromptTemplates
from src.model import ModelFactory
from src.config import get_config, config_manager


# 模拟向量存储类，用于演示
class MockVectorStore:
    """
    模拟向量存储，用于演示检索功能
    """
    
    def __init__(self, documents: List[Dict[str, Any]]):
        """
        初始化模拟向量存储
        
        @param documents: 文档列表
        """
        self.documents = documents
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        模拟搜索功能
        
        @param query: 查询文本
        @param top_k: 返回的结果数量
        @return: 搜索结果列表
        """
        # 简单的基于字符串匹配的搜索
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                results.append(doc)
            
            if len(results) >= top_k:
                break
        
        # 如果没有匹配结果，返回前top_k个文档
        if not results and len(self.documents) > 0:
            results = self.documents[:min(top_k, len(self.documents))]
        
        return results


# 使用ddgs实现的真实搜索引擎
from ddgs import DDGS

class DuckDuckGoSearchEngine:
    """
    基于DuckDuckGo的真实搜索引擎实现
    """
    
    def __init__(self):
        """
        初始化DuckDuckGo搜索引擎
        """
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


# 为了向后兼容，保留MockSearchEngine类
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


def create_agentic_rag(model_type: str = "mock", api_key: str = None, custom_model: str = None) -> AgenticRAG:
    """
    创建并初始化Agentic RAG系统
    
    @param model_type: 模型类型 (mock, smolagent, tongyi等)
    @param api_key: API密钥（如果未指定，将从配置中获取）
    @param custom_model: 自定义模型名称（用于Ollama，如果未指定，将从配置中获取）
    @return: 初始化后的Agentic RAG实例
    """
    # 加载配置
    # 从配置文件或环境变量获取模型类型，如果未指定
    if not model_type or model_type.lower() == "auto":
        model_type = get_config("model.type", "mock")
    
    # 创建模拟文档数据
    mock_documents = [
        {
            "id": "doc1",
            "content": "人工智能（AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。人工智能的研究领域包括机器学习、自然语言处理、计算机视觉等。",
            "metadata": {"source": "AI入门指南"}
        },
        {
            "id": "doc2",
            "content": "机器学习是人工智能的一个子集，专注于开发能够从数据中学习的算法。常见的机器学习方法包括监督学习、无监督学习和强化学习。",
            "metadata": {"source": "机器学习基础"}
        },
        {
            "id": "doc3",
            "content": "大型语言模型（LLM）是一类基于深度学习的模型，能够理解和生成人类语言。著名的大型语言模型包括GPT-4、Claude和LLaMA等。",
            "metadata": {"source": "自然语言处理进阶"}
        }
    ]
    
    # 创建向量存储
    vector_store = MockVectorStore(mock_documents)
    
    # 创建搜索工具
    retrieval_tool = RetrievalTool(vector_store)
    
    # 创建网络搜索工具
    search_engine = DuckDuckGoSearchEngine()
    search_tool = SearchTool(search_engine)
    
    # 添加所有工具
    tools = [retrieval_tool, search_tool]
    
    # 创建文档处理器
    document_processor = SimpleDocumentProcessor()
    
    # 创建代理环境
    environment = SimpleAgentEnvironment()
    
    # 创建记忆管理器
    memory_manager = MemoryManager()
    
    # 创建提示管理器
    prompt_manager = AgentPromptTemplates.create_prompt_manager()
    
    # 创建语言模型
    try:
        if model_type.lower() == "mock":
            model = MockLanguageModel()
        else:
            # 从配置获取通用模型参数
            model_params = {
                "max_tokens": get_config("model.common.max_tokens", 512),
                "temperature": get_config("model.common.temperature", 0.7)
            }
            
            if model_type.lower() == "tongyi":
                # 通义千问模型参数
                model_params["model_name"] = get_config("model.tongyi.model_name", "qwen-max")
                # 优先级: 函数参数 > 环境变量/配置文件 > 默认值
                model_params["api_key"] = api_key or get_config("model.tongyi.api_key", None)
                if not model_params["api_key"]:
                    raise ValueError("使用通义千问模型需要提供API密钥")
            elif model_type.lower() == "smolagent":
                # smolAgent模型参数
                model_params["model_name"] = get_config("model.smolagent.model_name", "gpt2")
                model_params["api_key"] = api_key or get_config("model.smolagent.api_key", None)
            elif model_type.lower() == "ollama":
                # Ollama模型参数
                model_params["model_name"] = custom_model or get_config("model.ollama.model_name", "llama3")
                model_params["base_url"] = get_config("model.ollama.base_url", "http://localhost:11434")
            
            model = ModelFactory.create_model(
                model_type,
                **model_params
            )
    except Exception as e:
        print(f"创建{model_type}模型失败，使用模拟模型: {str(e)}")
        # 如果指定模型不可用，使用模拟模型
        model = MockLanguageModel()
    
    # 创建并返回Agentic RAG实例
    agentic_rag = AgenticRAG(
        model=model,
        tools=tools,
        document_processor=document_processor,
        environment=environment,
        prompt_manager=prompt_manager,
        memory_manager=memory_manager,
        max_retries=3,
        max_tool_calls=5
    )
    
    return agentic_rag


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


def main():
    """
    主函数，演示Agentic RAG的使用
    """
    print("初始化Agentic RAG系统...")
    print("系统支持从配置文件(config.yaml)和环境变量加载配置，环境变量优先级高于配置文件。")
    print("环境变量格式: AGENTIC_RAG_{配置键}_...，使用下划线分隔嵌套键。例如: AGENTIC_RAG_MODEL_TONGYI_API_KEY")
    
    # 获取用户选择的模型类型
    # 如果配置中已经设置了模型类型，询问用户是否使用配置中的值
    config_model_type = get_config("model.type", "")
    if config_model_type:
        use_config = input(f"检测到配置文件中的模型类型为 '{config_model_type}'，是否使用？(y/n，默认y): ").strip().lower() or "y"
        if use_config == "y":
            model_type = config_model_type
        else:
            model_type = input("请选择模型类型 (mock/smolagent/tongyi/ollama/auto，默认mock): ").strip().lower() or "mock"
    else:
        model_type = input("请选择模型类型 (mock/smolagent/tongyi/ollama/auto，默认mock): ").strip().lower() or "mock"
    
    api_key = None
    custom_model = None
    
    if model_type != "mock" and model_type != "auto":
        if model_type == "tongyi":
            # 检查配置中是否已有API密钥
            config_api_key = get_config("model.tongyi.api_key", "")
            if config_api_key:
                use_config_key = input(f"检测到配置中的通义千问API密钥，是否使用？(y/n，默认y): ").strip().lower() or "y"
                if use_config_key != "y":
                    api_key = input("请输入阿里云通义千问的API密钥: ").strip()
            else:
                api_key = input("请输入阿里云通义千问的API密钥: ").strip()
        elif model_type == "ollama":
            # 检查配置中是否已有模型名称
            config_model_name = get_config("model.ollama.model_name", "")
            if config_model_name:
                use_config_model = input(f"检测到配置中的Ollama模型名称为 '{config_model_name}'，是否使用？(y/n，默认y): ").strip().lower() or "y"
                if use_config_model != "y":
                    custom_model_input = input("请输入Ollama模型名称 (默认llama3，可选): ").strip()
                    if custom_model_input:
                        custom_model = custom_model_input
            else:
                custom_model_input = input("请输入Ollama模型名称 (默认llama3，可选): ").strip()
                if custom_model_input:
                    custom_model = custom_model_input
        else:
            # 其他模型类型的API密钥处理
            config_api_key = get_config(f"model.{model_type}.api_key", "")
            if not config_api_key:
                api_key_input = input(f"请输入{model_type}的API密钥 (可选): ").strip()
                if api_key_input:
                    api_key = api_key_input
    
    # 创建Agentic RAG实例
    agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key, custom_model=custom_model)
    
    print("\nAgentic RAG系统已初始化完成，您可以开始提问。输入'quit'退出。\n")
    
    while True:
        try:
            # 获取用户输入
            query = input("您的问题: ")
            
            if query.lower() in ["quit", "exit", "退出"]:
                print("感谢使用Agentic RAG系统，再见！")
                break
            
            # 运行Agentic RAG处理查询
            print("\n正在处理您的问题...\n")
            result = agentic_rag.run(query)
            
            # 显示结果
            print("=== 回答 ===")
            print(result.get("answer", "没有生成回答"))
            print("\n=== 状态 ===")
            print(f"状态: {result.get('status', 'unknown')}")
            print(f"工具调用次数: {len(result.get('tool_calls', []))}")
            print(f"重试次数: {result.get('retries', 0)}")
            
            # 如果有自我批评，也显示
            if "critique" in result:
                print("\n=== 自我批评 ===")
                print(result["critique"])
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()