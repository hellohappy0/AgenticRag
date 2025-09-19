"""
Agent Builder模块，负责创建和初始化Agentic RAG系统
"""
from typing import Dict, List, Any, Optional
from src.agent import AgenticRAG
from src.tools import RetrievalTool, SearchTool, DuckDuckGoSearchEngine
from src.doc_process import SimpleDocumentProcessor
from src.agent_context import SimpleAgentEnvironment, SmolAgentMemoryManager
from src.prompt import AgentPromptTemplates
from src.model import ModelFactory
from src.config import get_config
from src.mock import MockLanguageModel
from src.vector_store import FAISSVectorStore


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
    
    # 创建文档处理器
    document_processor = SimpleDocumentProcessor()
    
    # 从docs目录加载实际文档
    from pathlib import Path
    
    # 获取docs目录路径
    current_dir = Path(__file__).resolve().parent.parent.parent
    docs_dir = current_dir / 'docs'
    
    # 加载并处理文档
    documents = []
    try:
        # 加载docs目录下的所有md文件
        doc_files = list(docs_dir.glob('*.md'))
        if doc_files:
            print(f"正在从docs目录加载{len(doc_files)}个文档...")
            
            # 处理每个文档
            for doc_file in doc_files:
                try:
                    # 使用文档处理器处理文件
                    processed_docs = document_processor.process(str(doc_file), chunk_size=500, chunk_overlap=50)
                    
                    # 将处理后的文档转换为向量存储所需的格式
                    for i, chunk in enumerate(processed_docs):
                        doc = {
                            'id': f"{doc_file.stem}_chunk_{i}",
                            'content': chunk,
                            'metadata': {'source': str(doc_file.name)}
                        }
                        documents.append(doc)
                except Exception as e:
                    print(f"处理文件 {doc_file.name} 时出错: {str(e)}")
            
            print(f"成功加载并处理了{len(documents)}个文档块")
    except Exception as e:
        print(f"从docs目录加载文档时出错: {str(e)}")
    
    # 如果没有加载到文档，使用模拟文档作为后备
    if not documents:
        print("未加载到有效文档，使用模拟文档...")
        documents = [
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
    
    # 创建向量存储 - 使用实际的FAISS向量数据库
    vector_store = FAISSVectorStore(documents)
    
    # 创建搜索工具
    retrieval_tool = RetrievalTool(vector_store)
    
    # 创建工具列表（默认只包含检索工具）
    tools = [retrieval_tool]
    
    # 尝试创建网络搜索工具（如果ddgs库可用）
    try:
        search_engine = DuckDuckGoSearchEngine()
        search_tool = SearchTool(search_engine)
        tools.append(search_tool)
        print("成功初始化网络搜索工具")
    except ImportError as e:
        print(f"警告: 无法初始化网络搜索工具: {str(e)}")
        print("提示: 可以通过 'pip install duckduckgo-search' 安装必要的库来启用网络搜索功能")
    
    # 创建代理环境
    environment = SimpleAgentEnvironment()
    
    # 创建记忆管理器
    memory_manager = SmolAgentMemoryManager()
    
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


# 导出函数
__all__ = ["create_agentic_rag"]