import io
import sys
from unittest.mock import patch
from src.agent import AgenticRAG
from src.tool.base_tool import BaseTool
from src.tool.retrieval_tool import RetrievalTool, SearchTool
from src.doc_process.simple_processor import SimpleDocumentProcessor
from src.agent_context.agent_env import SimpleAgentEnvironment, MemoryManager
from src.prompt.prompt_manager import AgentPromptTemplates
from src.model.language_model import ModelFactory
from src.config import get_config

# 模拟向量存储类，用于演示
class MockVectorStore:
    def __init__(self, documents=None):
        self.documents = documents or []
        
    def search(self, query, top_k=3):
        # 简单的模拟搜索，返回所有文档
        return self.documents[:top_k]

# 模拟搜索引擎类
class MockSearchEngine:
    def search(self, query, max_results=5):
        # 简单的模拟搜索结果
        return [
            {"title": "搜索结果标题1", "content": "搜索结果内容1", "url": "https://example.com/1"},
            {"title": "搜索结果标题2", "content": "搜索结果内容2", "url": "https://example.com/2"}
        ]

# 创建一个简单的测试函数
def test_agentic_rag_basic_functionality():
    """
    测试Agentic RAG系统的基本功能
    """
    print("\n===== 测试Agentic RAG系统基本功能 ======")
    
    try:
        # 使用mock模型进行测试，避免实际调用API
        model_type = "mock"
        
        # 模拟文档数据
        mock_documents = [
            {
                "content": "大型语言模型（LLM）是一类基于深度学习的模型，能够理解和生成人类语言。著名的大型语言模型包括GPT-4、Claude和LLaMA等。",
                "metadata": {"source": "自然语言处理进阶"}
            }
        ]
        
        # 创建向量存储
        vector_store = MockVectorStore(mock_documents)
        
        # 创建搜索工具
        retrieval_tool = RetrievalTool(vector_store)
        
        # 创建网络搜索工具
        search_engine = MockSearchEngine()
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
        
        # 打印所有可用的模板名称，用于调试
        template_names = list(prompt_manager.templates.keys())
        print(f"可用的提示模板: {template_names}")
        
        # 创建语言模型
        if model_type.lower() == "mock":
            # 导入项目中的MockLanguageModel
            from src.mock.mock_language_model import MockLanguageModel
            model = MockLanguageModel()
        else:
            # 从配置获取通用模型参数
            model_params = {
                "max_tokens": get_config("model.common.max_tokens", 512),
                "temperature": get_config("model.common.temperature", 0.7)
            }
            
            # 创建相应类型的模型
            model = ModelFactory.create_model(model_type, **model_params)
        
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=model,
            tools=tools,
            document_processor=document_processor,
            environment=environment,
            memory_manager=memory_manager,
            prompt_manager=prompt_manager
        )
        
        print("Agentic RAG实例创建成功！")
        
        # 测试运行Agent
        test_query = "什么是大型语言模型？"
        print(f"\n测试查询: {test_query}")
        
        # 添加更多调试信息，不捕获stdout
        print("开始运行agentic_rag.run...")
        result = agentic_rag.run(test_query)
        print(f"运行结果类型: {type(result)}")
        import json
        print(f"运行结果内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 验证记忆管理器是否记录了交互
        interactions = memory_manager.get_history()
        print(f"\n记忆管理器中记录的交互数量: {len(interactions)}")
        
        # 验证环境状态是否更新
        state = environment.get_state()
        print(f"环境状态是否包含last_query: {'last_query' in state}")
        
        print("\n✅ Agentic RAG系统基本功能测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_agentic_rag_basic_functionality()
    sys.exit(0 if success else 1)