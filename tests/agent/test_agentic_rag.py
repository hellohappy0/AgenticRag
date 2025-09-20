import unittest
from unittest.mock import Mock, patch, MagicMock
from src.agent.agent_refactored import AgenticRAG
from src.tools.retrieval_tool import RetrievalTool, SearchTool
from src.doc_process.simple_processor import SimpleDocumentProcessor
from src.agent_context.agent_env import SimpleAgentEnvironment
from src.memory.smol_memory_manager import SmolAgentMemoryManager
from src.prompt.prompt_manager import AgentPromptTemplates
from src.model.language_model import BaseLanguageModel

class TestAgenticRAG(unittest.TestCase):
    """
    测试Agentic RAG系统的功能
    """
    
    def setUp(self):
        """测试前的设置"""
        # 模拟文档数据
        self.mock_documents = [
            {
                "content": "大型语言模型（LLM）是一类基于深度学习的模型，能够理解和生成人类语言。著名的大型语言模型包括GPT-4、Claude和LLaMA等。",
                "metadata": {"source": "自然语言处理进阶"}
            }
        ]
        
        # 创建模拟的向量存储
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.search.return_value = self.mock_documents
        
        # 创建模拟的搜索引擎
        self.mock_search_engine = MagicMock()
        self.mock_search_engine.search.return_value = [
            {"title": "搜索结果标题1", "content": "搜索结果内容1", "url": "https://example.com/1"},
            {"title": "搜索结果标题2", "content": "搜索结果内容2", "url": "https://example.com/2"}
        ]
        
        # 创建工具列表
        self.retrieval_tool = RetrievalTool(self.mock_vector_store)
        self.search_tool = SearchTool(self.mock_search_engine)
        self.tools = [self.retrieval_tool, self.search_tool]
        
        # 创建其他组件
        self.document_processor = SimpleDocumentProcessor()
        self.environment = SimpleAgentEnvironment()
        self.memory_manager = SmolAgentMemoryManager()
        self.prompt_manager = AgentPromptTemplates.create_prompt_manager()
        
        # 创建模拟语言模型
        self.mock_model = MagicMock(spec=BaseLanguageModel)
        # 设置模型方法的默认返回值
        self.mock_model.generate.return_value = "这是一个测试回答。大型语言模型是一类基于深度学习的模型。"
        self.mock_model.generate_with_tools.return_value = {
            "response": "<|FunctionCallBegin|>[{\"name\": \"retrieve_documents\", \"parameters\": {\"query\": \"大型语言模型\", \"top_k\": 3}}]<|FunctionCallEnd|>",
            "tool_calls": [{"name": "retrieve_documents", "parameters": {"query": "大型语言模型", "top_k": 3}}],
            "raw_output": ""
        }
        
        # 测试查询
        self.test_query = "什么是大型语言模型？"
    
    def test_agent_initialization(self):
        """测试Agentic RAG的初始化"""
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.document_processor,
            environment=self.environment,
            memory_manager=self.memory_manager,
            prompt_manager=self.prompt_manager
        )
        
        # 验证初始化的组件是否正确
        self.assertEqual(agentic_rag.model, self.mock_model)
        # 验证tools字典包含预期的工具
        self.assertIsInstance(agentic_rag.tools, dict)
        self.assertEqual(len(agentic_rag.tools), len(self.tools))
        for tool in self.tools:
            self.assertIn(tool.name, agentic_rag.tools)
            self.assertEqual(agentic_rag.tools[tool.name], tool)
        self.assertEqual(agentic_rag.document_processor, self.document_processor)
        self.assertEqual(agentic_rag.environment, self.environment)
        self.assertEqual(agentic_rag.memory_manager, self.memory_manager)
        self.assertEqual(agentic_rag.prompt_manager, self.prompt_manager)
    
    def test_agent_run_basic_functionality(self):
        """测试Agentic RAG的基本运行功能"""
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.document_processor,
            environment=self.environment,
            memory_manager=self.memory_manager,
            prompt_manager=self.prompt_manager
        )
        
        # 使用patch直接模拟_agent_loop方法，让它返回包含正确答案的状态
        with patch.object(agentic_rag, '_agent_loop') as mock_agent_loop:
            # 设置模拟_agent_loop方法返回的状态
            mock_state = {
                "query": self.test_query,
                "original_query": self.test_query,
                "context": "",
                "answer": "大型语言模型是一类基于深度学习的模型。",
                "tool_calls": [],
                "retries": 0,
                "iterations": 0,
                "max_iterations": 3,
                "status": "success",
                "query_analysis": {},
                "tool_usage_history": [],
                "errors": []
            }
            mock_agent_loop.return_value = mock_state
            
            # 运行Agent
            result = agentic_rag.run(self.test_query)
            
            # 验证结果
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIn("status", result)
            self.assertEqual(result["answer"], "大型语言模型是一类基于深度学习的模型。")
            self.assertEqual(result["status"], "success")
    
    def test_tool_selection_and_execution(self):
        """测试工具选择和执行流程
        
        注意：这个测试与重构后的AgenticRAG代码不兼容，已经在tests/agent/test_agentic_rag_refactored.py中
        提供了一个更新的、与重构代码兼容的测试版本。请使用新的测试文件中的同名测试方法。
        """
        # 跳过测试，使用新的测试文件中的同名测试方法
        self.skipTest("此测试与重构后的代码不兼容，请使用tests/agent/test_agentic_rag_refactored.py中的同名测试")
    
    def test_memory_manager_integration(self):
        """测试记忆管理器的集成"""
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.document_processor,
            environment=self.environment,
            memory_manager=self.memory_manager,
            prompt_manager=self.prompt_manager
        )
        
        # 运行Agent
        agentic_rag.run(self.test_query)
        
        # 验证记忆管理器记录了交互
        interactions = self.memory_manager.get_history()
        self.assertGreaterEqual(len(interactions), 1)
    
    def test_environment_state_update(self):
        """测试环境状态更新"""
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.document_processor,
            environment=self.environment,
            memory_manager=self.memory_manager,
            prompt_manager=self.prompt_manager
        )
        
        # 运行Agent
        agentic_rag.run(self.test_query)
        
        # 验证环境状态更新
        state = self.environment.get_state()
        self.assertIn("query", state)
        self.assertEqual(state["query"], self.test_query)

if __name__ == "__main__":
    unittest.main()