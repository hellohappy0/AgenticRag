import unittest
from unittest.mock import Mock, patch, MagicMock
from src.agent.agent_refactored import AgenticRAG, AgentStateManager, QueryAnalyzer, ToolExecutor, AnswerEvaluator
from src.tools.retrieval_tool import RetrievalTool, SearchTool
from src.doc_process.simple_processor import SimpleDocumentProcessor
from src.agent_context.agent_env import SimpleAgentEnvironment
from src.memory.smol_memory_manager import SmolAgentMemoryManager
from src.prompt.prompt_manager import AgentPromptTemplates
from src.model.language_model import BaseLanguageModel


import unittest
from unittest.mock import Mock, patch, MagicMock
from src.agent.agent_refactored import AgenticRAG, AgentStateManager, QueryAnalyzer, ToolExecutor, AnswerEvaluator
from src.tools.retrieval_tool import RetrievalTool, SearchTool
from src.doc_process.simple_processor import SimpleDocumentProcessor
from src.agent_context.agent_env import SimpleAgentEnvironment
from src.memory.smol_memory_manager import SmolAgentMemoryManager
from src.prompt.prompt_manager import AgentPromptTemplates
from src.model.language_model import BaseLanguageModel


class TestAgentComponents(unittest.TestCase):
    """
    测试重构后的Agent组件
    """
    
    def setUp(self):
        # 创建模拟组件
        self.mock_model = MagicMock(spec=BaseLanguageModel)
        self.mock_prompt_manager = MagicMock()
        self.mock_memory_manager = MagicMock(spec=SmolAgentMemoryManager)
        self.mock_environment = MagicMock(spec=SimpleAgentEnvironment)
        self.mock_document_processor = MagicMock(spec=SimpleDocumentProcessor)
        
        # 设置默认返回值
        self.mock_model.generate.return_value = "这是一个测试回答。"
        self.mock_model.generate_with_tools.return_value = {
            "response": "这是一个测试回答。",
            "tool_calls": [],
            "raw_output": ""
        }
        self.mock_prompt_manager.generate_prompt.return_value = "测试提示"
        
        # 创建模拟工具
        self.mock_tool = Mock()
        self.mock_tool.name = "test_tool"
        self.mock_tool.run = Mock(return_value={"results": ["test_result"]})
        self.tools = [self.mock_tool]


class TestAgentStateManager(TestAgentComponents):
    """
    测试AgentStateManager组件
    """
    
    def test_initialize_state(self):
        """测试状态初始化功能"""
        state_manager = AgentStateManager()
        query = "测试查询"
        state = state_manager.initialize_state(query)
        
        # 验证状态包含所有必需字段
        self.assertEqual(state["query"], query)
        self.assertEqual(state["original_query"], query)
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")
        self.assertEqual(state["iterations"], 0)
        self.assertEqual(state["max_iterations"], 3)
        self.assertEqual(state["status"], "initialized")
    
    def test_update_state(self):
        """测试状态更新功能"""
        state_manager = AgentStateManager()
        initial_state = state_manager.initialize_state("测试查询")
        
        # 更新状态
        updated_state = state_manager.update_state(
            initial_state,
            answer="测试答案",
            status="success",
            iterations=1
        )
        
        # 验证状态已更新
        self.assertEqual(updated_state["answer"], "测试答案")
        self.assertEqual(updated_state["status"], "success")
        self.assertEqual(updated_state["iterations"], 1)
        # 验证原始状态未被修改
        self.assertEqual(initial_state["answer"], "")
    
    def test_validate_state(self):
        """测试状态验证功能"""
        state_manager = AgentStateManager()
        valid_state = state_manager.initialize_state("测试查询")
        invalid_state = {"answer": "测试答案"}  # 缺少必需字段
        
        # 验证状态有效性
        self.assertTrue(state_manager.validate_state(valid_state))
        self.assertFalse(state_manager.validate_state(invalid_state))


class TestQueryAnalyzer(TestAgentComponents):
    """
    测试QueryAnalyzer组件
    """
    
    def test_analyze_query(self):
        """测试查询分析功能"""
        analyzer = QueryAnalyzer(self.mock_model, self.mock_prompt_manager)
        query = "什么是大型语言模型？"
        analysis = analyzer.analyze_query(query)
        
        # 验证分析结果包含预期字段
        self.assertIn("type", analysis)
        self.assertIn("complexity", analysis)
        self.assertIn("domain", analysis)
        self.assertIn("requires_search", analysis)
        self.assertIn("requires_retrieval", analysis)
    
    def test_rewrite_query(self):
        """测试查询重写功能"""
        analyzer = QueryAnalyzer(self.mock_model, self.mock_prompt_manager)
        query = "什么是大型语言模型？"
        analysis = {"type": "general"}
        
        # 由于是简化实现，重写后的查询应该与原查询相同
        rewritten_query = analyzer.rewrite_query(query, analysis)
        self.assertEqual(rewritten_query, query)
    
    def test_decompose_query(self):
        """测试查询拆解功能"""
        analyzer = QueryAnalyzer(self.mock_model, self.mock_prompt_manager)
        query = "什么是大型语言模型？"
        analysis = {"type": "general"}
        
        # 对于简单查询，拆解后的子问题列表应该只包含原查询
        sub_queries = analyzer.decompose_query(query, analysis)
        self.assertEqual(len(sub_queries), 1)
        self.assertEqual(sub_queries[0], query)


class TestToolExecutor(TestAgentComponents):
    """
    测试ToolExecutor组件
    """
    
    def setUp(self):
        super().setUp()
        # 创建模拟工具
        self.mock_retrieval_tool = Mock()
        self.mock_retrieval_tool.name = "retrieve_documents"
        self.mock_retrieval_tool.run.return_value = [{"content": "测试文档内容"}]
        self.mock_retrieval_tool.get_tool_info.return_value = {
            "name": "retrieve_documents",
            "description": "检索文档工具"
        }
        
        self.mock_search_tool = Mock()
        self.mock_search_tool.name = "search"
        self.mock_search_tool.run.return_value = [{"title": "测试标题", "content": "测试内容"}]
        self.mock_search_tool.get_tool_info.return_value = {
            "name": "search",
            "description": "搜索工具"
        }
        
        self.tools = [self.mock_retrieval_tool, self.mock_search_tool]
    
    def test_execute_tool_success(self):
        """测试成功执行工具"""
        executor = ToolExecutor(self.tools)
        result = executor.execute_tool("retrieve_documents", {"query": "测试查询"})
        
        # 验证工具执行成功
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], [{"content": "测试文档内容"}])
        self.assertIsNone(result["error"])
        # 验证工具的run方法被调用
        self.mock_retrieval_tool.run.assert_called_once_with(query="测试查询")
    
    def test_execute_tool_failure(self):
        """测试执行工具失败"""
        # 设置工具抛出异常
        self.mock_retrieval_tool.run.side_effect = Exception("工具执行错误")
        
        executor = ToolExecutor(self.tools)
        result = executor.execute_tool("retrieve_documents", {"query": "测试查询"})
        
        # 验证工具执行失败
        self.assertFalse(result["success"])
        self.assertIsNone(result["result"])
        self.assertEqual(result["error"], "工具执行错误")
    
    def test_execute_unknown_tool(self):
        """测试执行未知工具"""
        executor = ToolExecutor(self.tools)
        
        # 验证抛出ValueError异常
        with self.assertRaises(ValueError):
            executor.execute_tool("unknown_tool", {"query": "测试查询"})
    
    def test_get_tool_info(self):
        """测试获取工具信息"""
        executor = ToolExecutor(self.tools)
        
        # 获取单个工具信息
        retrieval_info = executor.get_tool_info("retrieve_documents")
        self.assertEqual(retrieval_info["name"], "retrieve_documents")
        
        # 获取所有工具信息
        all_tools_info = executor.get_tool_info()
        self.assertEqual(len(all_tools_info), 2)
        self.assertIn("retrieve_documents", all_tools_info)
        self.assertIn("search", all_tools_info)


class TestAnswerEvaluator(TestAgentComponents):
    """
    测试AnswerEvaluator组件
    """
    
    def test_validate_answer(self):
        """测试答案验证功能"""
        evaluator = AnswerEvaluator(self.mock_model, self.mock_prompt_manager)
        
        # 验证有效答案
        valid_answer = "这是一个有效的测试答案，长度足够长。"
        self.assertTrue(evaluator.validate_answer(valid_answer, "测试查询"))
        
        # 验证无效答案
        invalid_answer = "简短"
        self.assertFalse(evaluator.validate_answer(invalid_answer, "测试查询"))
        self.assertFalse(evaluator.validate_answer("", "测试查询"))
    
    def test_evaluate_answer(self):
        """测试答案评估功能"""
        evaluator = AnswerEvaluator(self.mock_model, self.mock_prompt_manager)
        answer = "这是一个测试答案。"
        evaluation = evaluator.evaluate_answer(answer, "测试查询")
        
        # 验证评估结果包含预期字段
        self.assertIn("quality", evaluation)
        self.assertIn("relevance", evaluation)
        self.assertIn("completeness", evaluation)
        self.assertIn("feedback", evaluation)
    
    def test_optimize_answer(self):
        """测试答案优化功能"""
        evaluator = AnswerEvaluator(self.mock_model, self.mock_prompt_manager)
        answer = "这是一个测试答案。"
        evaluation = {"quality": "good"}
        
        # 由于是简化实现，优化后的答案应该与原答案相同
        optimized_answer = evaluator.optimize_answer(answer, "测试查询", evaluation)
        self.assertEqual(optimized_answer, answer)


class TestAgenticRAGRefactored(TestAgentComponents):
    """
    测试重构后的AgenticRAG类
    """
    
    def setUp(self):
        super().setUp()
        # 创建模拟工具
        self.mock_retrieval_tool = Mock()
        self.mock_retrieval_tool.name = "retrieve_documents"
        self.mock_retrieval_tool.run.return_value = [{"content": "测试文档内容"}]
        self.mock_retrieval_tool.get_tool_info.return_value = {
            "name": "retrieve_documents",
            "description": "检索文档工具"
        }
        
        self.tools = [self.mock_retrieval_tool]
    
    def test_agent_initialization(self):
        """测试AgenticRAG的初始化"""
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.mock_document_processor,
            environment=self.mock_environment,
            memory_manager=self.mock_memory_manager,
            prompt_manager=self.mock_prompt_manager
        )
        
        # 验证初始化的组件是否正确
        self.assertEqual(agentic_rag.model, self.mock_model)
        self.assertEqual(agentic_rag.document_processor, self.mock_document_processor)
        self.assertEqual(agentic_rag.environment, self.mock_environment)
        self.assertEqual(agentic_rag.memory_manager, self.mock_memory_manager)
        self.assertEqual(agentic_rag.prompt_manager, self.mock_prompt_manager)
        
        # 验证子组件是否正确初始化
        self.assertIsInstance(agentic_rag.state_manager, AgentStateManager)
        self.assertIsInstance(agentic_rag.query_analyzer, QueryAnalyzer)
        self.assertIsInstance(agentic_rag.tool_executor, ToolExecutor)
        self.assertIsInstance(agentic_rag.answer_evaluator, AnswerEvaluator)
    
    @patch.object(AgenticRAG, '_agent_loop')
    def test_agent_run_basic_functionality(self, mock_agent_loop):
        """测试AgenticRAG的基本运行功能"""
        # 设置模拟_agent_loop方法返回的状态
        mock_state = {
            "query": "测试查询",
            "original_query": "测试查询",
            "context": "测试上下文",
            "answer": "测试答案",
            "tool_calls": [],
            "retries": 0,
            "iterations": 1,
            "max_iterations": 3,
            "status": "success",
            "query_analysis": {},
            "tool_usage_history": [],
            "errors": []
        }
        mock_agent_loop.return_value = mock_state
        
        # 创建AgenticRAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.mock_document_processor,
            environment=self.mock_environment,
            memory_manager=self.mock_memory_manager,
            prompt_manager=self.mock_prompt_manager
        )
        
        # 运行Agent
        result = agentic_rag.run("测试查询")
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("status", result)
        self.assertEqual(result["answer"], "测试答案")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["context"], "测试上下文")
        self.assertEqual(result["iterations"], 1)
    
    @patch.object(AgenticRAG, '_process_query_based_on_analysis')
    def test_agent_loop(self, mock_process_query):
        """测试代理的核心循环逻辑"""
        # 创建AgenticRAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.mock_document_processor,
            environment=self.mock_environment,
            memory_manager=self.mock_memory_manager,
            prompt_manager=self.mock_prompt_manager
        )
        
        # 设置模拟方法的返回值
        initial_state = agentic_rag.state_manager.initialize_state("测试查询")
        processed_state = agentic_rag.state_manager.update_state(
            initial_state,
            context="测试上下文",
            query_analysis={"type": "general"}
        )
        mock_process_query.return_value = processed_state
        
        # 运行代理循环
        final_state = agentic_rag._agent_loop(initial_state)
        
        # 验证代理循环执行了预期的步骤
        mock_process_query.assert_called_once()
        self.assertIn("status", final_state)
    
    def test_reset(self):
        """测试重置代理状态"""
        # 创建AgenticRAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=self.tools,
            document_processor=self.mock_document_processor,
            environment=self.mock_environment,
            memory_manager=self.mock_memory_manager,
            prompt_manager=self.mock_prompt_manager
        )
        
        # 调用reset方法
        agentic_rag.reset()
        
        # 验证记忆管理器和环境的reset方法被调用
        self.mock_memory_manager.reset.assert_called_once()
        self.mock_environment.reset.assert_called_once()


class TestAgenticRAG(TestAgentComponents):
    """
    测试AgenticRAG类的核心功能
    """
    
    def test_tool_selection_and_execution(self):
        """测试工具选择和执行流程"""
        # 创建模拟工具
        mock_tool = Mock()
        mock_tool.name = "retrieve_documents"
        mock_tool.run = Mock(return_value={"results": [{"content": "测试内容"}]})
        mock_tool.get_tool_info = Mock(return_value={"name": "retrieve_documents", "description": "检索文档工具"})
        
        # 创建只包含模拟工具的工具列表
        tools_with_mock = [mock_tool]
        
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=self.mock_model,
            tools=tools_with_mock,
            document_processor=self.mock_document_processor,
            environment=self.mock_environment,
            memory_manager=self.mock_memory_manager,
            prompt_manager=self.mock_prompt_manager
        )
        
        # 模拟model.generate_with_tools方法返回使用retrieve_documents工具的调用
        self.mock_model.generate_with_tools.return_value = {
            "response": "<|FunctionCallBegin|>[{\"name\": \"retrieve_documents\", \"parameters\": {\"query\": \"测试查询\"}}]<|FunctionCallEnd|>",
            "tool_calls": [{"name": "retrieve_documents", "parameters": {"query": "测试查询"}}],
            "raw_output": ""
        }
        
        # 模拟_query_analyzer.analyze_query方法
        mock_query_analyzer = Mock()
        mock_query_analyzer.analyze_query.return_value = {"type": "information_retrieval", "intent": "查找信息"}
        agentic_rag.query_analyzer.analyze_query = mock_query_analyzer.analyze_query
        
        # 模拟_generate_prompt方法
        mock_generate_prompt = Mock(return_value="你现在需要回答用户的问题...")
        agentic_rag._generate_prompt = mock_generate_prompt
        
        # 模拟_should_continue_iteration方法
        mock_should_continue = Mock(return_value=False)
        agentic_rag._should_continue_iteration = mock_should_continue
        
        # 模拟AnswerEvaluator的方法
        mock_evaluate_answer = Mock(return_value={"quality": "good", "relevance": "high", "completeness": "complete"})
        mock_optimize_answer = Mock(return_value="优化后的答案")
        mock_validate_answer = Mock(return_value=True)
        agentic_rag.answer_evaluator.evaluate_answer = mock_evaluate_answer
        agentic_rag.answer_evaluator.optimize_answer = mock_optimize_answer
        agentic_rag.answer_evaluator.validate_answer = mock_validate_answer
        
        # 运行Agent
        result = agentic_rag.run("测试查询")
        
        # 验证工具执行
        mock_tool.run.assert_called_once_with(query="测试查询")


if __name__ == '__main__':
    unittest.main()