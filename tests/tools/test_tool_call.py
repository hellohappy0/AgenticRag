import unittest
from unittest.mock import Mock, MagicMock
from src.agent.agent_refactored import AgenticRAG

class SimpleTestToolCall(unittest.TestCase):
    def test_tool_call_direct(self):
        # 创建模拟工具
        mock_tool = Mock()
        mock_tool.name = "retrieve_documents"
        mock_tool.run = Mock(return_value={"results": [{"content": "测试内容"}]})
        mock_tool.get_tool_info = Mock(return_value={"name": "retrieve_documents", "description": "检索文档工具"})
        
        # 创建只包含模拟工具的工具列表
        tools_with_mock = [mock_tool]
        
        # 创建其他模拟组件
        mock_model = Mock()
        mock_document_processor = Mock()
        mock_environment = Mock()
        mock_memory_manager = Mock()
        mock_prompt_manager = Mock()
        
        # 模拟model.generate_with_tools方法返回使用retrieve_documents工具的调用
        mock_model.generate_with_tools.return_value = {
            "response": "<|FunctionCallBegin|>[{\"name\": \"retrieve_documents\", \"parameters\": {\"query\": \"测试查询\"}}]<|FunctionCallEnd|>",
            "tool_calls": [{"name": "retrieve_documents", "parameters": {"query": "测试查询"}}],
            "raw_output": ""
        }
        
        # 为QueryAnalyzer添加模拟实现
        mock_query_analyzer = Mock()
        mock_query_analyzer.analyze_query.return_value = {"type": "information_retrieval", "intent": "查找信息"}
        
        # 为_generate_prompt添加模拟实现
        mock_generate_prompt = Mock(return_value="你现在需要回答用户的问题...")
        
        # 为_should_continue_iteration添加模拟实现
        mock_should_continue = Mock(return_value=False)
        
        # 为AnswerEvaluator添加模拟实现
        mock_evaluate_answer = Mock(return_value={"quality": "good", "relevance": "high", "completeness": "complete"})
        mock_optimize_answer = Mock(return_value="优化后的答案")
        mock_validate_answer = Mock(return_value=True)
        
        # 创建Agentic RAG实例
        agentic_rag = AgenticRAG(
            model=mock_model,
            tools=tools_with_mock,
            document_processor=mock_document_processor,
            environment=mock_environment,
            memory_manager=mock_memory_manager,
            prompt_manager=mock_prompt_manager
        )
        
        # 替换AgenticRAG的一些方法，以便更好地控制执行流程
        agentic_rag.query_analyzer.analyze_query = mock_query_analyzer.analyze_query
        agentic_rag._generate_prompt = mock_generate_prompt
        agentic_rag._should_continue_iteration = mock_should_continue
        agentic_rag.answer_evaluator.evaluate_answer = mock_evaluate_answer
        agentic_rag.answer_evaluator.optimize_answer = mock_optimize_answer
        agentic_rag.answer_evaluator.validate_answer = mock_validate_answer
        
        # 运行Agent
        result = agentic_rag.run("测试查询")
        
        # 验证工具执行
        print(f"Tool run called: {mock_tool.run.called}")
        print(f"Tool run call args: {mock_tool.run.call_args}")
        mock_tool.run.assert_called_once_with(query="测试查询")

if __name__ == "__main__":
    unittest.main()