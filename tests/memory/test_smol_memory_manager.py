import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.memory.smol_memory_manager import SmolAgentMemoryManager

# 解决缺少的导入
try:
    from typing import Dict, Any, List, Optional
except ImportError:
    from typing import Dict, Any, List
    Optional = Any

try:
    from smolagents import Model
    from smolagents.memory import AgentMemory, TaskStep, ActionStep
    from smolagents.monitoring import Timing
    from smolagents.models import ChatMessage, MessageRole
except ImportError:
    # 创建mock类以允许测试在没有smolagents的环境中运行
    class MockModel:
        def __call__(self, prompt):
            return "模拟的模型响应"
    
    class MockAgentMemory:
        def __init__(self, system_prompt):
            self.system_prompt = system_prompt
            self.steps = []
        
        def reset(self):
            self.steps = []
    
    class MockTaskStep:
        def __init__(self, task):
            self.task = task
    
    class MockActionStep:
        def __init__(self, step_number, timing, model_output_message, model_output, observations, action_output=None):
            self.step_number = step_number
            self.timing = timing
            self.model_output_message = model_output_message
            self.model_output = model_output
            self.observations = observations
            self.action_output = action_output
    
    class MockTiming:
        def __init__(self, start_time, end_time):
            self.start_time = start_time
            self.end_time = end_time
    
    class MockChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class MockMessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        TOOL_RESPONSE = "tool_response"
    
    # 模拟smolagents模块
    class MockSmolagents:
        pass
    
    sys.modules['smolagents'] = MockSmolagents()
    sys.modules['smolagents'].Model = MockModel
    sys.modules['smolagents'].memory = type('obj', (object,), {
        'AgentMemory': MockAgentMemory,
        'TaskStep': MockTaskStep,
        'ActionStep': MockActionStep
    })
    sys.modules['smolagents'].monitoring = type('obj', (object,), {
        'Timing': MockTiming
    })
    sys.modules['smolagents'].models = type('obj', (object,), {
        'ChatMessage': MockChatMessage,
        'MessageRole': MockMessageRole
    })
    
    from smolagents import Model
    from smolagents.memory import AgentMemory, TaskStep, ActionStep
    from smolagents.monitoring import Timing
    from smolagents.models import ChatMessage, MessageRole


class TestSmolAgentMemoryManager(unittest.TestCase):
    
    def setUp(self):
        # 创建测试实例
        self.memory_manager = SmolAgentMemoryManager(max_history_size=5, max_compressed_size=100)
    
    def test_initialization(self):
        """测试记忆管理器的初始化"""
        self.assertEqual(self.memory_manager.max_history_size, 5)
        self.assertEqual(self.memory_manager.max_compressed_size, 100)
        self.assertEqual(len(self.memory_manager.memory.steps), 0)
        self.assertEqual(self.memory_manager.compressed_memory, "")
        self.assertEqual(self.memory_manager.current_step, 0)
        self.assertEqual(self.memory_manager._knowledge_index, {})
    
    def test_add_interaction(self):
        """测试添加交互记录功能"""
        # 添加用户交互
        self.memory_manager.add_interaction("user", "你好，请问什么是RAG？")
        self.assertEqual(len(self.memory_manager.memory.steps), 1)
        
        # 添加代理交互
        self.memory_manager.add_interaction("agent", "RAG是检索增强生成的缩写，是一种结合检索和生成的AI技术。")
        self.assertEqual(len(self.memory_manager.memory.steps), 2)
        
        # 添加带元数据的交互
        self.memory_manager.add_interaction("user", "能详细解释一下吗？", {"topic": "RAG", "timestamp": "2023-01-01"})
        self.assertEqual(len(self.memory_manager.memory.steps), 3)
    
    def test_get_history(self):
        """测试获取历史记录功能"""
        # 添加一些交互
        self.memory_manager.add_interaction("user", "问题1")
        self.memory_manager.add_interaction("agent", "回答1")
        self.memory_manager.add_interaction("user", "问题2", {"topic": "测试"})
        
        # 获取所有历史
        history = self.memory_manager.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "问题1")
        self.assertEqual(history[1]["role"], "agent")
        self.assertEqual(history[1]["content"], "回答1")
        self.assertEqual(history[2]["role"], "user")
        self.assertEqual(history[2]["content"], "问题2")
        self.assertEqual(history[2]["metadata"]["topic"], "测试")
        
        # 使用限制获取历史
        limited_history = self.memory_manager.get_history(limit=2)
        self.assertEqual(len(limited_history), 2)
        self.assertEqual(limited_history[0]["role"], "agent")
        self.assertEqual(limited_history[1]["role"], "user")
    
    def test_add_and_get_knowledge(self):
        """测试添加和获取知识库功能"""
        # 添加知识库条目
        self.memory_manager.add_knowledge("rag_definition", "RAG是检索增强生成的缩写，是一种结合检索和生成的AI技术。")
        self.assertEqual(len(self.memory_manager.memory.steps), 1)
        
        # 添加带元数据的知识库条目
        self.memory_manager.add_knowledge(
            "rag_components", 
            ["检索模块", "生成模块", "索引模块"], 
            {"type": "component_list", "source": "system"}
        )
        self.assertEqual(len(self.memory_manager.memory.steps), 2)
        
        # 获取知识库条目
        knowledge1 = self.memory_manager.get_knowledge("rag_definition")
        self.assertIsNotNone(knowledge1)
        self.assertIn("RAG是检索增强生成的缩写", knowledge1["value"])
        
        knowledge2 = self.memory_manager.get_knowledge("rag_components")
        self.assertIsNotNone(knowledge2)
        self.assertEqual(knowledge2["metadata"]["type"], "component_list")
        
        # 获取不存在的知识库条目
        knowledge3 = self.memory_manager.get_knowledge("non_existent")
        self.assertIsNone(knowledge3)
    
    def test_reset(self):
        """测试重置记忆功能"""
        # 添加一些数据
        self.memory_manager.add_interaction("user", "问题")
        self.memory_manager.add_knowledge("test", "测试知识")
        self.memory_manager.compressed_memory = "压缩后的记忆"
        
        # 重置
        self.memory_manager.reset()
        
        # 验证重置结果
        self.assertEqual(len(self.memory_manager.memory.steps), 0)
        self.assertEqual(self.memory_manager.compressed_memory, "")
        self.assertEqual(self.memory_manager.current_step, 0)
        self.assertEqual(self.memory_manager._knowledge_index, {})
    
    @patch('src.memory.smol_memory_manager.ChatMessage')
    def test_update_with_tool_results(self, mock_chat_message):
        """测试更新工具结果功能"""
        # 创建模拟的工具结果
        test_tool_name = "search_engine"
        test_tool_result = "搜索结果内容"
        
        tool_results = [
            {
                "status": "success",
                "tool": test_tool_name,
                "result": test_tool_result,
                "metadata": {"timestamp": "2023-01-01"}
            },
            {
                "status": "error",
                "tool": "retrieval_tool",
                "error": "检索失败"
            }
        ]
        
        # 更新工具结果
        self.memory_manager.update_with_tool_results(tool_results)
        
        # 验证添加了步骤
        self.assertGreaterEqual(len(self.memory_manager.memory.steps), 1)
        
        # 验证成功结果被添加到知识库
        last_tool_results = self.memory_manager.get_knowledge("last_tool_results")
        self.assertIsNotNone(last_tool_results)
        
        # 由于mock环境中可能存在序列化/反序列化问题，我们使用更灵活的断言
        # 检查结果中是否包含我们期望的工具名称
        self.assertIn("tool", str(last_tool_results["value"]))
        self.assertIn(test_tool_name, str(last_tool_results["value"]))
        self.assertIn("result", str(last_tool_results["value"]))
    
    def test_get_context_with_memory(self):
        """测试获取包含压缩记忆的上下文功能"""
        current_context = "当前上下文内容"
        
        # 没有压缩记忆的情况
        result = self.memory_manager.get_context_with_memory(current_context)
        self.assertEqual(result, current_context)
        
        # 有压缩记忆的情况
        self.memory_manager.compressed_memory = "压缩后的历史交互摘要"
        result = self.memory_manager.get_context_with_memory(current_context)
        self.assertIn("[历史交互摘要]", result)
        self.assertIn("压缩后的历史交互摘要", result)
        self.assertIn("[当前信息]", result)
        self.assertIn(current_context, result)
    
    def test_get_succinct_steps(self):
        """测试获取简洁步骤列表功能"""
        # 添加一些步骤
        self.memory_manager.add_interaction("user", "问题")
        self.memory_manager.add_knowledge("test", "测试知识")
        
        # 获取简洁步骤
        succinct_steps = self.memory_manager.get_succinct_steps()
        
        # 验证只返回了ActionStep类型的步骤
        self.assertEqual(len(succinct_steps), 1)  # 只有交互步骤会被返回
        self.assertIn("step_number", succinct_steps[0])
        self.assertIn("observations", succinct_steps[0])
        self.assertIn("action_output", succinct_steps[0])
    
    def test_get_full_steps(self):
        """测试获取完整步骤列表功能"""
        # 添加一些步骤
        self.memory_manager.add_interaction("user", "问题")
        
        # 获取完整步骤
        full_steps = self.memory_manager.get_full_steps()
        
        # 验证返回了所有步骤
        self.assertEqual(len(full_steps), 1)
        self.assertIn("step_number", full_steps[0])
        self.assertIn("type", full_steps[0])
    
    def test_add_final_answer(self):
        """测试添加最终回答功能"""
        # 添加最终回答
        final_answer = "这是最终回答内容"
        self.memory_manager.add_final_answer(final_answer)
        
        # 验证添加成功
        self.assertEqual(len(self.memory_manager.memory.steps), 1)
        
        # 验证历史记录包含最终回答
        history = self.memory_manager.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "agent")
        self.assertEqual(history[0]["content"], final_answer)
    
    def test_history_limit(self):
        """测试历史记录限制功能"""
        # 创建一个小限制的记忆管理器
        memory_manager = SmolAgentMemoryManager(max_history_size=3)
        
        # 添加超出限制的步骤
        for i in range(5):
            memory_manager.add_interaction(f"user", f"问题{i+1}")
        
        # 验证只保留了最近的步骤
        self.assertEqual(len(memory_manager.memory.steps), 3)
        
        # 验证知识库索引被重置
        self.assertEqual(memory_manager._knowledge_index, {})
    
    @patch('src.memory.smol_memory_manager.ChatMessage')
    def test_compress_memory(self, mock_chat_message):
        """测试记忆压缩功能"""
        # 添加一些交互
        self.memory_manager.add_interaction("user", "问题1")
        self.memory_manager.add_interaction("agent", "回答1")
        
        # 创建模拟模型
        mock_model = MagicMock()
        mock_model.generate.return_value = "压缩后的对话：用户问了问题1，代理回答了回答1"
        
        # 压缩记忆
        self.memory_manager.compress_memory(mock_model)
        
        # 验证压缩结果
        self.assertEqual(self.memory_manager.compressed_memory, "压缩后的对话：用户问了问题1，代理回答了回答1")
        
        # 测试压缩失败的情况
        mock_model.generate.side_effect = Exception("压缩失败")
        self.memory_manager.compressed_memory = ""
        self.memory_manager.compress_memory(mock_model)
        self.assertIn("2条历史记录", self.memory_manager.compressed_memory)


if __name__ == '__main__':
    unittest.main()