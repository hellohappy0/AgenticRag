import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.agent_builder import create_agentic_rag

# 创建AgenticRAG实例（使用mock模型）
agentic_rag = create_agentic_rag(model_type="mock")
evaluator = agentic_rag.answer_evaluator

print("===== 测试validate_answer方法的真实阈值 =====")

# 测试不同长度的答案
for length in range(5, 25):
    # 创建指定长度的非空答案字符串
    test_answer = "a" * length  # 使用'a'字符创建非空字符串
    is_valid = evaluator.validate_answer(test_answer, "测试查询")
    print(f"答案长度 {length}, 内容: '{test_answer}', 验证结果: {is_valid}")

# 测试空答案
print(f"空答案, 验证结果: {evaluator.validate_answer('', '测试查询')}")
print(f"空格答案, 验证结果: {evaluator.validate_answer('   ', '测试查询')}")

print("\n===== 测试完成 =====")