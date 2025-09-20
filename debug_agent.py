import traceback
import sys
import os
import traceback
from src.agent.agent_builder import create_agentic_rag
from src.config import get_config

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试不同的模型类型
def test_with_model(model_type):
    print(f"\n===== 测试模型类型: {model_type} =====")
    try:
        # 为tongyi模型准备API密钥
        api_key = None
        if model_type == "tongyi":
            api_key = get_config("model.tongyi.api_key", "")
            if api_key:
                print("使用配置中的通义千问API密钥")
            else:
                print("警告: 未配置通义千问API密钥，使用模拟响应")
                # 如果没有API密钥，我们可以模拟这个行为
                return
        
        # 创建AgenticRAG实例
        agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key)
        print("AgenticRAG实例创建成功")
        
        # 测试查询
        test_queries = [
            "什么是人工智能？",
            "agentic rag作者是谁？"
        ]
        
        for query in test_queries:
            print(f"\n测试查询: {query}")
            try:
                # 调用run方法并捕获详细的错误信息
                result = agentic_rag.run(query)
                
                # 打印结果
                print("=== 结果 ===")
                print(f"回答: {result.get('answer', '')}")
                print(f"状态: {result.get('status', '')}")
                print(f"错误: {result.get('error', '')}")
                print(f"迭代次数: {result.get('iterations', 0)}")
                print(f"工具使用: {result.get('tool_usage', [])}")
                print(f"上下文: {result.get('context', '')}")
            except Exception as e:
                print(f"\n发生未捕获的异常:")
                print(f"异常类型: {type(e).__name__}")
                print(f"异常信息: {str(e)}")
                print("\n完整错误堆栈:")
                traceback.print_exc(file=sys.stdout)
        
        # 测试validate_answer方法的阈值
        print("\n=== 测试validate_answer方法阈值 ===")
        try:
            evaluator = agentic_rag.answer_evaluator
            
            # 测试不同长度的答案
            for length in [5, 10, 11, 20]:
                test_answer = "" * length  # 创建指定长度的答案
                test_answer = test_answer if length <= 10 else "这是一个" + "长" * (length - 5)  # 为长答案添加内容
                is_valid = evaluator.validate_answer(test_answer, "测试查询")
                print(f"答案长度 {length}, 验证结果: {is_valid}")
        except Exception as e:
            print(f"\n测试validate_answer方法时发生异常:")
            print(f"异常类型: {type(e).__name__}")
            print(f"异常信息: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            
        # 直接测试AgenticRAG的内部方法
        print("\n=== 测试AgenticRAG内部方法 ===")
        try:
            # 测试_agent_loop方法的直接调用
            state_manager = agentic_rag.state_manager
            initial_state = state_manager.initialize_state("测试查询")
            initial_state["answer"] = "这是一个有效的测试答案，长度超过10个字符。"
            
            # 直接验证这个状态是否能通过_should_continue_iteration检查
            should_continue = agentic_rag._should_continue_iteration(initial_state)
            print(f"对于有效答案，是否应该继续迭代: {should_continue}")
            
            # 测试无效答案
            initial_state["answer"] = "短"
            should_continue = agentic_rag._should_continue_iteration(initial_state)
            print(f"对于无效答案，是否应该继续迭代: {should_continue}")
        except Exception as e:
            print(f"\n测试内部方法时发生异常:")
            print(f"异常类型: {type(e).__name__}")
            print(f"异常信息: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    except Exception as e:
        print(f"\n创建AgenticRAG实例时发生异常:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常信息: {str(e)}")
        traceback.print_exc(file=sys.stdout)

# 运行测试
test_with_model("mock")  # 首先测试mock模型
test_with_model("tongyi")  # 然后测试tongyi模型

print("\n===== 测试完成 =====")