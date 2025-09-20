import sys
import os
import json
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.agent_builder import create_agentic_rag
from src.config import get_config

# 创建专门的详细调试函数
def detailed_debug():
    print("===== Agentic RAG 详细流程调试 =====")
    
    # 设置调试查询
    query = "agentic rag作者是谁？"
    print(f"测试查询: {query}")
    
    try:
        # 准备API密钥
        api_key = get_config("model.tongyi.api_key", "")
        if not api_key:
            print("警告: 未配置通义千问API密钥")
            return
        print("使用配置中的通义千问API密钥")
        
        # 创建AgenticRAG实例
        agentic_rag = create_agentic_rag(model_type="tongyi", api_key=api_key)
        print("AgenticRAG实例创建成功")
        
        # 保存原始方法以便替换回来
        original_agent_loop = agentic_rag._agent_loop
        original_process_model_response = agentic_rag._process_model_response
        original_generate_prompt = agentic_rag._generate_prompt
        original_get_model_response = agentic_rag._get_model_response
        
        # 创建状态管理器的引用
        state_manager = agentic_rag.state_manager
        
        # 重写_agent_loop方法来跟踪状态变化
        def debug_agent_loop(state):
            print(f"\n[循环开始] 迭代次数: {state['iterations']}, 状态: {state['status']}")
            print(f"初始上下文长度: {len(state['context'])}字符")
            
            # 执行原始_agent_loop方法
            final_state = original_agent_loop(state)
            
            print(f"[循环结束] 最终状态: {final_state['status']}, 答案长度: {len(final_state.get('answer', ''))}字符")
            if final_state.get('answer'):
                print(f"答案内容预览: {final_state['answer'][:100]}...")
            print(f"最终上下文长度: {len(final_state.get('context', ''))}字符")
            print(f"工具调用次数: {len(final_state.get('tool_usage_history', []))}")
            
            return final_state
        
        # 重写_process_model_response方法来跟踪响应处理
        def debug_process_model_response(state, response):
            print(f"\n[处理模型响应]")
            print(f"响应类型: {type(response).__name__}")
            print(f"'response'字段存在: {'response' in response}")
            if 'response' in response:
                print(f"'response'长度: {len(response['response'])}字符")
                print(f"'response'内容预览: {response['response'][:100]}...")
            
            print(f"'tool_calls'字段存在: {'tool_calls' in response}")
            if 'tool_calls' in response:
                print(f"'tool_calls'数量: {len(response['tool_calls'])}")
                if response['tool_calls']:
                    print(f"第一个tool_call: {response['tool_calls'][0]['name']}")
            
            # 执行原始_process_model_response方法
            updated_state = original_process_model_response(state, response)
            
            print(f"[状态更新后]")
            print(f"答案长度: {len(updated_state.get('answer', ''))}字符")
            print(f"状态: {updated_state['status']}")
            if updated_state.get('answer'):
                print(f"答案内容预览: {updated_state['answer'][:100]}...")
            
            return updated_state
        
        # 重写_generate_prompt方法来跟踪提示生成
        def debug_generate_prompt(state):
            prompt = original_generate_prompt(state)
            print(f"\n[生成提示]")
            print(f"提示长度: {len(prompt)}字符")
            print(f"提示内容预览: {prompt[:100]}...")
            return prompt
        
        # 重写_get_model_response方法来跟踪模型响应
        def debug_get_model_response(prompt, state):
            print("[调用模型生成响应]...")
            response = original_get_model_response(prompt, state)
            return response
        
        # 应用重写的方法
        agentic_rag._agent_loop = debug_agent_loop
        agentic_rag._process_model_response = debug_process_model_response
        agentic_rag._generate_prompt = debug_generate_prompt
        agentic_rag._get_model_response = debug_get_model_response
        
        # 执行run方法
        print("\n===== 开始执行run方法 =====")
        result = agentic_rag.run(query)
        
        # 打印最终结果
        print("\n===== 最终结果 =====")
        print(f"答案: {result.get('answer', '')}")
        print(f"状态: {result.get('status', 'unknown')}")
        print(f"迭代次数: {result.get('iterations', 0)}")
        print(f"工具调用次数: {len(result.get('tool_usage', []))}")
        
        # 保存完整结果到文件以便详细分析
        with open("debug_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n完整结果已保存到debug_result.json文件")
        
    except Exception as e:
        print(f"\n调试过程中发生异常:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常信息: {str(e)}")
        traceback.print_exc(file=sys.stdout)
    
    print("\n===== 调试完成 =====")

if __name__ == "__main__":
    detailed_debug()