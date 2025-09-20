import sys
import os
import json
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.agent_builder import create_agentic_rag
from src.config import get_config

# 创建专门的调试函数
def debug_model_response():
    print("===== 调试模型响应格式 =====")
    
    # 测试两种模型类型
    model_types = ["mock", "tongyi"]
    
    for model_type in model_types:
        print(f"\n测试模型类型: {model_type}")
        
        try:
            # 为tongyi模型准备API密钥
            api_key = None
            if model_type == "tongyi":
                api_key = get_config("model.tongyi.api_key", "")
                if not api_key:
                    print("警告: 未配置通义千问API密钥，跳过tongyi模型测试")
                    continue
                print("使用配置中的通义千问API密钥")
            
            # 创建AgenticRAG实例
            agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key)
            print(f"{model_type}模型实例创建成功")
            
            # 直接访问模型对象
            model = agentic_rag.model
            print(f"模型类型: {type(model).__name__}")
            
            # 准备一个简单的提示和工具列表
            test_prompt = "你好，这是一个测试问题。"
            test_tools = [{"name": "test_tool", "description": "测试工具", "parameters": {}}]
            
            # 调用generate_with_tools方法
            print("调用generate_with_tools方法...")
            response = model.generate_with_tools(test_prompt, tools=test_tools)
            
            # 打印完整的响应结构
            print(f"模型响应类型: {type(response).__name__}")
            print(f"模型响应内容: {json.dumps(response, ensure_ascii=False, indent=2)}")
            
            # 检查关键字段
            print(f"\n响应中的关键字段:")
            print(f"- 'response'字段存在: {'response' in response}")
            if 'response' in response:
                print(f"  - 'response'值: '{response['response']}'")
                print(f"  - 'response'长度: {len(response['response'])}字符")
            
            print(f"- 'tool_calls'字段存在: {'tool_calls' in response}")
            if 'tool_calls' in response:
                print(f"  - 'tool_calls'类型: {type(response['tool_calls']).__name__}")
                print(f"  - 'tool_calls'数量: {len(response['tool_calls'])}")
                if response['tool_calls']:
                    print(f"  - 第一个tool_call: {json.dumps(response['tool_calls'][0], ensure_ascii=False, indent=2)}")
            
            # 测试实际查询的响应
            print("\n测试实际查询的响应...")
            real_prompt = "agentic rag作者是谁？"
            real_response = model.generate_with_tools(real_prompt, tools=test_tools)
            print(f"实际查询响应: {json.dumps(real_response, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"\n测试{model_type}模型时发生异常:")
            print(f"异常类型: {type(e).__name__}")
            print(f"异常信息: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    
    print("\n===== 调试完成 =====")

if __name__ == "__main__":
    debug_model_response()