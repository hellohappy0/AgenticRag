import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.agent_builder import create_agentic_rag
from src.config import get_config

# 自动测试函数
def auto_test():
    print("===== Agentic RAG 自动测试 =====")
    
    # 测试查询列表
    test_queries = [
        "agentic rag作者是谁？",
        "什么是人工智能？"
    ]
    
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
        
        # 测试每个查询
        for idx, query in enumerate(test_queries):
            print(f"\n===== 测试查询 {idx+1}/{len(test_queries)} =====")
            print(f"查询: {query}")
            
            # 执行查询
            print("正在处理查询...")
            result = agentic_rag.run(query)
            
            # 打印结果
            print("\n结果:")
            print(f"答案: {result.get('answer', '')}")
            print(f"状态: {result.get('status', 'unknown')}")
            print(f"迭代次数: {result.get('iterations', 0)}")
            print(f"工具调用次数: {len(result.get('tool_usage', []))}")
            print(f"上下文长度: {len(result.get('context', ''))}字符")
    
    except Exception as e:
        print(f"\n测试过程中发生异常:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常信息: {str(e)}")
        import traceback
        traceback.print_exc(file=sys.stdout)
    
    print("\n===== 自动测试完成 =====")

if __name__ == "__main__":
    auto_test()