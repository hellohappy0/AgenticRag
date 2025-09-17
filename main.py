from typing import Dict, Any, List
from typing import Dict, List, Any, Optional
from src.agent import AgenticRAG
from src.config import get_config, config_manager
from src.agent.agent_builder import create_agentic_rag

def main():
    """
    主函数，演示Agentic RAG的使用
    """
    print("初始化Agentic RAG系统...")
    print("系统支持从配置文件(config.yaml)和环境变量加载配置，默认优先使用环境变量，其次是配置文件。")
    print("环境变量格式: AGENTIC_RAG_{配置键}_...，使用下划线分隔嵌套键。例如: AGENTIC_RAG_MODEL_TONGYI_API_KEY")
    
    # 获取模型类型，默认优先使用环境变量，其次是配置文件
    model_type = get_config("model.type", "mock")
    print(f"使用配置中的模型类型: '{model_type}'")
    
    api_key = None
    custom_model = None
    
    if model_type != "mock" and model_type != "auto":
        if model_type == "tongyi":
            # 直接使用配置中的通义千问API密钥，默认优先使用环境变量，其次是配置文件
            config_api_key = get_config("model.tongyi.api_key", "")
            if config_api_key:
                print("使用配置中的通义千问API密钥")
                api_key = config_api_key
            else:
                print("警告: 未配置通义千问API密钥，请在config.yaml中设置或通过环境变量提供")
        elif model_type == "ollama":
            # 直接使用配置中的Ollama模型名称，默认优先使用环境变量，其次是配置文件
            config_model_name = get_config("model.ollama.model_name", "llama3")
            print(f"使用配置中的Ollama模型名称: '{config_model_name}'")
            custom_model = config_model_name
        else:
            # 其他模型类型的API密钥处理，默认优先使用环境变量，其次是配置文件
            config_api_key = get_config(f"model.{model_type}.api_key", "")
            if config_api_key:
                print(f"使用配置中的{model_type} API密钥")
                api_key = config_api_key
    
    # 创建Agentic RAG实例
    agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key, custom_model=custom_model)
    
    print("\nAgentic RAG系统已初始化完成，您可以开始提问。输入'quit'退出。\n")
    
    while True:
        try:
            # 获取用户输入
            query = input("您的问题: ")
            
            if query.lower() in ["quit", "exit", "退出"]:
                print("感谢使用Agentic RAG系统，再见！")
                break
            
            # 运行Agentic RAG处理查询
            print("\n正在处理您的问题...\n")
            result = agentic_rag.run(query)
            
            # 显示结果
            print("=== 回答 ===")
            print(result.get("answer", "没有生成回答"))
            print("\n=== 状态 ===")
            print(f"状态: {result.get('status', 'unknown')}")
            print(f"工具调用次数: {len(result.get('tool_calls', []))}")
            print(f"重试次数: {result.get('retries', 0)}")
            
            # 如果有自我批评，也显示
            if "critique" in result:
                print("\n=== 自我批评 ===")
                print(result["critique"])
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()