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
    print("系统支持从配置文件(config.yaml)和环境变量加载配置，环境变量优先级高于配置文件。")
    print("环境变量格式: AGENTIC_RAG_{配置键}_...，使用下划线分隔嵌套键。例如: AGENTIC_RAG_MODEL_TONGYI_API_KEY")
    
    # 获取用户选择的模型类型
    # 如果配置中已经设置了模型类型，询问用户是否使用配置中的值
    config_model_type = get_config("model.type", "")
    if config_model_type:
        use_config = input(f"检测到配置文件中的模型类型为 '{config_model_type}'，是否使用？(y/n，默认y): ").strip().lower() or "y"
        if use_config == "y":
            model_type = config_model_type
        else:
            model_type = input("请选择模型类型 (mock/smolagent/tongyi/ollama/auto，默认mock): ").strip().lower() or "mock"
    else:
        model_type = input("请选择模型类型 (mock/smolagent/tongyi/ollama/auto，默认mock): ").strip().lower() or "mock"
    
    api_key = None
    custom_model = None
    
    if model_type != "mock" and model_type != "auto":
        if model_type == "tongyi":
            # 检查配置中是否已有API密钥
            config_api_key = get_config("model.tongyi.api_key", "")
            if config_api_key:
                use_config_key = input(f"检测到配置中的通义千问API密钥，是否使用？(y/n，默认y): ").strip().lower() or "y"
                if use_config_key != "y":
                    api_key = input("请输入阿里云通义千问的API密钥: ").strip()
            else:
                api_key = input("请输入阿里云通义千问的API密钥: ").strip()
        elif model_type == "ollama":
            # 检查配置中是否已有模型名称
            config_model_name = get_config("model.ollama.model_name", "")
            if config_model_name:
                use_config_model = input(f"检测到配置中的Ollama模型名称为 '{config_model_name}'，是否使用？(y/n，默认y): ").strip().lower() or "y"
                if use_config_model != "y":
                    custom_model_input = input("请输入Ollama模型名称 (默认llama3，可选): ").strip()
                    if custom_model_input:
                        custom_model = custom_model_input
            else:
                custom_model_input = input("请输入Ollama模型名称 (默认llama3，可选): ").strip()
                if custom_model_input:
                    custom_model = custom_model_input
        else:
            # 其他模型类型的API密钥处理
            config_api_key = get_config(f"model.{model_type}.api_key", "")
            if not config_api_key:
                api_key_input = input(f"请输入{model_type}的API密钥 (可选): ").strip()
                if api_key_input:
                    api_key = api_key_input
    
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