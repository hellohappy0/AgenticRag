from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import json
import requests
from smolagents import ToolCallingAgent, TransformersModel

# 尝试导入dashscope库（用于通义千问模型）
try:
    import dashscope
except ImportError:
    print("警告: dashscope库未安装，将无法使用通义千问模型。请运行 pip install dashscope 来安装")


class BaseLanguageModel(ABC):
    """
    语言模型基类，定义了与语言模型交互的基本接口
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成模型响应
        
        @param prompt: 输入提示
        @param kwargs: 模型参数
        @return: 模型生成的响应
        """
        pass
    
    @abstractmethod
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        生成模型响应，支持工具调用
        
        @param prompt: 输入提示
        @param tools: 可用工具列表
        @param kwargs: 模型参数
        @return: 包含模型响应和可能的工具调用的字典
        """
        pass


class SmolAgentModel(BaseLanguageModel):
    """
    基于smolAgent的语言模型实现
    """
    
    def __init__(self, model_name: str = "gpt2", api_key: Optional[str] = None, **kwargs):
        """
        初始化smolAgent模型
        
        @param model_name: 模型名称
        @param api_key: API密钥
        @param kwargs: 其他参数
        """
        self.model_name = model_name
        self.api_key = api_key
        self.model_params = kwargs
        
        # 初始化模型和代理
        try:
            # 创建模型实例
            if self.api_key:
                self.model = TransformersModel(model_id=self.model_name, token=self.api_key)
            else:
                self.model = TransformersModel(model_id=self.model_name)
            
            # 初始时创建无工具的代理
            self.agent = ToolCallingAgent(model=self.model, tools=[])
        except Exception as e:
            raise RuntimeError(f"初始化smolAgent失败: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成模型响应
        
        @param prompt: 输入提示
        @param kwargs: 模型参数
        @return: 模型生成的响应
        """
        try:
            # 直接使用已初始化的代理生成响应
            response = self.agent.run(prompt, **kwargs)
            return str(response)
        except Exception as e:
            raise RuntimeError(f"模型生成失败: {str(e)}")
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        生成模型响应，支持工具调用
        
        @param prompt: 输入提示
        @param tools: 可用工具列表
        @param kwargs: 模型参数
        @return: 包含模型响应和可能的工具调用的字典
        """
        try:
            # 创建带有工具的agent
            self.agent = ToolCallingAgent(model=self.model, tools=tools)
            
            # 生成响应
            response = self.agent.run(prompt, **kwargs)
            
            # 解析响应，提取工具调用信息
            result = {
                "response": str(response),
                "tool_calls": [],  # 简化处理，实际应从response中提取
                "raw_output": response
            }
            
            return result
        except Exception as e:
            raise RuntimeError(f"带工具的模型生成失败: {str(e)}")


class OllamaModel(BaseLanguageModel):
    """
    基于本地Ollama部署的语言模型实现
    """
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", **kwargs):
        """
        初始化Ollama模型
        
        @param model_name: 模型名称，如llama3、mistral等
        @param base_url: Ollama API的基础URL，默认是本地部署的地址
        @param kwargs: 其他参数
        """
        self.model_name = model_name
        self.base_url = base_url
        self.model_params = kwargs
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成模型响应
        
        @param prompt: 输入提示
        @param kwargs: 其他参数
        @return: 生成的文本
        """
        # 合并默认参数和传入参数
        params = {**self.model_params, **kwargs}
        
        # 构建API请求
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            **params
        }
        
        # 发送请求到Ollama API
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line)
                    if "response" in line_data:
                        full_response += line_data["response"]
                    if line_data.get("done", False):
                        break
            
            return full_response
        except Exception as e:
            raise RuntimeError(f"Ollama模型调用失败: {str(e)}")
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        生成模型响应，支持工具调用
        
        @param prompt: 输入提示
        @param tools: 可用工具列表
        @param kwargs: 其他参数
        @return: 包含响应和工具调用信息的字典
        """
        # 构建包含工具信息的提示
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        tool_prompt = f"""你可以使用以下工具来回答问题：
{tools_json}

如果需要使用工具，请按照以下格式输出：
<|FunctionCallBegin|>[{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}]<|FunctionCallEnd|>

问题：{prompt}"""
        
        # 获取模型响应
        response = self.generate(tool_prompt, **kwargs)
        
        # 解析工具调用
        tool_calls = []
        try:
            # 查找工具调用标记
            if "<|FunctionCallBegin|>" in response and "<|FunctionCallEnd|>" in response:
                begin_idx = response.index("<|FunctionCallBegin|>") + len("<|FunctionCallBegin|>")
                end_idx = response.index("<|FunctionCallEnd|>")
                tool_call_str = response[begin_idx:end_idx]
                
                # 解析工具调用JSON
                if tool_call_str.strip():  # 确保不为空
                    tool_calls = json.loads(tool_call_str)
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
        except json.JSONDecodeError:
            # 如果解析失败，返回空的工具调用列表
            pass
        
        return {
            "response": response,
            "tool_calls": tool_calls,
            "raw_output": response
        }


class TongyiQianwenModel(BaseLanguageModel):
    """
    基于阿里云通义千问API的语言模型实现
    """
    
    def __init__(self, api_key: str, model_name: str = "qwen-max", **kwargs):
        """
        初始化通义千问模型
        
        @param api_key: DashScope API密钥
        @param model_name: 模型名称
        @param kwargs: 其他参数
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model_params = kwargs
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成模型响应
        
        @param prompt: 输入提示
        @param kwargs: 模型参数
        @return: 模型生成的响应
        """
        import dashscope
        dashscope.api_key = self.api_key
        
        try:
            response = dashscope.Generation.call(
                self.model_name,
                prompt=prompt,
                **self.model_params,
                **kwargs
            )
            
            if response.status_code == 200:
                # 尝试从不同的字段获取响应内容，适配不同的API版本
                if hasattr(response.output, 'text') and response.output.text:
                    return response.output.text
                elif hasattr(response.output, 'choices') and response.output.choices:
                    # 适配新的响应格式
                    for choice in response.output.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            return choice.message.content
                # 如果所有尝试都失败，返回空字符串
                return ""
            else:
                raise RuntimeError(f"通义千问模型调用失败: {response.message}")
        except Exception as e:
            raise RuntimeError(f"通义千问模型调用失败: {str(e)}")
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        生成模型响应，支持工具调用
        """
        # 简化实现，实际应根据通义千问的工具调用格式调整
        # 通义千问的工具调用功能可能需要参考最新API文档
        try:
            # 使用标准的generate方法生成响应
            response_text = self.generate(prompt, **kwargs)
            
            # 这里仅作为示例，实际需要根据模型的输出格式解析工具调用
            # 检查响应中是否包含工具调用格式
            tool_calls = []
            try:
                # 尝试检查是否包含工具调用标记
                if "<|FunctionCallBegin|>" in response_text and "<|FunctionCallEnd|>" in response_text:
                    start_idx = response_text.find("<|FunctionCallBegin|>") + len("<|FunctionCallBegin|>")
                    end_idx = response_text.find("<|FunctionCallEnd|>")
                    tool_call_str = response_text[start_idx:end_idx].strip()
                    
                    # 解析JSON格式的工具调用
                    tool_calls = json.loads(tool_call_str)
                    
                    # 确保返回的是列表格式
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
            except Exception as e:
                print(f"解析工具调用失败: {str(e)}")
            
            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "raw_output": response_text
            }
        except Exception as e:
            raise RuntimeError(f"通义千问带工具的模型调用失败: {str(e)}")


class ModelFactory:
    """
    模型工厂类，用于创建不同类型的语言模型实例
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseLanguageModel:
        """
        创建语言模型实例
        
        @param model_type: 模型类型
        @param kwargs: 模型参数
        @return: 语言模型实例
        """
        if model_type.lower() == "smolagent":
            return SmolAgentModel(**kwargs)
        elif model_type.lower() == "anthropic":
            # 可以添加Anthropic Claude模型的实现
            raise NotImplementedError("Anthropic模型尚未实现")
        elif model_type.lower() == "tongyi":
            return TongyiQianwenModel(**kwargs)
        elif model_type.lower() == "ollama":
            return OllamaModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class ModelResponseParser:
    """
    模型响应解析器，用于解析模型输出中的工具调用和结构化信息
    """
    
    @staticmethod
    def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
        """
        解析响应中的工具调用
        
        @param response: 模型响应字符串
        @return: 工具调用列表
        """
        tool_calls = []
        
        # 检查并解析工具调用标记
        try:
            start_marker = "<|FunctionCallBegin|>"
            end_marker = "<|FunctionCallEnd|>"
            
            if start_marker in response and end_marker in response:
                # 提取工具调用内容
                start_idx = response.find(start_marker) + len(start_marker)
                end_idx = response.find(end_marker)
                tool_call_str = response[start_idx:end_idx].strip()
                
                # 解析JSON并确保返回列表格式
                if tool_call_str:
                    tool_calls = json.loads(tool_call_str)
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
        except Exception as e:
            print(f"解析工具调用失败: {str(e)}")
        
        return tool_calls