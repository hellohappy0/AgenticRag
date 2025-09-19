# 大模型API集成指南

本指南将帮助您了解当前常用的大模型API接口及其获取方式，并提供如何将这些真实模型集成到Agentic RAG系统中的详细步骤。

## 一、常用大模型API接口介绍

### 1. Ollama本地部署模型
- **模型**: llama3、mistral、gemma等多种开源大模型
- **特点**: 本地部署，无需联网，隐私性好，完全免费
- **官方网站**: https://ollama.com/

### 2. 阿里云通义千问
- **模型**: qwen-max、qwen-plus等
- **特点**: 中文表现优异，响应速度快
- **官方网站**: https://dashscope.aliyun.com/

### 3. 讯飞星火认知大模型
- **模型**: SparkDesk系列
- **特点**: 语音处理能力强，多模态支持
- **官方网站**: https://xinghuo.xfyun.cn/

### 4. 百度文心一言
- **模型**: ERNIE-Bot系列
- **特点**: 中文理解能力强，知识覆盖广
- **官方网站**: https://cloud.baidu.com/product/wenxinworkshop.html

### 5. Anthropic Claude系列
- **模型**: Claude 2、Claude 3 Opus/Sonnet/Haiku
- **特点**: 长文本处理能力强，安全性能好
- **官方网站**: https://www.anthropic.com/

### 6. 字节跳动豆包
- **模型**: Doubao系列
- **特点**: 创意内容生成能力出色
- **官方网站**: https://www.doubao.com/

## 二、获取大模型API的一般步骤

### 1. 注册账号
访问相应平台的官方网站，注册开发者账号。

### 2. 申请API密钥
- 登录开发者平台
- 找到API密钥管理页面
- 创建新的API密钥
- 保存密钥信息（通常需要保密保存）

### 3. 了解API使用规范
- 阅读平台的服务条款和使用政策
- 了解API的调用限制、计费方式等
- 查看API文档，了解请求格式和参数

### 4. 安装必要的依赖库
大多数平台提供官方SDK，或可以使用通用的HTTP库如requests进行调用。

## 三、集成大模型API到Agentic RAG系统

### 1. 模型实现类结构
项目在`src/model/language_model.py`文件中实现了几种语言模型，包括基于smolAgent的模型、Ollama模型和通义千问模型。以下是主要模型实现的概述：

#### SmolAgent模型实现

```python
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
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """
        初始化ToolCallingAgent
        """
        try:
            # 创建模型实例
            if self.api_key:
                self.model = TransformersModel(model_id=self.model_name, token=self.api_key)
            else:
                self.model = TransformersModel(model_id=self.model_name)
            
            # 初始时不设置工具
            self.agent = None
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
            # 如果没有agent，创建一个没有工具的agent
            if self.agent is None:
                self.agent = ToolCallingAgent(model=self.model, tools=[])
            
            # 生成响应
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
        # 实现工具调用逻辑
        # ...
```

#### 其他模型实现

项目还包括Ollama模型和通义千问模型的实现，它们都继承自BaseLanguageModel抽象基类，并实现了generate和generate_with_tools方法。在使用这些模型时，需要确保安装了相应的依赖库（如dashscope库用于通义千问模型）。

### 2. ModelFactory类使用

项目使用`create_agentic_rag`函数来创建Agentic RAG实例，该函数会根据配置选择合适的模型类型：

```python
# 创建Agentic RAG实例
agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key, custom_model=custom_model)
```

其中，model_type可以是：
- `mock`: 模拟模型（用于测试）
- `ollama`: Ollama本地模型
- `tongyi`: 通义千问模型
- 其他支持的模型类型

### 3. 配置模型参数

在使用真实模型时，需要在配置文件或环境变量中设置相应的参数：

```yaml
# 配置文件示例
model:
  type: "ollama"  # 或 "tongyi"
  ollama:
    model_name: "llama3"
    base_url: "http://localhost:11434"
  tongyi:
    model_name: "qwen-max"
    api_key: "your_api_key_here"
```

### 2. 更新ModelFactory类
在`ModelFactory`类中添加对新模型的支持：

```python
class ModelFactory:
    """
    模型工厂类，用于创建不同类型的语言模型实例
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseLanguageModel:
        """
        创建语言模型实例
        """
        if model_type.lower() == "smolagent":
            return SmolAgentModel(**kwargs)
        elif model_type.lower() == "tongyi":
            return TongyiQianwenModel(**kwargs)
        elif model_type.lower() == "ollama":
            return OllamaModel(**kwargs)
        elif model_type.lower() == "anthropic":
            # 可以添加Anthropic Claude模型的实现
            raise NotImplementedError("Anthropic模型尚未实现")
        elif model_type.lower() == "spark":
            # 可以添加讯飞星火模型的实现
            raise NotImplementedError("讯飞星火模型尚未实现")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
```

### 3. 修改main.py中的模型创建逻辑
在`create_agentic_rag()`函数中，添加对真实模型的支持：

```python
def create_agentic_rag(model_type: str = "mock", api_key: str = None, model_name: str = None) -> AgenticRAG:
    """
    创建并初始化Agentic RAG系统
    
    @param model_type: 模型类型 (mock, smolagent, tongyi, ollama等)
    @param api_key: API密钥
    @param model_name: 可选的模型名称，用于Ollama等模型
    @return: 初始化后的Agentic RAG实例
    """
    # ... [其他代码保持不变] ...
    
    # 创建语言模型
    try:
        if model_type.lower() == "mock":
            model = MockLanguageModel()
        else:
            model_params = {
                "max_tokens": 512
            }
            
            if model_type.lower() == "tongyi":
                # 通义千问模型参数
                model_params["model_name"] = "qwen-max"
                if api_key:
                    model_params["api_key"] = api_key
                else:
                    raise ValueError("使用通义千问模型需要提供API密钥")
            elif model_type.lower() == "smolagent":
                # smolAgent模型参数
                model_params["model_name"] = "gpt2"
                if api_key:
                    model_params["api_key"] = api_key
            elif model_type.lower() == "ollama":
                # Ollama模型参数
                model_params["model_name"] = model_name or "llama3"
                # Ollama通常不需要API密钥
            
            model = ModelFactory.create_model(
                model_type,
                **model_params
            )
    except Exception as e:
        print(f"创建{model_type}模型失败，使用模拟模型: {str(e)}")
        model = MockLanguageModel()
    
    # ... [其他代码保持不变] ...
```

### 4. 更新主函数以支持模型选择
修改`main()`函数，允许用户选择使用的模型：

```python
def main():
    """
    主函数，演示Agentic RAG的使用
    """
    print("初始化Agentic RAG系统...")
    
    # 获取用户选择的模型类型
    model_type = input("请选择模型类型 (mock/smolagent/tongyi/ollama，默认mock): ").strip().lower() or "mock"
    api_key = None
    custom_model = None
    
    if model_type != "mock":
        if model_type == "tongyi":
            api_key = input("请输入阿里云通义千问的API密钥: ").strip()
        elif model_type == "ollama":
            # Ollama通常不需要API密钥，这里可以添加自定义配置选项
            custom_model_input = input("请输入Ollama模型名称 (默认llama3，可选): ").strip()
            if custom_model_input:
                custom_model = custom_model_input
        else:
            api_key_input = input(f"请输入{model_type}的API密钥 (可选): ").strip()
            if api_key_input:
                api_key = api_key_input
    
    # 创建Agentic RAG实例
    if custom_model:
        agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key, model_name=custom_model)
    else:
        agentic_rag = create_agentic_rag(model_type=model_type, api_key=api_key)
    
    # ... [其他代码保持不变] ...
```

## 四、安装必要的依赖

在`requirements.txt`文件中添加新的依赖：

```txt
# 大模型API依赖
requests>=2.28.0
dashscope>=1.10.0  # 通义千问依赖
# Ollama模型不需要额外依赖，使用现有的requests库即可
```

## 五、安全注意事项

1. **API密钥保护**：不要将API密钥硬编码在代码中，建议通过环境变量或配置文件加载
2. **请求限制**：注意各平台的API调用频率和配额限制，避免超出限制
3. **错误处理**：添加适当的错误处理逻辑，处理API调用可能出现的各种异常
4. **数据隐私**：注意不要将敏感数据发送到第三方API

## 六、扩展建议

1. **添加缓存机制**：对频繁使用的查询结果进行缓存，减少API调用次数
2. **实现异步调用**：使用异步编程提高系统的并发处理能力
3. **添加模型评测功能**：比较不同模型在特定任务上的表现
4. **支持多模型协作**：根据任务特点选择最合适的模型

## 七、常见问题解决

1. **API调用失败**：检查API密钥是否正确，网络连接是否正常
2. **超出配额限制**：调整调用频率，或考虑升级账号套餐
3. **响应不符合预期**：优化提示词工程，调整模型参数
4. **性能问题**：添加请求超时设置，实现并发处理

通过以上步骤，您可以将真实的大模型API集成到Agentic RAG系统中，提升系统的智能水平和实用性。