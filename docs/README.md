# Agentic RAG 系统

一个基于smolAgent的高级代理式检索增强生成（Agentic RAG）系统，结合了智能代理、检索增强和答案校验功能，支持多种大语言模型集成。

## 项目结构

```
MyAgenticRAG/
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── README.md            # 项目说明文档
├── docs/                # 文档目录
│   ├── CONFIGURATION_GUIDE.md         # 配置管理指南
│   ├── LLM_API_INTEGRATION_GUIDE.md   # 大模型API集成指南
│   ├── PROMPT_MODULE_DOCUMENTATION.md # 提示模块文档
│   └── README.md        # 项目说明文档
└── src/
    ├── config.py        # 配置管理模块
    ├── agent/           # 代理模块
    │   ├── agent.py         # 代理核心逻辑
    │   └── agent_builder.py # 代理构建器
    ├── tools/           # 工具模块
    │   ├── base_tool.py       # 工具基类
    │   └── retrieval_tool.py  # 检索工具实现
    ├── doc_process/     # 文档处理模块
    │   ├── base_processor.py   # 文档处理器基类
    │   └── simple_processor.py # 简单文档处理器实现
    ├── agent_context.py # 代理上下文和记忆管理
    ├── prompt/          # 提示模板管理模块
    │   ├── prompt_manager.py   # 提示管理器实现
    │   ├── template_loader.py  # 模板加载器实现
    │   └── templates/          # 提示模板文件
    └── model/           # 语言模型接口模块
        └── language_model.py   # 语言模型实现
```

## 功能特点

1. **智能代理**：基于smolAgent实现的智能代理，能够自主决策、使用工具和迭代优化
2. **检索增强**：集成文档检索功能，为回答提供相关上下文信息
3. **工具调用**：支持动态工具调用，可扩展多种工具（检索、搜索等）
4. **工具使用历史记录**：完整记录所有工具的使用历史，包括工具名称、查询参数和结果质量，提高上下文理解能力
5. **答案校验**：包含自我批评和反思机制，确保回答质量
6. **记忆管理**：记录代理交互历史，支持上下文理解
7. **模块化设计**：采用高内聚低耦合的设计原则，便于扩展和维护
8. **多模型支持**：支持多种大语言模型，包括Ollama本地模型、通义千问等
9. **灵活配置**：支持通过配置文件和环境变量进行系统配置，环境变量优先级高于配置文件
10. **迭代优化**：支持答案的多轮迭代优化，不断改进回答质量

## 安装依赖

项目依赖在`requirements.txt`文件中定义，包括基础依赖、文档处理、网络请求、向量存储和其他工具库。

```bash
# 安装项目依赖
pip install -r requirements.txt

# 部分模型可能需要额外安装特定依赖
# 例如，通义千问模型需要dashscope库
# pip install dashscope>=1.10.0

# Ollama本地模型需要先安装Ollama客户端
# 详见: https://ollama.com/
```

## 使用方法

### 1. 创建配置文件（可选但推荐）

在项目根目录创建`config.yaml`文件，添加以下配置示例：

```yaml
# 模型配置
model:
  type: "ollama"  # 可选: mock, ollama, tongyi等
  common:
    max_tokens: 1024  # 通用参数
  tongyi:  # 通义千问模型特定参数
    model_name: "qwen-max"
    api_key: "your_api_key_here"
  ollama:  # Ollama模型特定参数
    model_name: "llama3"
    base_url: "http://localhost:11434"

# 代理配置
agent:
  max_retries: 3
  max_tool_calls: 5

# 文档处理配置
doc_process:
  chunk_size: 1000
  chunk_overlap: 200

# 搜索工具配置
search:
  max_results: 5

# 向量存储配置
vector_store:
  top_k: 3
```

### 2. 设置环境变量（可选，优先级高于配置文件）

环境变量使用以下格式：`AGENTIC_RAG_{配置键}_...`

示例：
```bash
# Windows系统
set AGENTIC_RAG_MODEL_TYPE=ollama
set AGENTIC_RAG_MODEL_TONGYI_API_KEY=your_api_key_here

# Linux/macOS系统
export AGENTIC_RAG_MODEL_TYPE=ollama
export AGENTIC_RAG_MODEL_TONGYI_API_KEY=your_api_key_here
```

### 3. 运行程序

```bash
python main.py
```

程序启动后，您可以输入问题进行交互。输入'quit'、'exit'或'退出'结束程序。

## 核心模块说明

### 1. 配置模块 (config.py)

实现了配置管理功能，支持从配置文件和环境变量加载配置，环境变量优先级高于配置文件。提供了便捷的函数来获取、设置和保存配置。

### 2. 代理模块 (agent/)

#### agent.py
实现了AgenticRAG类，是整个系统的核心，负责协调各个组件完成任务。包含查询分析、代理循环、工具调用、答案评估和优化等核心逻辑。

#### agent_builder.py
提供了创建AgenticRAG实例的便捷函数，负责组装各个组件（模型、工具、文档处理器等）。

### 3. 工具模块 (tool/)

提供了工具的基类和具体实现，包括检索工具等。支持动态工具调用和扩展。

### 4. 文档处理模块 (doc_process/)

负责文档的加载、解析和分割，为检索提供预处理功能。

### 5. 环境模块 (env/)

管理代理的运行环境和状态，包括记忆管理和交互历史记录。

### 6. 提示模块 (prompt/)

管理各种提示模板，支持动态生成提示。包含提示管理器和提示模板文件。

### 7. 模型模块 (model/)

提供了与语言模型交互的接口，支持不同类型的模型，包括：
- Ollama本地模型（如llama3、mistral等）
- 阿里云通义千问模型
- 其他大语言模型（可扩展）

## 扩展指南

### 1. 添加新工具

1. 在`src/tool/`目录下创建一个新的Python文件
2. 创建一个继承自BaseTool的新类
3. 实现必要的方法，特别是`run`方法
4. 在`create_agentic_rag`函数中添加到工具列表

示例：
```python
from src.tool.base_tool import BaseTool
from typing import Dict, Any

class MyNewTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_new_tool", description="我的新工具描述")
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 实现工具逻辑
        result = {"output": "工具执行结果"}
        return result
```

### 2. 添加新的语言模型

1. 在`src/model/language_model.py`文件中添加新的模型实现类
2. 继承BaseLanguageModel类
3. 实现必要的方法，如`generate`和`generate_with_tools`
4. 在ModelFactory类中添加新的创建逻辑
5. 在配置文件或环境变量中设置相应的模型参数

详见`docs/LLM_API_INTEGRATION_GUIDE.md`文件获取更多详细信息。

### 3. 自定义文档处理器

1. 在`src/doc_process/`目录下创建一个新的Python文件
2. 创建一个继承自DocumentProcessor的新类
3. 实现必要的方法，如`process`和`load`
4. 在`create_agentic_rag`函数中替换文档处理器

示例：
```python
from src.doc_process.base_processor import DocumentProcessor
from typing import List, Dict, Any

class MyDocumentProcessor(DocumentProcessor):
    def process(self, documents: List[str]) -> List[Dict[str, Any]]:
        # 实现文档处理逻辑
        processed_docs = []
        for doc in documents:
            processed_docs.append({
                "content": doc,
                "metadata": {}
            })
        return processed_docs
    
    def load(self, file_path: str) -> List[str]:
        # 实现文档加载逻辑
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
```

## 注意事项

1. **配置安全**：不要将包含敏感信息（如API密钥）的配置文件提交到版本控制系统
2. **环境变量优先级**：环境变量的配置优先级高于配置文件，使用时请注意
3. **模型可用性**：
   - Ollama本地模型需要先安装Ollama客户端并下载相应模型
   - 通义千问等第三方API模型需要有效的API密钥
4. **系统限制**：
   - 默认的向量存储和搜索引擎是模拟的，实际应用中需要替换为真实的数据库和搜索引擎
   - 系统支持的最大工具调用次数和重试次数可以在配置中修改
5. **开发环境**：建议在虚拟环境中运行本项目，避免依赖冲突
6. **日志与调试**：系统运行过程中的重要信息会打印到控制台，可用于调试

## 故障排除

### 1. 模型连接问题

**问题**：无法连接到Ollama本地模型
**解决方法**：
- 确保Ollama客户端已经安装并正在运行
- 检查配置中的`base_url`是否正确（默认为`http://localhost:11434`）
- 确认所需的模型已经下载完成

**问题**：通义千问API调用失败
**解决方法**：
- 检查API密钥是否正确设置
- 确认API密钥是否有效且有足够的调用额度
- 检查网络连接是否正常

### 2. 配置问题

**问题**：配置不生效
**解决方法**：
- 检查配置文件格式是否正确（YAML或JSON）
- 确认环境变量的格式是否正确（`AGENTIC_RAG_{配置键}_...`）
- 注意环境变量的优先级高于配置文件

### 3. 依赖问题

**问题**：缺少依赖包
**解决方法**：
- 运行`pip install -r requirements.txt`安装所有依赖
- 对于特定模型，可能需要额外安装依赖（如`pip install dashscope`）

### 4. 其他问题

如果遇到其他问题，请检查控制台输出的错误信息，这通常会提供问题的详细描述。您也可以参考`docs`目录下的文档获取更多信息。

## License

MIT