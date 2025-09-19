# Agentic RAG 配置管理指南

本文档详细介绍了Agentic RAG系统的配置管理机制，包括配置文件和环境变量的使用方法。系统采用层次化的配置管理，支持从配置文件和环境变量加载配置，环境变量优先级高于配置文件。

## 配置管理概述

Agentic RAG系统采用层次化的配置管理机制，支持从以下来源加载配置：

1. **配置文件**：`config.yaml`（或`config.json`）
2. **环境变量**：以`AGENTIC_RAG_`为前缀的环境变量

配置加载遵循**环境变量优先级高于配置文件**的原则，即当同一配置项在环境变量和配置文件中同时存在时，环境变量的值将覆盖配置文件中的值。

## 配置文件

### 创建配置文件

系统支持YAML和JSON格式的配置文件：

1. 直接编辑`config.yaml`文件，根据您的需求进行配置。

2. 配置文件支持YAML或JSON格式，系统会自动识别。

### 配置文件结构

配置文件采用嵌套的键值对结构，主要包括以下几个部分：

```yaml
# 模型配置
model:
  type: "mock"  # 默认模型类型
  common:
    max_tokens: 512  # 通用参数
  smolagent:  # smolAgent模型特定参数
    model_name: "gpt2"
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

## 环境变量

### 环境变量格式

环境变量使用以下格式：

```
AGENTIC_RAG_{配置键}_...
```

其中，配置键使用下划线分隔嵌套结构。例如，配置文件中的`model.tongyi.api_key`对应的环境变量为`AGENTIC_RAG_MODEL_TONGYI_API_KEY`。

### 常用环境变量示例

```bash
# 设置默认模型类型
AGENTIC_RAG_MODEL_TYPE="ollama"

# 设置通义千问API密钥
AGENTIC_RAG_MODEL_TONGYI_API_KEY="your_api_key_here"

# 设置Ollama模型名称
AGENTIC_RAG_MODEL_OLLAMA_MODEL_NAME="mistral"

# 设置Ollama API地址
AGENTIC_RAG_MODEL_OLLAMA_BASE_URL="http://localhost:11434"

# 设置通用模型参数
AGENTIC_RAG_MODEL_COMMON_MAX_TOKENS="1024"
AGENTIC_RAG_MODEL_COMMON_TEMPERATURE="0.5"
```

### 在Windows中设置环境变量

在Windows系统中，可以通过以下方式设置环境变量：

1. **临时设置（当前命令行会话）**：
   ```powershell
   set AGENTIC_RAG_MODEL_TYPE=ollama
   ```

2. **永久设置**：
   - 右键点击「此电脑」->「属性」->「高级系统设置」->「环境变量」
   - 在「系统变量」或「用户变量」中点击「新建」
   - 输入变量名和变量值
   - 点击「确定」保存

### 在Linux/macOS中设置环境变量

在Linux或macOS系统中，可以通过以下方式设置环境变量：

1. **临时设置（当前终端会话）**：
   ```bash
   export AGENTIC_RAG_MODEL_TYPE=ollama
   ```

2. **永久设置**：
   将环境变量添加到`~/.bashrc`、`~/.zshrc`或`~/.profile`文件中：
   ```bash
   echo 'export AGENTIC_RAG_MODEL_TYPE=ollama' >> ~/.bashrc
   source ~/.bashrc
   ```

## 配置优先级

配置项的优先级从高到低为：

1. **命令行参数**（如果通过命令行直接指定）
2. **环境变量**
3. **配置文件** (`config.yaml`或`config.json`)
4. **默认值**（代码中定义的默认配置）

## 自动类型转换

配置管理器会自动尝试将环境变量的值转换为适当的Python类型：

- `"true"` 和 `"false"`（不区分大小写）会转换为布尔值 `True` 和 `False`
- 数字字符串会转换为整数或浮点数
- JSON格式的字符串会转换为相应的Python对象（列表或字典）
- 其他字符串保持不变

例如：
- `"123"` -> `123`（整数）
- `"0.5"` -> `0.5`（浮点数）
- `"true"` -> `True`（布尔值）
- `"[1,2,3]"` -> `[1, 2, 3]`（列表）
- `"{\"key\": \"value\"}"` -> `{"key": "value"}`（字典）

## 使用配置示例

### 在程序中访问配置

配置管理器提供了便捷的函数来访问配置：

```python
from src.config import get_config, set_config, save_config, reload_config

# 获取配置值
model_type = get_config("model.type", "mock")  # 第二个参数是默认值
api_key = get_config("model.tongyi.api_key")

# 设置配置值
set_config("model.ollama.model_name", "llama3")

# 保存配置到配置文件
save_config()

# 重新加载配置
reload_config()
```

### 运行时配置交互

当您运行Agentic RAG系统时，如果配置文件中已经存在配置，系统会询问您是否使用这些配置：

```
初始化Agentic RAG系统...
系统支持从配置文件(config.yaml)和环境变量加载配置，环境变量优先级高于配置文件。
环境变量格式: AGENTIC_RAG_{配置键}_...，使用下划线分隔嵌套键。例如: AGENTIC_RAG_MODEL_TONGYI_API_KEY
检测到配置文件中的模型类型为 'ollama'，是否使用？(y/n，默认y): 
```

您可以选择使用配置文件中的值，或者输入新的值。

## 配置最佳实践

1. **安全性**：不要将敏感信息（如API密钥）直接硬编码在代码中，使用配置文件或环境变量
2. **版本控制**：不要将包含敏感信息的配置文件提交到版本控制系统
3. **环境隔离**：为不同环境（开发、测试、生产）使用不同的配置文件
4. **文档化**：为您的配置添加注释，说明每个配置项的用途

## 故障排除

如果遇到配置相关的问题，可以尝试以下方法：

1. 检查配置文件是否存在且格式正确
2. 验证环境变量是否正确设置
3. 使用`print`语句或日志输出当前加载的配置值进行调试
4. 确保配置键的大小写和嵌套结构正确