# Agentic RAG 系统

一个基于smolAgent的代理式检索增强生成（Agentic RAG）系统，支持智能文档检索、自动工具选择和答案校验。

## 项目结构

```
MyAgenticRAG/
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── create_dirs.py       # 目录创建脚本
└── src/
    ├── agent.py         # 代理核心逻辑
    ├── tool/            # 工具模块
    │   ├── base_tool.py       # 工具基类
    │   └── retrieval_tool.py  # 检索工具实现
    ├── doc_process/     # 文档处理模块
    │   ├── base_processor.py   # 文档处理器基类
    │   └── simple_processor.py # 简单文档处理器实现
    ├── env/             # 环境和状态管理模块
    │   └── agent_env.py        # 代理环境实现
    ├── prompt/          # 提示模板管理模块
    │   └── prompt_manager.py   # 提示管理器实现
    └── model/           # 语言模型接口模块
        └── language_model.py   # 语言模型实现
```

## 功能特点

1. **智能代理**：基于smolAgent实现的智能代理，能够自主决策和使用工具
2. **检索增强**：集成文档检索功能，为回答提供相关上下文信息
3. **工具调用**：支持动态工具调用，可扩展多种工具
4. **答案校验**：包含自我批评和反思机制，确保回答质量
5. **记忆管理**：记录代理交互历史，支持上下文理解
6. **模块化设计**：采用高内聚低耦合的设计原则，便于扩展和维护

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

直接运行主程序：

```bash
python main.py
```

程序启动后，您可以输入问题进行交互。输入'quit'、'exit'或'退出'结束程序。

## 核心模块说明

### 1. 代理模块 (agent.py)

实现了AgenticRAG类，是整个系统的核心，负责协调各个组件完成任务。

### 2. 工具模块 (tool/)

提供了工具的基类和具体实现，包括检索工具等。

### 3. 文档处理模块 (doc_process/)

负责文档的加载、解析和分割，为检索提供预处理功能。

### 4. 环境模块 (env/)

管理代理的运行环境和状态，包括记忆管理。

### 5. 提示模块 (prompt/)

管理各种提示模板，支持动态生成提示。

### 6. 模型模块 (model/)

提供了与语言模型交互的接口，支持不同类型的模型。

## 扩展指南

### 添加新工具

1. 创建一个继承自BaseTool的新类
2. 实现run方法
3. 在main.py中添加到工具列表

### 自定义文档处理器

1. 创建一个继承自DocumentProcessor的新类
2. 实现process和load方法
3. 在create_agentic_rag函数中替换文档处理器

### 使用不同的语言模型

1. 在ModelFactory中添加新的模型实现
2. 在create_agentic_rag函数中指定模型类型

## 注意事项

1. 本项目使用了模拟的向量存储和搜索引擎进行演示
2. 在实际应用中，需要替换为真实的向量数据库和搜索引擎
3. smolAgent的具体使用可能需要配置API密钥

## License

MIT