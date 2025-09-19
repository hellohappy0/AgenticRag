# Prompt 模块文档

## 1. 模块概述

Prompt模块是Agentic RAG系统的核心组件之一，负责管理、加载和生成各种任务所需的提示模板。该模块提供了灵活的模板管理机制，支持动态加载、缓存和生成提示，使系统能够根据不同任务场景生成合适的提示。

## 2. 目录结构

```
src/prompt/
├── __init__.py          # 模块导出文件
├── prompt_manager.py    # 提示管理器实现
├── template_loader.py   # 模板加载器实现
├── templates/           # 提示模板文件目录
│   ├── main.txt                 # 主代理提示模板
│   ├── iteration_optimization.txt  # 迭代优化提示模板
│   ├── answer_evaluation.txt    # 答案评估提示模板
│   ├── query_rewrite.txt        # 查询重写提示模板
│   ├── rag_answer.txt           # RAG回答生成提示模板
│   ├── reflection.txt           # 反思提示模板
│   └── self_critique.txt        # 自我批评提示模板
└── PROMPT_MODULE_DOCUMENTATION.md  # 模块文档
```

## 3. 核心类说明

### 3.1 PromptManager

`PromptManager`类是提示管理的核心，负责根据模板名称和参数生成具体的提示。

**主要功能**：
- 管理和缓存已加载的提示模板
- 根据模板名称和参数动态生成提示
- 支持模板的增删改查操作

**关键方法**：
- `generate_prompt(template_name, **kwargs)`: 根据模板名称和参数生成提示
- `add_template(template_name, template)`: 添加新的提示模板
- `remove_template(template_name)`: 删除指定的提示模板
- `get_template(template_name)`: 获取指定的提示模板

### 3.2 PromptTemplateLoader

`PromptTemplateLoader`类实现了单例模式，负责从文件系统加载提示模板文件。

**主要功能**：
- 从指定目录加载提示模板文件
- 缓存已加载的模板，提高性能
- 支持模板的热重载

**关键方法**：
- `get_instance()`: 获取单例实例
- `load_template(template_name)`: 加载指定名称的模板
- `load_all_templates()`: 加载所有可用的模板
- `reload_templates()`: 重新加载所有模板

### 3.3 BasePromptTemplate 和 SimplePromptTemplate

`BasePromptTemplate`是所有提示模板的抽象基类，`SimplePromptTemplate`是其具体实现。

**主要功能**：
- 定义提示模板的基本接口
- 支持参数化提示生成
- 实现模板内容的格式化

**关键方法**：
- `format(**kwargs)`: 格式化模板内容，替换参数
- `get_content()`: 获取模板原始内容

### 3.4 AgentPromptTemplates

`AgentPromptTemplates`类是一个提示模板集合，包含了代理系统常用的提示模板。

**主要功能**：
- 提供预定义的代理提示模板
- 集中管理代理系统的提示资源

## 4. 提示模板详解

### 4.1 main.txt

主代理提示模板，用于指导代理的整体行为和决策过程。

**主要用途**：
- 定义代理的角色和任务
- 提供工具使用的指导原则
- 指导代理的决策流程
- 包含迭代优化的相关指示
- 考虑工具使用历史记录

**主要参数**：
- `query`: 用户查询
- `context`: 上下文信息
- `tools`: 可用工具列表
- `last_evaluation`: 上一次评估结果
- `optimization_plan`: 优化计划
- `tool_usage_history`: 历史工具使用记录（包含工具名称、查询和结果质量）

### 4.2 iteration_optimization.txt

迭代优化专家提示模板，用于根据评估结果优化回答。

**主要用途**：
- 分析当前回答的不足
- 制定优化计划
- 指导下一步行动

**主要参数**：
- `query`: 用户查询
- `answer`: 当前回答
- `context`: 上下文信息
- `evaluation`: 评估结果

### 4.3 answer_evaluation.txt

答案评估提示模板，用于从多个维度评估生成的回答。

**主要用途**：
- 评估回答的准确性
- 评估回答的完整性
- 评估回答的相关性
- 评估回答的清晰度
- 评估回答的证据支持

**主要参数**：
- `query`: 用户查询
- `answer`: 生成的回答
- `context`: 上下文信息

### 4.4 query_rewrite.txt

查询重写提示模板，用于分析和优化用户查询。

**主要用途**：
- 分析用户查询是否清晰、具体
- 判断是否需要改写或拆解查询
- 生成更适合检索的查询表述

**主要参数**：
- `query`: 用户原始查询

### 4.5 rag_answer.txt

RAG回答生成提示模板，用于基于上下文信息生成准确的回答。

**主要用途**：
- 指导模型基于上下文生成回答
- 确保回答的准确性和相关性
- 避免添加未提及的信息

**主要参数**：
- `query`: 用户查询
- `context`: 上下文信息

### 4.6 reflection.txt

反思提示模板，用于系统地分析解决问题的过程。

**主要用途**：
- 反思决策过程
- 评估工具使用效果
- 分析信息处理方式
- 评估回答质量

**主要参数**：
- `query`: 用户问题
- `thought_process`: 思考过程

### 4.7 self_critique.txt

自我批评提示模板，用于严格检查AI生成的回答质量。

**主要用途**：
- 评估回答的准确性
- 评估回答的完整性
- 评估回答的相关性
- 评估回答的清晰度
- 提出具体的修改建议

**主要参数**：
- `answer`: 原始回答
- `query`: 用户问题
- `context`: 上下文信息

## 5. 在Agent中的使用

在`AgenticRAG`类中，提示模块的使用主要体现在以下几个方面：

1. **查询分析**：使用`query_rewrite`模板分析和优化用户查询
   ```python
   query_analysis_prompt = self.prompt_manager.generate_prompt(
       "query_rewrite",
       query=state["query"]
   )
   ```

2. **主代理提示生成**：使用`main`模板生成代理决策提示
   ```python
   prompt = self.prompt_manager.generate_prompt(
       "main",
       query=state["query"],
       context=context_to_use,
       tools=tools_str,
       last_evaluation=json.dumps(last_evaluation, ensure_ascii=False) if last_evaluation else "无",
       optimization_plan=json.dumps(optimization_plan, ensure_ascii=False) if optimization_plan else "无",
       tool_usage_history=json.dumps(state["tool_usage_history"], ensure_ascii=False) if state.get("tool_usage_history") else "无"
   )
   ```

3. **答案生成**：使用`rag_answer`模板基于上下文生成回答
   ```python
   answer_prompt = self.prompt_manager.generate_prompt(
       "rag_answer",
       query=state["query"],
       context=state["context"]
   )
   ```

4. **答案评估**：使用`answer_evaluation`模板评估生成的回答
   ```python
   evaluation_prompt = self.prompt_manager.generate_prompt(
       "answer_evaluation",
       query=state["query"],
       answer=state["answer"],
       context=state["context"]
   )
   ```

5. **迭代优化**：使用`iteration_optimization`模板优化回答
   ```python
   optimization_prompt = self.prompt_manager.generate_prompt(
       "iteration_optimization",
       query=state["query"],
       answer=state["answer"],
       context=state["context"],
       evaluation=json.dumps(state["evaluation"], ensure_ascii=False)
   )
   ```

## 6. 改进建议

基于当前实现，提出以下改进建议：

1. **增加模板使用监控**：添加统计功能，记录每个模板的使用次数和效果，便于后续优化

2. **优化模板缓存机制**：目前的缓存机制较为简单，可以考虑增加缓存过期时间和自动刷新功能

3. **统一模板参数格式**：标准化各个模板的参数命名和格式，提高一致性

4. **添加模板版本控制**：为模板文件添加版本标识，便于追踪和管理模板变更

5. **实现模板热重载**：添加监控机制，当模板文件变更时自动重新加载，无需重启服务

6. **使用反射和自我批评模板**：当前系统中`reflection.txt`和`self_critique.txt`模板尚未在主流程中使用，建议将其集成到代理的迭代优化过程中

7. **增加模板验证功能**：在加载模板时验证其格式和参数，提前发现问题

8. **提供模板测试工具**：创建专用工具，方便测试和调试不同模板的效果

## 7. 检查结果总结

经过全面检查，prompt相关内容整体结构合理，实现完整，但存在以下几点需要完善的地方：

1. **未使用的模板**：`reflection.txt`和`self_critique.txt`模板已创建但未在主流程中使用

2. **文档缺失**：缺少对整个prompt模块的系统说明和使用指南

3. **优化空间**：模板管理和缓存机制可以进一步优化

4. **一致性**：各模板的参数命名和格式可以进一步标准化

通过实施上述改进建议，可以使prompt模块更加内聚、高效和易于维护。