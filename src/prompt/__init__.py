"""
提示模板管理模块，负责管理和生成代理使用的提示模板
"""
# 导出prompt相关的类和函数
from .prompt_manager import BasePromptTemplate, SimplePromptTemplate, PromptManager, AgentPromptTemplates
from .template_loader import PromptTemplateLoader

__all__ = [
    "BasePromptTemplate",
    "SimplePromptTemplate",
    "PromptManager",
    "PromptTemplateLoader",
    "AgentPromptTemplates"
]