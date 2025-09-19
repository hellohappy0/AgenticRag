"""
提示模板模块

此模块提供了提示模板的管理和加载功能，基于smolagents框架实现。
"""

from .prompt_manager import (
    PromptManager,
    Jinja2PromptTemplate,
    AgentPromptTemplates,
    prompt_template_loader
)

__all__ = [
    'PromptManager',
    'Jinja2PromptTemplate',
    'AgentPromptTemplates',
    'prompt_template_loader'
]