from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BasePromptTemplate(ABC):
    """
    提示模板基类，定义了提示模板的基本接口
    """
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        格式化提示模板
        
        @param kwargs: 用于填充模板的参数
        @return: 格式化后的提示字符串
        """
        pass


class SimplePromptTemplate(BasePromptTemplate):
    """
    简单的提示模板实现
    """
    
    def __init__(self, template: str):
        """
        初始化提示模板
        
        @param template: 提示模板字符串，使用{variable}格式标记可替换变量
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """
        格式化提示模板
        
        @param kwargs: 用于填充模板的参数
        @return: 格式化后的提示字符串
        @raises KeyError: 当模板中需要的变量在kwargs中不存在时抛出异常
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"提示模板中缺少必需的参数: {str(e)}")


class PromptManager:
    """
    提示管理器，用于管理多个提示模板和生成提示
    """
    
    def __init__(self):
        """
        初始化提示管理器
        """
        self.templates = {}
    
    def add_template(self, name: str, template: Union[BasePromptTemplate, str]) -> None:
        """
        添加提示模板
        
        @param name: 模板名称
        @param template: 提示模板对象或模板字符串
        """
        if isinstance(template, str):
            template = SimplePromptTemplate(template)
        
        if not isinstance(template, BasePromptTemplate):
            raise TypeError("template必须是BasePromptTemplate的实例或字符串")
        
        self.templates[name] = template
    
    def get_template(self, name: str) -> BasePromptTemplate:
        """
        获取提示模板
        
        @param name: 模板名称
        @return: 提示模板对象
        @raises KeyError: 当模板不存在时抛出异常
        """
        if name not in self.templates:
            raise KeyError(f"不存在名为'{name}'的提示模板")
        
        return self.templates[name]
    
    def generate_prompt(self, name: str, **kwargs) -> str:
        """
        生成提示
        
        @param name: 模板名称
        @param kwargs: 用于填充模板的参数
        @return: 生成的提示字符串
        """
        template = self.get_template(name)
        return template.format(**kwargs)


class AgentPromptTemplates:
    """
    代理提示模板集合
    """
    
    @staticmethod
    def get_default_templates() -> Dict[str, str]:
        """
        获取默认的代理提示模板
        
        @return: 包含默认提示模板的字典
        """
        return {
            "main": """你是一个智能助手，能够使用各种工具来完成任务。
可用工具：{tools}

请根据用户的问题和可用工具，决定是直接回答还是使用工具。
如果需要使用工具，请以以下格式输出：
<|FunctionCallBegin|>[{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}]<|FunctionCallEnd|>

如果不需要使用工具，请直接回答问题。

用户问题：{query}

上下文信息：{context}

请记住，你可以使用工具来获取更多信息，确保你的回答准确。""",
            
            "self_critique": """请评估你之前的回答是否准确、完整和相关。
如果你认为回答不充分，请指出需要改进的地方，并决定是否需要使用工具获取更多信息。

问题：{query}
你的回答：{answer}
评估结果：""",
            
            "rag_answer": """请使用以下检索到的信息来回答用户问题。
确保你的回答完全基于提供的信息，不要添加外部知识。

用户问题：{query}
检索到的信息：{context}
回答：""",
            
            "reflection": """你已经获取了一些信息，请思考这些信息是否足够回答用户问题，
如果不够，考虑需要使用哪些工具获取更多信息。

用户问题：{query}
已获取的信息：{context}
思考过程："""
        }

    @staticmethod
    def create_prompt_manager() -> PromptManager:
        """
        创建并初始化带有默认模板的提示管理器
        
        @return: 初始化后的提示管理器
        """
        manager = PromptManager()
        default_templates = AgentPromptTemplates.get_default_templates()
        
        for name, template_str in default_templates.items():
            manager.add_template(name, template_str)
        
        return manager