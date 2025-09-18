import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from src.prompt.template_loader import template_loader


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
    
    # 预编译正则表达式，避免每次调用format方法时重新编译
    # 使用\w+表示变量名，匹配字母、数字、下划线，避免贪婪匹配和ReDoS风险
    VARIABLE_PATTERN = re.compile(r'\{(\w+)\}')
    
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
        # 预检查模板中需要的所有变量
        # 使用预编译的正则表达式提取模板中所有的变量名
        required_vars = set(match.group(1) for match in self.VARIABLE_PATTERN.finditer(self.template))
        # 找出缺失的变量
        missing_vars = required_vars - set(kwargs.keys())
        
        if missing_vars:
            # 如果有缺失的变量，抛出包含所有缺失变量的异常
            missing_vars_str = ', '.join(missing_vars)
            raise KeyError(f"提示模板中缺少必需的参数: {missing_vars_str}")
        
        try:
            return self.template.format(**kwargs)
        except Exception as e:
            # 捕获其他可能的格式化错误
            raise ValueError(f"提示模板格式化失败: {str(e)}")


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
        从模板加载器获取默认的代理提示模板
        
        @return: 包含默认提示模板的字典
        """
        return template_loader.load_all_templates()

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