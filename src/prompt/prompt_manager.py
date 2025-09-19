import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# 使用smolagents的模板功能
from smolagents import Template
from smolagents.agents import populate_template


class Jinja2PromptTemplate:
    """
    基于smolagents的模板实现
    """
    
    def __init__(self, template: str):
        """
        初始化提示模板
        
        @param template: 提示模板字符串，支持Jinja2语法
        """
        # 保存原始模板字符串
        self.template_str = template
        # 使用smolagents的Template类
        self.template = Template(template)
        # 提取模板中使用的变量名
        self.required_vars = self._extract_required_variables(template)
    
    def _extract_required_variables(self, template: str) -> List[str]:
        """
        提取模板中使用的变量名
        
        @param template: 模板字符串
        @return: 变量名列表
        """
        # 正则表达式匹配Jinja2语法中的变量
        # 匹配 {{ variable }}, {{ variable.property }}, {{ variable|filter }}, {{ variable.property|filter }}
        # 也匹配简单的 {variable} 格式
        pattern = r'\{\{\s*([\w.]+)\s*(?:\|\s*\w+\s*)*\}\}|\{(\w+)\}'
        matches = re.findall(pattern, template)
        
        # 提取变量名，去除可能的属性访问（如 obj.prop 中的 obj）
        variables = []
        for match in matches:
            # 处理Jinja2语法的匹配
            if match[0]:
                var_name = match[0].split('.')[0]
                variables.append(var_name)
            # 处理简单的 {variable} 格式
            elif match[1]:
                variables.append(match[1])
        
        # 去重并返回
        return list(set(variables))
    
    def format(self, **kwargs) -> str:
        """
        格式化提示模板，替换参数
        
        @param kwargs: 用于填充模板的参数
        @return: 格式化后的提示字符串
        """
        # 检查是否提供了所有必需的变量
        missing_vars = []
        for var in self.required_vars:
            if var not in kwargs:
                missing_vars.append(var)
        
        # 如果有缺少的变量，抛出KeyError异常
        if missing_vars:
            if len(missing_vars) == 1:
                raise KeyError(f"缺少必需的参数: {missing_vars[0]}")
            else:
                raise KeyError(f"缺少必需的参数: {', '.join(missing_vars)}")
        
        try:
            # 使用smolagents的populate_template函数渲染模板
            return populate_template(self.template_str, kwargs)
        except Exception as e:
            # 处理其他可能的异常
            raise RuntimeError(f"渲染模板时出错: {str(e)}")


class PromptManager:
    """
    提示管理器，用于管理多个提示模板和生成提示
    """
    
    def __init__(self):
        """
        初始化提示管理器
        """
        self.templates: Dict[str, Jinja2PromptTemplate] = {}
    
    def add_template(self, name: str, template: Union[Jinja2PromptTemplate, str]) -> None:
        """
        添加提示模板
        
        @param name: 模板名称
        @param template: 提示模板对象或模板字符串
        """
        if isinstance(template, str):
            template = Jinja2PromptTemplate(template)
        
        if not isinstance(template, Jinja2PromptTemplate):
            raise TypeError("template必须是Jinja2PromptTemplate的实例或字符串")
        
        self.templates[name] = template
    
    def get_template(self, name: str) -> Jinja2PromptTemplate:
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


class PromptTemplateLoader:
    """
    提示模板加载器，负责从文件系统加载提示模板
    """
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        初始化模板加载器
        
        @param templates_dir: 模板目录路径，如果为None则使用默认路径
        """
        self._templates_dir = templates_dir or self._get_default_templates_dir()
        self._template_cache: Dict[str, str] = {}
    
    def _get_default_templates_dir(self) -> Path:
        """
        获取默认模板目录的路径
        """
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        # 返回templates子目录
        return current_dir / "templates"
    
    def load_template(self, template_name: str, use_cache: bool = True) -> str:
        """
        加载指定名称的模板
        
        @param template_name: 模板名称
        @param use_cache: 是否使用缓存的模板
        @return: 模板内容字符串
        @raises FileNotFoundError: 当模板文件不存在时抛出异常
        """
        # 检查缓存
        if use_cache and template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # 构建模板文件路径
        template_file = self._templates_dir / f"{template_name}.txt"
        
        # 检查文件是否存在
        if not template_file.exists():
            raise FileNotFoundError(f"找不到模板文件: {template_file}")
        
        # 读取模板内容
        with open(template_file, "r", encoding="utf-8") as f:
            template_content = f.read()
        
        # 缓存模板内容
        self._template_cache[template_name] = template_content
        
        return template_content
    
    def clear_cache(self) -> None:
        """
        清除模板缓存
        """
        self._template_cache.clear()


class AgentPromptTemplates:
    """
    代理提示模板集合，提供便捷的模板管理功能
    """
    
    @staticmethod
    def get_default_templates() -> Dict[str, str]:
        """
        从文件系统加载默认的代理提示模板
        
        @return: 包含默认提示模板的字典
        """
        templates = {}
        loader = PromptTemplateLoader()
        templates_dir = loader._get_default_templates_dir()
        
        # 遍历templates目录中的所有txt文件
        for template_file in templates_dir.glob("*.txt"):
            # 使用文件名（不含扩展名）作为模板名称
            template_name = template_file.stem
            # 读取模板内容
            with open(template_file, "r", encoding="utf-8") as f:
                templates[template_name] = f.read()
        
        return templates

    @staticmethod
    def create_prompt_manager() -> PromptManager:
        """
        创建并初始化带有默认模板的提示管理器
        
        @return: 初始化后的提示管理器
        """
        manager = PromptManager()
        for name, template_str in AgentPromptTemplates.get_default_templates().items():
            manager.add_template(name, template_str)
        
        return manager


# 创建全局的模板加载器实例
prompt_template_loader = PromptTemplateLoader()