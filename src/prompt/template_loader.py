import os
from pathlib import Path
from typing import Dict, Optional


class PromptTemplateLoader:
    """
    提示模板加载器，使用单例模式，负责从文件系统加载提示模板
    """
    _instance: Optional['PromptTemplateLoader'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptTemplateLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化模板加载器
        只在第一次创建实例时执行初始化
        """
        if not PromptTemplateLoader._initialized:
            # 获取模板目录的路径
            self._templates_dir = self._get_templates_dir()
            # 缓存已加载的模板
            self._template_cache: Dict[str, str] = {}
            PromptTemplateLoader._initialized = True
    
    def _get_templates_dir(self) -> Path:
        """
        获取模板目录的路径
        
        @return: 模板目录的Path对象
        """
        # 获取当前文件所在目录
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
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
    
    def load_all_templates(self, use_cache: bool = True) -> Dict[str, str]:
        """
        加载所有的模板文件
        
        @param use_cache: 是否使用缓存的模板
        @return: 包含所有模板的字典，键为模板名称，值为模板内容
        """
        # 如果使用缓存且缓存不为空，直接返回缓存
        if use_cache and self._template_cache:
            return self._template_cache.copy()
        
        templates = {}
        
        # 遍历templates目录中的所有txt文件
        for template_file in self._templates_dir.glob("*.txt"):
            # 使用文件名（不含扩展名）作为模板名称
            template_name = template_file.stem
            # 读取模板内容
            with open(template_file, "r", encoding="utf-8") as f:
                template_content = f.read()
            # 添加到模板字典
            templates[template_name] = template_content
            # 更新缓存
            self._template_cache[template_name] = template_content
        
        return templates
    
    def clear_cache(self) -> None:
        """
        清除模板缓存
        """
        self._template_cache.clear()
    
    def get_templates_dir(self) -> Path:
        """
        获取模板目录的路径
        
        @return: 模板目录的Path对象
        """
        return self._templates_dir


# 创建全局单例实例，方便直接访问
template_loader = PromptTemplateLoader()