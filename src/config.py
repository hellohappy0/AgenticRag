import os
import json
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """
    配置管理器，支持从环境变量和配置文件加载配置，环境变量优先级高于配置文件
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化配置管理器
        
        @param config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """
        从配置文件和环境变量加载配置
        """
        # 首先从配置文件加载配置
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    # 支持yaml和json格式
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        self.config = yaml.safe_load(f) or {}
                    elif self.config_file.endswith('.json'):
                        self.config = json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {str(e)}")
        
        # 然后从环境变量覆盖配置
        # 遍历配置中的所有键，检查是否有对应的环境变量
        self._override_with_env_vars()
    
    def _override_with_env_vars(self) -> None:
        """
        使用环境变量覆盖配置文件中的值
        环境变量格式: AGENTIC_RAG_{配置键}_...，使用下划线分隔嵌套键
        """
        prefix = "AGENTIC_RAG_"
        
        # 遍历所有环境变量
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀
                config_key = key[len(prefix):].lower()
                # 将下划线分隔的键转换为嵌套结构
                self._set_nested_config(config_key, value)
    
    def _set_nested_config(self, flat_key: str, value: str) -> None:
        """
        将扁平化的键值对设置到嵌套的配置字典中
        
        @param flat_key: 扁平化的键，如 "model.tongyi.api_key"
        @param value: 配置值
        """
        keys = flat_key.split('_')
        current = self.config
        
        # 遍历除最后一个键外的所有键，确保嵌套结构存在
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # 设置最后一个键的值，尝试转换类型
        final_key = keys[-1]
        current[final_key] = self._convert_value_type(value)
    
    def _convert_value_type(self, value: str) -> Any:
        """
        将字符串值转换为适当的Python类型
        
        @param value: 字符串值
        @return: 转换后的Python对象
        """
        # 尝试转换为布尔值
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # 尝试转换为数字
        try:
            # 先尝试转换为整数
            return int(value)
        except ValueError:
            try:
                # 再尝试转换为浮点数
                return float(value)
            except ValueError:
                pass
        
        # 尝试转换为列表或字典
        if value.startswith('[') and value.endswith(']') or value.startswith('{') and value.endswith('}'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 如果以上都失败，返回原始字符串
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        @param key: 配置键，可以是嵌套键，如 "model.tongyi.api_key"
        @param default: 默认值
        @return: 配置值或默认值
        """
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        @param key: 配置键，可以是嵌套键
        @param value: 配置值
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self) -> None:
        """
        保存配置到配置文件
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
                elif self.config_file.endswith('.json'):
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
    
    def reload(self) -> None:
        """
        重新加载配置
        """
        self.config = {}
        self._load_config()


# 创建全局配置管理器实例
config_manager = ConfigManager()


def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置的便捷函数
    
    @param key: 配置键
    @param default: 默认值
    @return: 配置值或默认值
    """
    return config_manager.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    设置配置的便捷函数
    
    @param key: 配置键
    @param value: 配置值
    """
    config_manager.set(key, value)


def save_config() -> None:
    """
    保存配置的便捷函数
    """
    config_manager.save()


def reload_config() -> None:
    """
    重新加载配置的便捷函数
    """
    config_manager.reload()