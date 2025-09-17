from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class AgentEnvironment(ABC):
    """
    代理环境基类，定义了代理运行环境的基本接口
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态
        
        @return: 环境状态字典
        """
        pass
    
    @abstractmethod
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新环境状态
        
        @param updates: 要更新的状态字段
        @return: 更新后的环境状态
        """
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        重置环境状态
        
        @return: 重置后的环境状态
        """
        pass


class SimpleAgentEnvironment(AgentEnvironment):
    """
    简单的代理环境实现
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        初始化代理环境
        
        @param initial_state: 初始环境状态
        """
        self._state = initial_state or {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态
        
        @return: 环境状态字典的深拷贝
        """
        # 返回状态的深拷贝，避免外部直接修改内部状态
        return self._state.copy()
    
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新环境状态
        
        @param updates: 要更新的状态字段
        @return: 更新后的环境状态
        """
        if not isinstance(updates, dict):
            raise TypeError("updates必须是字典类型")
        
        # 更新状态
        self._state.update(updates)
        
        # 返回更新后的状态
        return self.get_state()
    
    def reset(self) -> Dict[str, Any]:
        """
        重置环境状态
        
        @return: 重置后的环境状态
        """
        self._state = {}
        return self.get_state()


class MemoryManager:
    """
    记忆管理器，用于存储和管理代理的历史交互和重要信息
    """
    
    def __init__(self, max_history_size: int = 10, max_compressed_size: int = 500):
        """
        初始化记忆管理器
        
        @param max_history_size: 历史记录的最大条目数
        @param max_compressed_size: 压缩后记忆的最大字符数
        """
        self.max_history_size = max_history_size
        self.max_compressed_size = max_compressed_size
        self.history = []
        self.knowledge = {}
        self.compressed_memory = ""
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加交互记录
        
        @param role: 角色（如user、agent）
        @param content: 交互内容
        @param metadata: 交互元数据
        """
        interaction = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        # 添加到历史记录
        self.history.append(interaction)
        
        # 如果历史记录超过最大大小，移除最早的记录
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        @param limit: 返回的最大记录数
        @return: 历史记录列表
        """
        if limit is None or limit >= len(self.history):
            return self.history.copy()
        
        # 返回最近的limit条记录
        return self.history[-limit:].copy()
    
    def add_knowledge(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加知识库条目
        
        @param key: 知识键
        @param value: 知识值
        @param metadata: 知识元数据
        """
        self.knowledge[key] = {
            "value": value,
            "metadata": metadata or {}
        }
    
    def get_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取知识库条目
        
        @param key: 知识键
        @return: 知识条目，包含值和元数据
        """
        return self.knowledge.get(key)
    
    def reset(self) -> None:
        """
        重置记忆管理器
        """
        self.history = []
        self.knowledge = {}
        self.compressed_memory = ""
        
    def compress_memory(self, model) -> None:
        """
        使用语言模型压缩历史记忆，保留关键信息
        
        @param model: 用于压缩的语言模型实例
        """
        if not self.history:
            return
        
        # 准备压缩提示
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in self.history])
        prompt = f"请将以下对话历史压缩为简洁的摘要，保留最重要的信息和关键点，不要超过{self.max_compressed_size}字符：\n{history_text}"
        
        # 使用模型进行压缩
        try:
            self.compressed_memory = model.generate(prompt)
            if len(self.compressed_memory) > self.max_compressed_size:
                # 如果压缩结果仍过长，截断
                self.compressed_memory = self.compressed_memory[:self.max_compressed_size] + "..."
        except Exception as e:
            # 如果压缩失败，使用默认压缩方式
            self.compressed_memory = f"[{len(self.history)}条对话历史，包含用户查询和代理回答]"
    
    def get_context_with_memory(self, current_context: str) -> str:
        """
        获取包含压缩记忆的完整上下文
        
        @param current_context: 当前上下文
        @return: 包含压缩记忆的完整上下文
        """
        if self.compressed_memory:
            return f"[历史交互摘要]\n{self.compressed_memory}\n\n[当前信息]\n{current_context}"
        return current_context
    
    def update_with_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        将工具结果更新到记忆中
        
        @param tool_results: 工具调用结果列表
        """
        if not tool_results:
            return
        
        # 提取成功的工具结果
        success_results = []
        for result in tool_results:
            if result.get("status") == "success":
                success_results.append({
                    "tool": result["tool"],
                    "result": result["result"],
                    "timestamp": result.get("metadata", {}).get("timestamp", "")
                })
        
        if success_results:
            # 将工具结果添加到知识库
            self.add_knowledge(
                "last_tool_results",
                success_results,
                {"type": "tool_results", "timestamp": "now"}
            )