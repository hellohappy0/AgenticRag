from typing import Dict, Any, List, Optional
import datetime
import json

# 导入smolagents相关类
from smolagents import Model
from smolagents.memory import AgentMemory, TaskStep, ActionStep
from smolagents.monitoring import Timing
from smolagents.models import ChatMessage, MessageRole


class SmolAgentMemoryManager:
    """
    基于smolAgent框架memory能力的记忆管理器实现
    采用smolagents的记忆管理思想，使用AgentMemory作为核心存储结构
    """
    
    def __init__(self, max_history_size: int = 10, max_compressed_size: int = 500):
        """
        初始化记忆管理器
        
        @param max_history_size: 历史记录的最大条目数
        @param max_compressed_size: 压缩后记忆的最大字符数
        """
        self.max_history_size = max_history_size
        self.max_compressed_size = max_compressed_size
        
        # 使用smolAgent的AgentMemory类作为核心存储结构
        self.memory = AgentMemory(system_prompt="Agent memory management system")
        
        # 存储压缩后的记忆
        self.compressed_memory = ""
        
        # 用于跟踪步骤编号
        self.current_step = 0
        
        # 用于快速查找知识库条目
        self._knowledge_index = {}
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加交互记录到AgentMemory中，使用ActionStep存储交互信息
        
        @param role: 角色（如user、agent）
        @param content: 交互内容
        @param metadata: 交互元数据
        """
        # 确保步骤数量不超过最大历史记录大小
        self._ensure_history_limit()
        
        # 创建一个ActionStep来存储交互信息到AgentMemory
        self.current_step += 1
        
        # 创建消息对象 - 如果有元数据，将其添加到消息内容中
        message_content = [{"type": "text", "text": content}]
        if metadata:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            message_content.append({"type": "text", "text": f"[METADATA]: {metadata_str}"})
            
        chat_message = ChatMessage(role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT, content=message_content)
        
        # 创建时间对象和ActionStep
        current_time = datetime.datetime.now().timestamp()
        action_step = ActionStep(
            step_number=self.current_step,
            timing=Timing(start_time=current_time, end_time=current_time),
            model_output_message=chat_message,
            model_output=content,
            observations=f"[ROLE]: {role}"
        )
        self.memory.steps.append(action_step)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取历史交互记录，从AgentMemory中直接提取
        
        @param limit: 返回的最大记录数
        @return: 历史记录列表
        """
        interactions = []
        
        # 遍历所有步骤查找交互记录
        for step in self.memory.steps:
            if isinstance(step, ActionStep) and step.model_output_message:
                # 从模型输出消息中提取角色
                role = "unknown"
                if hasattr(step.model_output_message, 'role'):
                    if step.model_output_message.role == MessageRole.USER:
                        role = "user"
                    elif step.model_output_message.role == MessageRole.ASSISTANT:
                        role = "agent"
                    elif step.model_output_message.role == MessageRole.TOOL_RESPONSE:
                        role = "tool"
                
                # 从observations中提取角色信息（如果有）
                if step.observations and step.observations.startswith("[ROLE]: "):
                    role = step.observations[8:].strip()
                
                # 提取内容和元数据
                content = step.model_output or ""
                metadata = {}
                
                if hasattr(step.model_output_message, 'content'):
                    for item in step.model_output_message.content:
                        if isinstance(item, dict) and item.get("type") == "text" and item.get("text", "").startswith("[METADATA]: "):
                            try:
                                metadata_text = item["text"][len("[METADATA]: "):]
                                metadata = json.loads(metadata_text)
                            except:
                                pass
                            break
                
                interactions.append({"role": role, "content": content, "metadata": metadata})
        
        # 应用限制
        if limit is not None and limit < len(interactions):
            return interactions[-limit:]
        
        return interactions
    
    def add_knowledge(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加知识库条目到AgentMemory，使用TaskStep存储知识
        
        @param key: 知识键
        @param value: 知识值
        @param metadata: 知识元数据
        """
        self._ensure_history_limit()
        
        # 增加步骤计数并创建知识字符串
        self.current_step += 1
        knowledge_str = f"KNOWLEDGE [{key}]: "
        
        # 根据值的类型转换为字符串格式
        if isinstance(value, str):
            knowledge_str += value[:200] + ("..." if len(value) > 200 else "")
        elif isinstance(value, (list, dict)):
            try:
                knowledge_str += json.dumps(value, ensure_ascii=False)[:200] + "..."
            except:
                knowledge_str += str(value)[:200] + "..."
        else:
            knowledge_str += str(value)[:200] + "..."
        
        # 创建TaskStep并添加到memory
        task_step = TaskStep(task=knowledge_str)
        self.memory.steps.append(task_step)
        
        # 在索引中存储完整信息
        self._knowledge_index[key] = {
            "step": task_step,
            "value": str(value),
            "metadata": metadata or {}
        }
    
    def get_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取知识库条目，从索引中直接检索
        
        @param key: 知识键
        @return: 知识条目，包含值和元数据
        """
        if key in self._knowledge_index:
            knowledge_entry = self._knowledge_index[key]
            return {
                "value": knowledge_entry.get("value", ""),
                "metadata": knowledge_entry.get("metadata", {})
            }
        return None
    
    def reset(self) -> None:
        """
        重置记忆管理器，清空所有步骤但保留system prompt
        """
        # 重置smolAgent的AgentMemory
        self.memory.reset()
        
        # 重置本地存储
        self.compressed_memory = ""
        self.current_step = 0
        self._knowledge_index = {}
        
    def compress_memory(self, model) -> None:
        """
        使用语言模型压缩历史记忆，保留关键信息
        
        @param model: 用于压缩的语言模型实例
        """
        # 从AgentMemory中直接访问steps
        steps = self.memory.steps
        
        # 如果没有步骤，直接返回
        if not steps:
            self.compressed_memory = ""
            return
        
        # 构建记忆内容
        memory_content = ""
        for step in steps:
            if isinstance(step, TaskStep) and step.task:
                # 查找对应的知识库键
                knowledge_key = "unknown"
                for key, entry in self._knowledge_index.items():
                    if entry.get("step") == step:
                        knowledge_key = key
                        break
                memory_content += f"[知识 {knowledge_key}]: {step.task}\n"
            elif isinstance(step, ActionStep) and step.model_output:
                # 提取角色信息
                role = "unknown"
                if hasattr(step.model_output_message, 'role'):
                    role = "user" if step.model_output_message.role == MessageRole.USER else "agent" if step.model_output_message.role == MessageRole.ASSISTANT else "unknown"
                elif step.observations and step.observations.startswith("[ROLE]: "):
                    role = step.observations[8:].strip()
                memory_content += f"[{role}]: {step.model_output}\n"
        
        prompt = f"请将以下对话历史压缩为简洁的摘要，保留最重要的信息和关键点，不要超过{self.max_compressed_size}字符：\n{memory_content}"
        
        # 使用模型进行压缩
        try:
            if hasattr(model, 'generate'):
                # 兼容项目中的模型接口
                self.compressed_memory = model.generate(prompt)
            elif isinstance(model, Model):
                # 直接使用smolAgent的Model接口
                self.compressed_memory = str(model(prompt))
            else:
                raise TypeError("model must implement generate() method or be a smolAgent Model")
            
            if len(self.compressed_memory) > self.max_compressed_size:
                # 如果压缩结果仍过长，截断
                self.compressed_memory = self.compressed_memory[:self.max_compressed_size] + "..."
        except Exception as e:
            # 如果压缩失败，使用默认压缩方式
            self.compressed_memory = f"[{len(steps)}条历史记录，包含用户查询、代理回答和知识库信息]"
    
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
        将工具结果更新到记忆中，使用ActionStep存储工具结果
        
        @param tool_results: 工具调用结果列表
        """
        if not tool_results:
            return
        
        self._ensure_history_limit()
        
        # 提取成功的工具结果
        success_results = []
        current_time = datetime.datetime.now()
        
        for result in tool_results:
            if result.get("status") == "success":
                # 保存成功结果
                success_results.append({
                    "tool": result["tool"],
                    "result": result["result"],
                    "timestamp": result.get("metadata", {}).get("timestamp", str(current_time))
                })
                
                # 创建ActionStep存储工具结果
                self.current_step += 1
                time_stamp = current_time.timestamp()
                
                # 构建工具结果消息
                tool_result_content = f"TOOL RESULT [{result['tool']}]: {str(result['result'])[:200]}..."
                message_content = [{"type": "text", "text": tool_result_content}]
                chat_message = ChatMessage(role=MessageRole.TOOL_RESPONSE, content=message_content)
                
                # 创建并添加ActionStep
                action_step = ActionStep(
                    step_number=self.current_step,
                    timing=Timing(start_time=time_stamp, end_time=time_stamp),
                    model_output_message=chat_message,
                    model_output=tool_result_content,
                    observations=f"TOOL_RESULT: {result['tool']}",
                    action_output=result['result']
                )
                self.memory.steps.append(action_step)
        
        # 将成功的工具结果添加到知识库
        if success_results:
            self.add_knowledge(
                "last_tool_results",
                success_results,
                {"type": "tool_results", "timestamp": str(current_time)}
            )
    
    def _ensure_history_limit(self) -> None:
        """
        确保历史记录不超过最大限制
        """
        if len(self.memory.steps) >= self.max_history_size:
            # 保留最近的步骤并清空索引（因为无法从steps重建完整索引）
            self.memory.steps = self.memory.steps[-self.max_history_size+1:]
            self._knowledge_index = {}
                    
    def get_succinct_steps(self) -> List[Dict[str, Any]]:
        """
        获取简洁的步骤列表，用于在工具调用之间传递
        
        @return: 简洁的步骤列表
        """
        # 只返回ActionStep类型的步骤，提取关键信息
        return [
            {
                "step_number": step.step_number,
                "observations": step.observations,
                "action_output": step.action_output
            }
            for step in self.memory.steps
            if step.__class__.__name__ == "ActionStep"
        ]
    
    def get_full_steps(self) -> List[Dict[str, Any]]:
        """
        获取完整的步骤列表
        
        @return: 完整的步骤列表
        """
        # 返回所有步骤的完整信息
        return [
            {
                "step_number": step.step_number,
                "type": step.__class__.__name__,
                "observations": getattr(step, "observations", None),
                "action_output": getattr(step, "action_output", None),
                "prompt": getattr(step, "prompt", None),
                "model_output": getattr(step, "model_output", None),
                "model_output_message": getattr(step, "model_output_message", None),
                "timing": getattr(step, "timing", None)
            }
            for step in self.memory.steps
        ]
    
    def add_final_answer(self, answer: str) -> None:
        """
        添加最终回答到记忆中
        
        @param answer: 最终回答
        """
        self._ensure_history_limit()
        
        self.current_step += 1
        current_time = datetime.datetime.now().timestamp()
        
        # 构建最终回答消息
        message_content = [{"type": "text", "text": answer}]
        chat_message = ChatMessage(role=MessageRole.ASSISTANT, content=message_content)
        
        # 创建并添加包含最终回答的ActionStep
        action_step = ActionStep(
            step_number=self.current_step,
            timing=Timing(start_time=current_time, end_time=current_time),
            model_output_message=chat_message,
            model_output=answer,
            observations="FINAL_ANSWER",
            action_output=answer
        )
        self.memory.steps.append(action_step)