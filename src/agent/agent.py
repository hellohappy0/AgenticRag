from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from src.tool import BaseTool
from src.doc_process import DocumentProcessor
from src.agent_context import AgentEnvironment, MemoryManager
from src.prompt import PromptManager
from src.model import BaseLanguageModel, ModelResponseParser


class BaseAgent(ABC):
    """
    代理基类，定义了代理的基本接口
    """
    
    @abstractmethod
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行代理处理查询
        
        @param query: 用户查询
        @param kwargs: 其他参数
        @return: 代理处理结果
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        重置代理状态
        """
        pass


class AgenticRAG(BaseAgent):
    """
    基于代理的检索增强生成系统实现
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        document_processor: DocumentProcessor,
        environment: AgentEnvironment,
        prompt_manager: PromptManager,
        memory_manager: Optional[MemoryManager] = None,
        max_retries: int = 3,
        max_tool_calls: int = 5
    ):
        """
        初始化Agentic RAG系统
        
        @param model: 语言模型
        @param tools: 可用工具列表
        @param document_processor: 文档处理器
        @param environment: 代理环境
        @param prompt_manager: 提示管理器
        @param memory_manager: 记忆管理器，默认为None
        @param max_retries: 最大重试次数
        @param max_tool_calls: 最大工具调用次数
        """
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.document_processor = document_processor
        self.environment = environment
        self.prompt_manager = prompt_manager
        self.memory_manager = memory_manager or MemoryManager()
        self.max_retries = max_retries
        self.max_tool_calls = max_tool_calls
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行Agentic RAG处理查询
        
        @param query: 用户查询
        @param kwargs: 其他参数
        @return: 包含答案和相关信息的字典
        """
        # 记录用户查询
        self.memory_manager.add_interaction("user", query)
        
        # 初始化状态
        state = {
            "query": query,
            "context": "",
            "answer": "",
            "tool_calls": [],
            "retries": 0,
            "status": "processing"
        }
        
        self.environment.update_state(state)
        
        try:
            # 主循环：使用代理逻辑处理查询
            state = self._agent_loop(state)
            
            # 校验答案
            if state["status"] == "success":
                state = self._validate_answer(state)
            
            # 更新记忆
            if state.get("answer"):
                self.memory_manager.add_interaction("agent", state["answer"])
            
        except Exception as e:
            state["status"] = "error"
            state["error"] = str(e)
        
        return state
    
    def _agent_loop(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        代理主循环逻辑
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        tool_call_count = 0
        
        while tool_call_count < self.max_tool_calls:
            # 生成提示
            prompt = self._generate_prompt(state)
            
            # 使用模型生成响应
            model_response = self.model.generate_with_tools(
                prompt=prompt,
                tools=[tool.get_tool_info() for tool in self.tools.values()]
            )
            
            # 解析模型响应
            tool_calls = ModelResponseParser.parse_tool_calls(model_response["response"])
            
            # 如果没有工具调用，说明模型直接回答了问题
            if not tool_calls:
                state["answer"] = model_response["response"]
                state["status"] = "success"
                break
            
            # 执行工具调用
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("parameters", {})
                
                if tool_name not in self.tools:
                    tool_results.append({
                        "status": "error",
                        "tool": tool_name,
                        "message": f"工具 '{tool_name}' 不存在"
                    })
                    continue
                
                try:
                    # 执行工具
                    result = self.tools[tool_name].run(**tool_params)
                    tool_results.append({
                        "status": "success",
                        "tool": tool_name,
                        "result": result
                    })
                    
                    # 更新上下文
                    if tool_name == "retrieve_documents" and result.get("status") == "success":
                        state["context"] += self._format_tool_results(tool_results)
                    
                except Exception as e:
                    tool_results.append({
                        "status": "error",
                        "tool": tool_name,
                        "message": str(e)
                    })
                
                tool_call_count += 1
                
                # 检查是否达到最大工具调用次数
                if tool_call_count >= self.max_tool_calls:
                    break
            
            # 更新状态
            state["tool_calls"].append({
                "calls": tool_calls,
                "results": tool_results
            })
            
            # 如果有上下文信息，生成最终答案
            if state["context"]:
                answer_prompt = self.prompt_manager.generate_prompt(
                    "rag_answer",
                    query=state["query"],
                    context=state["context"]
                )
                state["answer"] = self.model.generate(answer_prompt)
                state["status"] = "success"
                break
        
        # 如果达到最大工具调用次数还没有答案，尝试直接生成
        if state["status"] == "processing":
            fallback_prompt = self.prompt_manager.generate_prompt(
                "rag_answer",
                query=state["query"],
                context=state["context"] or "没有找到相关信息"
            )
            state["answer"] = self.model.generate(fallback_prompt)
            state["status"] = "success"
        
        return state
    
    def _generate_prompt(self, state: Dict[str, Any]) -> str:
        """
        生成代理提示
        
        @param state: 当前状态
        @return: 生成的提示字符串
        """
        # 获取工具信息
        tools_info = []
        for tool_name, tool in self.tools.items():
            tool_info = tool.get_tool_info()
            tools_info.append(f"{tool_name}: {tool_info['description']}")
        
        tools_str = "\n".join(tools_info)
        
        # 生成提示
        return self.prompt_manager.generate_prompt(
            "main",
            tools=tools_str,
            query=state["query"],
            context=state["context"]
        )
    
    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        格式化工具调用结果
        
        @param tool_results: 工具调用结果列表
        @return: 格式化后的字符串
        """
        formatted_results = []
        
        for result in tool_results:
            if result["status"] == "success":
                tool_data = result["result"]
                if result["tool"] == "retrieve_documents" and "results" in tool_data:
                    for doc in tool_data["results"]:
                        formatted_results.append(f"文档内容: {doc.get('content', '')}")
            
        return "\n\n".join(formatted_results)
    
    def _validate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验生成的答案
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        if not state.get("answer"):
            state["status"] = "error"
            state["error"] = "没有生成答案"
            return state
        
        # 使用自我批评提示让模型评估自己的答案
        critique_prompt = self.prompt_manager.generate_prompt(
            "self_critique",
            query=state["query"],
            answer=state["answer"],
            context=state["context"]
        )
        
        critique = self.model.generate(critique_prompt)
        state["critique"] = critique
        
        # 如果模型认为需要改进，尝试重新生成
        if "需要改进" in critique or "不准确" in critique or "不完整" in critique:
            state["retries"] += 1
            
            if state["retries"] < self.max_retries:
                # 生成反思提示
                reflection_prompt = self.prompt_manager.generate_prompt(
                    "reflection",
                    query=state["query"],
                    thought_process="根据已获取的信息进行分析思考"
                )
                
                reflection = self.model.generate(reflection_prompt)
                state["reflection"] = reflection
                
                # 重新进入代理循环
                return self._agent_loop(state)
        
        return state
    
    def reset(self) -> None:
        """
        重置代理状态
        """
        self.environment.reset()
        self.memory_manager.reset()