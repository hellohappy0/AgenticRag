from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import json
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
            "original_query": query,  # 保存原始查询
            "context": "",
            "answer": "",
            "tool_calls": [],
            "retries": 0,
            "iterations": 0,
            "max_iterations": kwargs.get("max_iterations", 3),
            "status": "processing",
            "query_analysis": {},
            "tool_usage_history": []  # 记录工具使用历史
        }
        
        self.environment.update_state(state)
        
        try:
            # 1. 分析和处理查询
            state = self._analyze_query(state)
            
            # 2. 根据分析结果执行查询处理
            state = self._process_query_based_on_analysis(state)
            
            # 3. 主循环：使用代理逻辑处理查询
            state = self._agent_loop(state)
            
            # 4. 评估答案并进行迭代优化
            while state["status"] == "success" and state["iterations"] < state["max_iterations"]:
                state = self._evaluate_answer(state)
                
                if state.get("need_iteration", False):
                    state = self._optimize_answer(state)
                    state = self._agent_loop(state)
                    state["iterations"] += 1
                else:
                    break
            
            # 5. 更新记忆
            if state.get("answer"):
                # 记录代理回答
                self.memory_manager.add_interaction(
                    "agent", 
                    state["answer"],
                    {"evaluation": state.get("evaluation", {}), "iterations": state["iterations"]}
                )
                
                # 如果有评估结果，也添加到记忆中
                if state.get("evaluation"):
                    self.memory_manager.add_knowledge(
                        f"evaluation_iteration_{state['iterations']}",
                        state["evaluation"],
                        {"type": "evaluation", "iteration": state["iterations"]}
                    )
            
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
            print("="*60)
            print(f"【迭代步骤 {state['iterations'] + 1}】开始处理")
            print("="*60)
            
            # 更新记忆中的工具结果
            if state.get('tool_calls'):
                last_tool_calls = state['tool_calls'][-1]
                self.memory_manager.update_with_tool_results(last_tool_calls.get('results', []))
            
            # 压缩记忆并生成包含历史信息的提示
            self.memory_manager.compress_memory(self.model)
            augmented_context = self.memory_manager.get_context_with_memory(state['context'])
            
            # 记录工具使用历史信息，特别是连续使用相同工具的情况
            tool_usage_history_str = """
工具使用历史：
"""
            if state.get('tool_usage_history'):
                for i, usage in enumerate(state['tool_usage_history']):
                    tool_usage_history_str += f"{i+1}. 使用工具: {usage['tool']}, 查询: {usage['query']}, 结果质量: {'相关' if usage['result_quality'] > 0.5 else '不相关'}\n"
            else:
                tool_usage_history_str += "无"
            
            # 生成提示（使用增强后的上下文和工具使用历史）
            prompt = self._generate_prompt(state, augmented_context, tool_usage_history_str)
            print("\n[决策过程] 生成包含历史记忆和工具使用记录的代理提示，准备调用模型...")
            
            # 使用模型生成响应
            model_response = self.model.generate_with_tools(
                prompt=prompt,
                tools=[tool.get_tool_info() for tool in self.tools.values()]
            )
            
            # 解析模型响应
            tool_calls = ModelResponseParser.parse_tool_calls(model_response["response"])
            
            # 如果没有工具调用，说明模型直接回答了问题
            if not tool_calls:
                print("\n[决策结果] 未使用工具，模型直接生成回答")
                state["answer"] = model_response["response"]
                state["status"] = "success"
                break
            
            # 执行工具调用
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("parameters", {})
                
                print(f"\n[工具决策] 选择工具: {tool_name}")
                print(f"[工具参数] {json.dumps(tool_params, ensure_ascii=False, indent=2)}")
                
                if tool_name not in self.tools:
                    tool_results.append({
                        "status": "error",
                        "tool": tool_name,
                        "message": f"工具 '{tool_name}' 不存在"
                    })
                    print(f"[工具错误] 工具 '{tool_name}' 不存在")
                    continue
                
                try:
                    # 执行工具
                    print("[工具执行] 正在调用工具...")
                    result = self.tools[tool_name].run(**tool_params)
                    tool_results.append({
                        "status": "success",
                        "tool": tool_name,
                        "result": result
                    })
                    print(f"[工具结果] 工具 '{tool_name}' 调用成功")
                    
                except Exception as e:
                    tool_results.append({
                        "status": "error",
                        "tool": tool_name,
                        "message": str(e)
                    })
                    print(f"[工具错误] 工具 '{tool_name}' 调用失败: {str(e)}")
                
                tool_call_count += 1
                
                # 检查是否达到最大工具调用次数
                if tool_call_count >= self.max_tool_calls:
                    break
            
            # 更新状态
            state["tool_calls"].append({
                "calls": tool_calls,
                "results": tool_results
            })
            
            # 记录工具使用历史
            for i, result in enumerate(tool_results):
                if result["status"] == "success" and result["tool"] == "retrieve_documents":
                    tool_call = tool_calls[i] if i < len(tool_calls) else {}
                    # 评估检索结果质量（简单判断：相似度分数大于0.5视为相关）
                    results = result.get("result", {}).get("results", [])
                    avg_score = sum(item.get("score", 0) for item in results) / len(results) if results else 0
                    
                    # 记录工具使用历史
                    state["tool_usage_history"].append({
                        "tool": "retrieve_documents",
                        "query": tool_call.get("parameters", {}).get("query", ""),
                        "result_quality": avg_score  # 0-1之间的分数，表示结果质量
                    })
                    
                    # 保持历史记录不超过3条
                    if len(state["tool_usage_history"]) > 3:
                        state["tool_usage_history"].pop(0)
            
            # 详细打印工具调用结果
            print(f"\n{'='*40}")
            print(f"【工具调用详情】")
            for i, result in enumerate(tool_results):
                print(f"\n工具 {i+1}: {result['tool']}")
                print(f"状态: {result['status']}")
                if result['status'] == 'success':
                    print(f"返回结果: {json.dumps(result['result'], ensure_ascii=False, indent=2)}")
                else:
                    print(f"错误信息: {result['message']}")
            print(f"{'='*40}")
            
            # 更新上下文 - 在所有工具调用完成后更新一次上下文
            for result in tool_results:
                if result["status"] == "success":
                    formatted_result = self._format_tool_results([result])
                    print(f"\n[上下文更新] 格式化后的工具结果已添加到上下文")
                    state["context"] += formatted_result + "\n\n"
            
            # 如果有上下文信息，生成最终答案
            if state["context"]:
                print("\n[生成答案] 基于上下文信息生成回答...")
                answer_prompt = self.prompt_manager.generate_prompt(
                    "rag_answer",
                    query=state["query"],
                    context=state["context"]
                )
                state["answer"] = self.model.generate(answer_prompt)
                state["status"] = "success"
                print(f"\n[回答生成] 已生成初步回答")
                break
        
        # 如果达到最大工具调用次数还没有答案，尝试直接生成
        if state["status"] == "processing":
            print("\n[回退机制] 达到最大工具调用次数，尝试直接生成回答...")
            fallback_prompt = self.prompt_manager.generate_prompt(
                "rag_answer",
                query=state["query"],
                context=state["context"] or "没有找到相关信息"
            )
            state["answer"] = self.model.generate(fallback_prompt)
            state["status"] = "success"
        
        return state
    
    def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析用户查询，决定是否需要拆解或改写
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        # 生成查询分析提示
        query_analysis_prompt = self.prompt_manager.generate_prompt(
            "query_rewrite",
            query=state["query"]
        )
        
        # 获取模型对查询的分析
        analysis_response = self.model.generate(query_analysis_prompt)
        
        # 解析分析结果
        query_analysis = {
            "original_query": state["query"],
            "analysis_result": analysis_response
        }
        
        # 检查是否需要改写或拆解
        if "改写后的查询: " in analysis_response:
            # 提取改写后的查询
            rewritten_query = analysis_response.split("改写后的查询: ")[-1].strip()
            state["query"] = rewritten_query
            query_analysis["rewritten_query"] = rewritten_query
            query_analysis["action"] = "rewrite"
        elif "拆解后的子问题:" in analysis_response:
            # 提取子问题
            sub_questions_part = analysis_response.split("拆解后的子问题:")[-1].strip()
            sub_questions = []
            for line in sub_questions_part.split("\n"):
                if line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3."):
                    sub_questions.append(line.strip().split(".", 1)[1].strip())
            
            if sub_questions:
                query_analysis["sub_questions"] = sub_questions
                query_analysis["action"] = "decompose"
        else:
            # 不需要改写或拆解
            query_analysis["action"] = "none"
        
        state["query_analysis"] = query_analysis
        return state
    
    def _process_query_based_on_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据查询分析结果处理查询
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        query_analysis = state["query_analysis"]
        
        # 如果查询被拆解为子问题，依次处理每个子问题
        if query_analysis.get("action") == "decompose" and "sub_questions" in query_analysis:
            sub_questions = query_analysis["sub_questions"]
            state["sub_questions_results"] = []
            
            for i, sub_question in enumerate(sub_questions):
                # 创建子问题状态
                sub_state = {
                    "query": sub_question,
                    "original_query": state["original_query"],
                    "context": state["context"],
                    "answer": "",
                    "tool_calls": [],
                    "retries": 0,
                    "status": "processing"
                }
                
                # 处理子问题
                sub_state = self._agent_loop(sub_state)
                
                # 收集子问题结果
                state["sub_questions_results"].append({
                    "question": sub_question,
                    "answer": sub_state.get("answer", ""),
                    "context": sub_state.get("context", ""),
                    "status": sub_state.get("status", "error")
                })
                
                # 将子问题的上下文添加到主上下文
                state["context"] += "\n\n" + sub_state.get("context", "")
            
            # 使用子问题结果生成最终答案
            if state["sub_questions_results"]:
                sub_answers = [f"{res['question']}\n{res['answer']}" for res in state["sub_questions_results"] if res['status'] == 'success']
                combined_answers = "\n\n".join(sub_answers)
                
                final_answer_prompt = self.prompt_manager.generate_prompt(
                    "rag_answer",
                    query=state["original_query"],
                    context=f"子问题回答:\n{combined_answers}\n\n原始上下文:{state['context']}"
                )
                
                state["answer"] = self.model.generate(final_answer_prompt)
                state["status"] = "success"
        
        return state
    
    def _generate_prompt(self, state: Dict[str, Any], augmented_context: str = None, tool_usage_history_str: str = None) -> str:
        """
        生成代理提示
        
        @param state: 当前状态
        @param augmented_context: 包含历史记忆的增强上下文
        @param tool_usage_history_str: 工具使用历史字符串
        @return: 生成的提示字符串
        """
        # 获取查询分析信息
        query_analysis = state.get("query_analysis", {})
        
        # 获取最后一次评估结果
        last_evaluation = state.get("evaluation", {})
        
        # 获取优化计划
        optimization_plan = state.get("optimization_plan", {})
        
        # 确定要使用的上下文
        context_to_use = augmented_context if augmented_context else state["context"]
        
        # 添加工具使用历史到上下文
        if tool_usage_history_str:
            context_to_use = f"{tool_usage_history_str}\n\n{context_to_use}"
        
        # 准备工具信息
        tools_info = []
        for tool_name, tool in self.tools.items():
            tool_info = tool.get_tool_info()
            tools_info.append(f"{tool_name}: {tool_info['description']}")
        
        tools_str = "\n".join(tools_info)
        
        # 生成提示
        prompt = self.prompt_manager.generate_prompt(
            "main",
            query=state["query"],
            context=context_to_use,
            tools=tools_str,
            last_evaluation=json.dumps(last_evaluation, ensure_ascii=False) if last_evaluation else "无",
            optimization_plan=json.dumps(optimization_plan, ensure_ascii=False) if optimization_plan else "无"
        )
        
        return prompt
    
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
                elif result["tool"] == "web_search" and "results" in tool_data:
                    for idx, search_result in enumerate(tool_data["results"], 1):
                        title = search_result.get("title", "无标题")
                        snippet = search_result.get("snippet", "")
                        formatted_results.append(f"搜索结果 {idx}:\n标题: {title}\n摘要: {snippet}")
            
        return "\n\n".join(formatted_results)
    
    def _validate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验生成的答案（简化版，保留向后兼容性）
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        return self._evaluate_answer(state)
    
    def _evaluate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估生成的答案，从多个维度进行全面评估
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        if not state.get("answer"):
            state["status"] = "error"
            state["error"] = "没有生成答案"
            return state
        
        print("\n" + "="*60)
        print(f"【回答评估】第{state['iterations'] + 1}轮迭代评估答案质量")
        print("="*60)
        
        # 生成答案评估提示
        evaluation_prompt = self.prompt_manager.generate_prompt(
            "answer_evaluation",
            query=state["query"],
            answer=state["answer"],
            context=state["context"]
        )
        
        # 获取模型的评估结果
        evaluation_response = self.model.generate(evaluation_prompt)
        
        # 解析评估结果
        try:
            # 尝试提取JSON部分
            if "{" in evaluation_response and "}" in evaluation_response:
                start_idx = evaluation_response.index("{")
                end_idx = evaluation_response.rindex("}") + 1
                evaluation_json = evaluation_response[start_idx:end_idx]
                evaluation = json.loads(evaluation_json)
            else:
                # 如果不是有效的JSON，创建基本评估
                evaluation = {
                    "overall_rating": "一般",
                    "suggestions": "无法解析详细评估结果",
                    "need_iteration": "否"
                }
        except Exception as e:
            print(f"解析评估结果失败: {str(e)}")
            evaluation = {
                "overall_rating": "一般",
                "suggestions": f"评估结果解析错误: {str(e)}",
                "need_iteration": "否"
            }
        
        state["evaluation"] = evaluation
        
        # 打印详细评估结果
        print("\n[评估结果详情]")
        print(f"整体评分: {evaluation.get('overall_rating', '未评分')}")
        print(f"完备性评分: {evaluation.get('completeness_score', '未评分')}")
        print(f"准确性评分: {evaluation.get('accuracy_score', '未评分')}")
        print(f"相关性评分: {evaluation.get('relevance_score', '未评分')}")
        print(f"建议: {evaluation.get('suggestions', '无建议')}")
        
        # 确定是否需要迭代
        state["need_iteration"] = evaluation.get("need_iteration", "否") == "是" or \
                                 evaluation.get("overall_rating", "") in ["较差", "差"]
        
        print(f"\n[迭代决策] 是否需要继续迭代: {'是' if state['need_iteration'] else '否'}")
        print("="*60)
        
        return state
    
    def _optimize_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据评估结果优化答案
        
        @param state: 当前状态
        @return: 更新后的状态
        """
        if "evaluation" not in state:
            return state
        
        print("\n" + "="*60)
        print(f"【迭代优化】第{state['iterations'] + 1}轮迭代根据评估结果优化回答")
        print("="*60)
        
        # 生成自我批评提示
        self_critique_prompt = self.prompt_manager.generate_prompt(
            "self_critique",
            answer=state["answer"],
            query=state["query"],
            context=state["context"]
        )
        self_critique = self.model.generate(self_critique_prompt)
        print(f"\n[自我批评结果]")
        print(self_critique)
        
        # 生成反思提示
        reflection_prompt = self.prompt_manager.generate_prompt(
            "reflection",
            query=state["query"],
            thought_process=f"评估结果: {json.dumps(state['evaluation'], ensure_ascii=False)}\n自我批评: {self_critique}"
        )
        reflection = self.model.generate(reflection_prompt)
        print(f"\n[反思结果]")
        print(reflection)
        
        # 生成迭代优化提示，整合自我批评和反思结果
        optimization_prompt = self.prompt_manager.generate_prompt(
            "iteration_optimization",
            query=state["query"],
            answer=state["answer"],
            context=state["context"],
            evaluation=json.dumps(state["evaluation"], ensure_ascii=False),
            self_critique=self_critique,
            reflection=reflection
        )
        
        # 获取模型的优化建议
        optimization_response = self.model.generate(optimization_prompt)
        
        # 解析优化建议
        try:
            # 尝试提取JSON部分
            if "{" in optimization_response and "}" in optimization_response:
                start_idx = optimization_response.index("{")
                end_idx = optimization_response.rindex("}") + 1
                optimization_json = optimization_response[start_idx:end_idx]
                optimization_plan = json.loads(optimization_json)
            else:
                # 如果不是有效的JSON，创建基本优化计划
                optimization_plan = {
                    "next_action": "重新生成",
                    "action_details": "基于当前上下文重新生成更准确的答案"
                }
        except Exception as e:
            print(f"解析优化计划失败: {str(e)}")
            optimization_plan = {
                "next_action": "重新生成",
                "action_details": f"优化计划解析错误: {str(e)}"
            }
        
        state["optimization_plan"] = optimization_plan
        
        # 打印优化计划
        print("\n[优化计划]")
        print(f"下一步动作: {optimization_plan.get('next_action', '无')}")
        print(f"动作详情: {optimization_plan.get('action_details', '无')}")
        
        # 根据优化计划更新状态
        if optimization_plan.get("next_action") == "重新检索":
            print("\n[优化执行] 准备重新检索相关信息...")
            # 清除现有上下文，准备重新检索
            state["context"] = ""
            # 可以根据action_details添加特定的检索关键词
            if "action_details" in optimization_plan:
                state["query"] += f"\n重点关注: {optimization_plan['action_details']}"
        elif optimization_plan.get("next_action") == "拆解问题" and "action_details" in optimization_plan:
            print("\n[优化执行] 准备拆解问题并分别处理...")
            # 重新分析查询并拆解
            state = self._analyze_query(state)
            state = self._process_query_based_on_analysis(state)
        else:
            print(f"\n[优化执行] 按照计划 '{optimization_plan.get('next_action', '重新生成')}' 进行优化")
            
        print("="*60)
        
        return state
    
    def reset(self) -> None:
        """
        重置代理状态
        """
        self.environment.reset()
        self.memory_manager.reset()