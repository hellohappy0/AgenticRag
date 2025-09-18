#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试提示模板的参数校验功能
验证SimplePromptTemplate能够正确检测并报告所有缺失的变量
"""

import unittest
from src.prompt import PromptManager


class TestPromptValidation(unittest.TestCase):
    """测试提示模板的参数校验功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建提示管理器
        self.prompt_manager = PromptManager()
        
        # 添加一个包含多个变量的模板用于测试
        self.test_template = "用户查询: {query}\n上下文信息: {context}\n历史记录: {history}\n回答要求: {requirements}"
        self.prompt_manager.add_template("test_template", self.test_template)
    
    def test_all_parameters_provided(self):
        """测试提供所有必需的参数"""
        try:
            prompt = self.prompt_manager.generate_prompt(
                "test_template",
                query="什么是Agentic RAG?",
                context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
                history="用户之前没有提问过相关问题",
                requirements="请用通俗易懂的语言解释"
            )
            # 如果没有抛出异常，测试通过
            self.assertTrue(isinstance(prompt, str))
            print("✅ 测试通过: 所有参数都提供，成功生成提示")
        except Exception as e:
            self.fail(f"提供所有参数时抛出异常: {str(e)}")
    
    def test_missing_one_parameter(self):
        """测试缺少一个必需的参数"""
        with self.assertRaises(KeyError) as context:
            self.prompt_manager.generate_prompt(
                "test_template",
                query="什么是Agentic RAG?",
                context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
                requirements="请用通俗易懂的语言解释"
                # 缺少history参数
            )
        
        # 检查异常信息是否包含缺失的参数名
        self.assertIn("history", str(context.exception))
        print(f"✅ 测试通过: 正确检测到缺少参数: {str(context.exception)}")
    
    def test_missing_multiple_parameters(self):
        """测试缺少多个必需的参数"""
        with self.assertRaises(KeyError) as context:
            self.prompt_manager.generate_prompt(
                "test_template",
                query="什么是Agentic RAG?"
                # 缺少context, history和requirements参数
            )
        
        # 检查异常信息是否包含所有缺失的参数名
        exception_message = str(context.exception)
        self.assertIn("context", exception_message)
        self.assertIn("history", exception_message)
        self.assertIn("requirements", exception_message)
        print(f"✅ 测试通过: 正确检测到缺少参数: {str(context.exception)}")
    
    def test_extra_parameters_provided(self):
        """测试提供额外的参数（不应该报错）"""
        try:
            prompt = self.prompt_manager.generate_prompt(
                "test_template",
                query="什么是Agentic RAG?",
                context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
                history="用户之前没有提问过相关问题",
                requirements="请用通俗易懂的语言解释",
                extra_param="这个是额外的参数，不应该影响功能"
            )
            # 如果没有抛出异常，测试通过
            self.assertTrue(isinstance(prompt, str))
            print("✅ 测试通过: 额外参数不影响功能，成功生成提示")
        except Exception as e:
            self.fail(f"提供额外参数时抛出异常: {str(e)}")
    
    def test_real_template_parameters(self):
        """测试实际使用的模板参数验证"""
        # 测试rag_answer模板的参数验证
        rag_answer_template = "基于以下上下文回答用户的查询:\n\n上下文:\n{context}\n\n用户查询:{query}\n\n请根据上下文信息，用中文回答用户的问题，确保回答准确、全面。"
        self.prompt_manager.add_template("rag_answer_test", rag_answer_template)
        
        # 测试缺少参数的情况
        with self.assertRaises(KeyError) as context:
            self.prompt_manager.generate_prompt(
                "rag_answer_test",
                query="什么是Agentic RAG?"
                # 缺少context参数
            )
        
        self.assertIn("context", str(context.exception))
        print(f"✅ 测试通过: 正确检测到rag_answer模板缺少参数: {str(context.exception)}")


if __name__ == "__main__":
    print("开始测试参数校验功能...\n")
    unittest.main()