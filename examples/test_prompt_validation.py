#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试提示模板的参数校验功能
验证基于smolagents的提示模板能够正确检测并报告所有缺失的变量
"""

from src.prompt import PromptManager


def test_parameter_validation():
    """测试参数校验功能"""
    print("开始测试参数校验功能...\n")
    
    # 创建提示管理器
    prompt_manager = PromptManager()
    
    # 添加一个包含多个变量的模板
    test_template = "用户查询: {query}\n上下文信息: {context}\n历史记录: {history}\n回答要求: {requirements}"
    prompt_manager.add_template("test_template", test_template)
    
    # 测试用例1: 提供所有必需的参数
    print("测试用例1: 提供所有必需的参数")
    try:
        prompt = prompt_manager.generate_prompt(
            "test_template",
            query="什么是Agentic RAG?",
            context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
            history="用户之前没有提问过相关问题",
            requirements="请用通俗易懂的语言解释"
        )
        print("✅ 测试通过: 所有参数都提供，成功生成提示\n")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}\n")
    
    # 测试用例2: 缺少一个必需的参数
    print("测试用例2: 缺少一个必需的参数")
    try:
        prompt = prompt_manager.generate_prompt(
            "test_template",
            query="什么是Agentic RAG?",
            context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
            requirements="请用通俗易懂的语言解释"
            # 缺少history参数
        )
        print("❌ 测试失败: 应该检测到缺少参数\n")
    except Exception as e:
        print(f"✅ 测试通过: 正确检测到缺少参数: {str(e)}\n")
    
    # 测试用例3: 缺少多个必需的参数
    print("测试用例3: 缺少多个必需的参数")
    try:
        prompt = prompt_manager.generate_prompt(
            "test_template",
            query="什么是Agentic RAG?"
            # 缺少context, history和requirements参数
        )
        print("❌ 测试失败: 应该检测到缺少多个参数\n")
    except Exception as e:
        print(f"✅ 测试通过: 正确检测到缺少参数: {str(e)}\n")
    
    # 测试用例4: 提供额外的参数（不应该报错）
    print("测试用例4: 提供额外的参数")
    try:
        prompt = prompt_manager.generate_prompt(
            "test_template",
            query="什么是Agentic RAG?",
            context="Agentic RAG是一种结合代理能力和检索增强生成的技术",
            history="用户之前没有提问过相关问题",
            requirements="请用通俗易懂的语言解释",
            extra_param="这个是额外的参数，不应该影响功能"
        )
        print("✅ 测试通过: 额外参数不影响功能，成功生成提示\n")
    except Exception as e:
        print(f"❌ 测试失败: 额外参数导致错误: {str(e)}\n")
    
    print("参数校验功能测试完成！")


if __name__ == "__main__":
    test_parameter_validation()