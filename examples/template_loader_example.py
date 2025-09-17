"""
模板加载器使用示例

这个示例展示了如何在系统的其他部分直接使用template_loader单例来加载和使用提示模板
"""

from src.prompt.template_loader import template_loader


def main():
    print("===== 模板加载器使用示例 =====")
    
    # 示例1：直接加载单个模板
    try:
        main_template = template_loader.load_template("main")
        print(f"\n1. 加载'main'模板成功！")
        print(f"   模板内容长度: {len(main_template)} 字符")
        # 显示模板的前100个字符
        print(f"   模板前100字符预览: {main_template[:100]}...")
    except FileNotFoundError as e:
        print(f"加载模板失败: {e}")
    
    # 示例2：加载所有模板
    all_templates = template_loader.load_all_templates()
    print(f"\n2. 加载所有模板成功！")
    print(f"   共加载 {len(all_templates)} 个模板")
    print(f"   模板名称列表: {list(all_templates.keys())}")
    
    # 示例3：检查模板目录路径
    templates_dir = template_loader.get_templates_dir()
    print(f"\n3. 模板目录路径:")
    print(f"   {templates_dir}")
    
    # 示例4：使用模板内容
    if "rag_answer" in all_templates:
        rag_template = all_templates["rag_answer"]
        print(f"\n4. 使用'rag_answer'模板格式化示例:")
        # 模拟格式化模板
        formatted_prompt = rag_template.format(
            query="什么是大型语言模型？",
            context="大型语言模型（LLM）是一类基于深度学习的模型，能够理解和生成人类语言。"
        )
        print(f"   格式化后的提示:\n{formatted_prompt}")
    
    # 示例5：清除缓存并重新加载
    print(f"\n5. 清除模板缓存并重新加载...")
    template_loader.clear_cache()
    # 重新加载所有模板
    all_templates_after_clear = template_loader.load_all_templates()
    print(f"   清除缓存后，模板数量: {len(all_templates_after_clear)}")
    

if __name__ == "__main__":
    main()