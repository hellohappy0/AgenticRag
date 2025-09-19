import smolagents
import os
import re
from smolagents import Template

# 定义模板目录路径
templates_dir = os.path.join(os.path.dirname(__file__), 'src', 'prompt', 'templates')

def extract_variables_from_template(template_content):
    """从模板内容中提取使用的变量"""
    # 简单的正则表达式来匹配{{ variable }}或{variable}格式的变量
    variables = set()
    
    # 匹配{{ variable }}格式
    double_bracket_vars = re.findall(r'{{\s*(\w+)\s*}}', template_content)
    variables.update(double_bracket_vars)
    
    # 匹配{variable}格式
    single_bracket_vars = re.findall(r'(?<!{){(?!{)\s*(\w+)\s*(?!)}', template_content)
    variables.update(single_bracket_vars)
    
    return list(variables)

def check_template_file(template_file):
    """检查单个模板文件是否能被smolagents的Template类正确解析"""
    try:
        # 读取模板文件内容
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # 创建smolagents的Template实例
        template = Template(template_content)
        
        # 提取模板中使用的变量（使用自定义方法）
        variables = extract_variables_from_template(template_content)
        
        # 输出成功信息
        print(f"✅ 模板文件 '{os.path.basename(template_file)}' 检查通过")
        print(f"   变量列表: {variables}")
        return True
    except Exception as e:
        # 输出错误信息
        print(f"❌ 模板文件 '{os.path.basename(template_file)}' 检查失败")
        print(f"   错误信息: {str(e)}")
        return False

def main():
    """主函数，检查所有模板文件"""
    print("开始检查所有模板文件是否符合smolagents的要求...")
    print(f"模板目录: {templates_dir}")
    print("=" * 80)
    
    # 获取模板目录下的所有txt文件
    template_files = []
    for file in os.listdir(templates_dir):
        if file.endswith('.txt'):
            template_files.append(os.path.join(templates_dir, file))
    
    # 打印找到的模板文件数量
    print(f"找到 {len(template_files)} 个模板文件")
    print("=" * 80)
    
    # 检查每个模板文件
    success_count = 0
    for template_file in template_files:
        if check_template_file(template_file):
            success_count += 1
        print("-" * 80)
    
    # 输出检查结果汇总
    print("检查结果汇总:")
    print(f"总模板数: {len(template_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(template_files) - success_count}")
    
    if success_count == len(template_files):
        print("✅ 所有模板文件都符合smolagents的要求！")
    else:
        print("❌ 有模板文件不符合smolagents的要求，请检查并修复！")

if __name__ == "__main__":
    main()