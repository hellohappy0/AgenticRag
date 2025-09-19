import unittest
import os
import re
import subprocess
import sys

# 检查每个测试文件是否有语法错误或导入错误
def check_test_file(file_path):
    try:
        # 使用python -m py_compile检查语法错误
        compile_result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], capture_output=True, text=True)
        if compile_result.returncode != 0:
            return False, f"语法错误或导入错误: {compile_result.stderr}"
        
        # 尝试导入测试文件来检查导入错误
        module_name = file_path.replace(os.path.sep, '.').replace('.py', '')
        __import__(module_name)
        return True, "没有错误"
    except Exception as e:
        return False, f"导入错误: {str(e)}"

# 获取所有测试文件
test_files = []
for root, _, files in os.walk('tests'):
    for file in files:
        if file.startswith('test_') and file.endswith('.py'):
            test_files.append(os.path.join(root, file))

# 检查每个测试文件
print("测试文件检查结果:")
valid_files = []
invalid_files = []
for file in test_files:
    is_valid, message = check_test_file(file)
    if is_valid:
        print(f"✅ {file}: {message}")
        valid_files.append(file)
    else:
        print(f"❌ {file}: {message}")
        invalid_files.append(file)

print(f"\n有效文件数量: {len(valid_files)}")
print(f"无效文件数量: {len(invalid_files)}")

# 使用更直接的方法获取所有发现的测试方法
def collect_tests(suite):
    tests = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            tests.extend(collect_tests(test))
        elif isinstance(test, unittest.TestCase):
            tests.append(f"{test.__class__.__module__}.{test.__class__.__name__}.{test._testMethodName}")
    return tests

# 只有在有有效文件时才继续
discovered_tests = []
if valid_files:
    # 运行unittest discover并捕获发现的测试
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    discovered_tests = collect_tests(test_suite)
    
    # 打印发现的测试数量
    print(f"\ncollect_tests函数发现的测试数量: {len(discovered_tests)}")

# 手动统计所有测试方法
all_test_methods = []
all_test_details = []  # 存储更详细的测试方法信息

for root, _, files in os.walk('tests'):
    for file in files:
        if file.startswith('test_') and file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找所有测试类
                class_matches = re.finditer(r'class\s+(\w+)\s*\(unittest\.TestCase\)', content)
                
                for class_match in class_matches:
                    class_name = class_match.group(1)
                    # 获取类定义之后到下一个类定义或文件结束的内容
                    class_start = class_match.end()
                    next_class_match = re.search(r'\bclass\s+\w+\s*\(', content[class_start:])
                    class_content = content[class_start:class_start + next_class_match.start()] if next_class_match else content[class_start:]
                    
                    # 查找该类中所有以test_开头的方法
                    test_methods = re.findall(r'def\s+(test_\w+)\s*\(', class_content)
                    
                    # 获取模块路径
                    module_path = file_path.replace('\\', '.').replace('/', '.').replace('.py', '')
                    
                    # 格式化为unittest发现的格式并存储详细信息
                    for method in test_methods:
                        test_full_name = f"{module_path}.{class_name}.{method}"
                        all_test_methods.append(test_full_name)
                        all_test_details.append({
                            'full_name': test_full_name,
                            'module': module_path,
                            'class': class_name,
                            'method': method,
                            'file': file_path
                        })

# 按文件统计测试方法
file_test_counts = {}

# 初始化每个文件的测试计数
for file_path in test_files:
    file_key = file_path.replace('\\', '/')
    file_test_counts[file_key] = {
        'unittest_discovered': 0,
        'manual_discovered': 0,
        'undiscovered': [],
        'unittest_only': []
    }

# 统计unittest发现的每个文件的测试方法数量
for test in discovered_tests:
    # 提取模块路径的文件部分
    module_parts = test.split('.')
    # 构建可能的文件路径
    for i in range(len(module_parts)):
        candidate_path = '.'.join(module_parts[:i+1])
        found = False
        for file_path in test_files:
            file_key = file_path.replace('\\', '/')
            file_module = file_path.replace('\\', '.').replace('/', '.').replace('.py', '')
            if file_module.endswith(candidate_path) or candidate_path.endswith(file_module):
                file_test_counts[file_key]['unittest_discovered'] += 1
                found = True
                break
        if found:
            break

# 统计手动发现的每个文件的测试方法数量
for test_detail in all_test_details:
    file_path = test_detail['file'].replace('\\', '/')
    file_test_counts[file_path]['manual_discovered'] += 1

# 比较发现的测试和手动统计的测试
print(f"unittest实际发现的测试数量: {len(discovered_tests)}")
print(f"手动统计的测试方法数量: {len(all_test_methods)}")

# 打印前5个unittest发现的测试方法示例
print("\nunittest发现的测试方法示例（前5个）:")
for test in discovered_tests[:5]:
    print(f"- {test}")

# 打印前5个手动统计的测试方法示例
print("\n手动统计的测试方法示例（前5个）:")
for test in all_test_methods[:5]:
    print(f"- {test}")

# 打印按文件统计的测试方法数量
print("\n按文件统计的测试方法数量:")
for file_path, counts in file_test_counts.items():
    print(f"- {file_path}:")
    print(f"  unittest发现: {counts['unittest_discovered']}")
    print(f"  手动统计: {counts['manual_discovered']}")
    print(f"  差异: {counts['manual_discovered'] - counts['unittest_discovered']}")

# 统一测试方法格式进行比较
def normalize_test_name(test_name):
    # 移除tests.前缀（如果有）
    if test_name.startswith('tests.'):
        normalized = test_name[6:]
    else:
        normalized = test_name
    
    # 标准化路径分隔符（处理Windows和Unix路径差异）
    normalized = normalized.replace('\\', '.').replace('/', '.')
    
    return normalized

normalized_manual_tests = {normalize_test_name(test) for test in all_test_methods}
normalized_unittest_tests = {normalize_test_name(test) for test in discovered_tests}

# 打印标准化后的测试方法示例
print("\n标准化后的unittest测试方法示例（前5个）:")
for i, test in enumerate(normalized_unittest_tests):
    if i >= 5:
        break
    print(f"- {test}")

print("\n标准化后的手动统计测试方法示例（前5个）:")
for i, test in enumerate(normalized_manual_tests):
    if i >= 5:
        break
    print(f"- {test}")

# 调试信息：打印测试方法格式差异
if discovered_tests and all_test_methods:
    print("\n调试信息：")
    print(f"- 手动统计的一个测试方法: {all_test_methods[0]}")
    print(f"- unittest发现的一个测试方法: {discovered_tests[0]}")
    print(f"- 标准化后的手动测试: {normalize_test_name(all_test_methods[0])}")
    print(f"- 标准化后的unittest测试: {normalize_test_name(discovered_tests[0])}")

# 找出未被unittest发现的测试方法
undiscovered_tests = [test for test in all_test_methods if normalize_test_name(test) not in normalized_unittest_tests]

# 找出被unittest发现但不在手动统计中的测试方法
extra_tests = [test for test in discovered_tests if normalize_test_name(test) not in normalized_manual_tests]

if undiscovered_tests:
    print(f"\n以下{len(undiscovered_tests)}个测试方法在手动统计中但未被unittest发现:")
    for test in undiscovered_tests:
        print(f"- {test}")
else:
    print("\n所有手动统计的测试方法都被unittest发现了！")

if extra_tests:
    print(f"\n以下{len(extra_tests)}个测试方法被unittest发现但不在手动统计中:")
    for test in extra_tests:
        print(f"- {test}")
else:
    print("\nunittest没有发现额外的测试方法。")