import smolagents
import inspect
import pkgutil

def explore_package(package, depth=0, max_depth=2):
    """递归探索包的结构"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    print(f"{indent}📦 {package.__name__}")
    
    # 获取包中的所有模块和子包
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
        try:
            # 导入模块
            module = __import__(f"{package.__name__}.{name}", fromlist=[name])
            if is_pkg:
                # 如果是子包，递归探索
                explore_package(module, depth + 1, max_depth)
            else:
                # 如果是模块，列出其内容
                print(f"{indent}  📄 {name}")
                
                # 尝试列出模块中的主要类和函数
                try:
                    for attr_name in dir(module):
                        if not attr_name.startswith("_"):
                            attr = getattr(module, attr_name)
                            if (inspect.isclass(attr) or inspect.isfunction(attr)) and attr.__module__ == module.__name__:
                                print(f"{indent}    🔹 {attr_name}")
                except Exception:
                    pass
        except Exception as e:
            print(f"{indent}  ❌ 无法导入 {name}: {e}")

if __name__ == "__main__":
    print("开始探索smolagents库的结构...")
    explore_package(smolagents)
    
    # 特别检查一下是否有prompt相关的模块
    try:
        import smolagents.prompt
        print("\n🔍 找到smolagents.prompt模块")
        explore_package(smolagents.prompt, max_depth=1)
    except ImportError:
        print("\n❌ 未找到smolagents.prompt模块")
    
    # 检查是否有template相关的功能
    print("\n🔍 检查smolagents中与template相关的内容:")
    for attr_name in dir(smolagents):
        if "template" in attr_name.lower():
            print(f"  - {attr_name}")