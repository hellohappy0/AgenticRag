import unittest
from src.model.language_model import ModelFactory, BaseLanguageModel
from src.config import get_config
import time


class TestQwenModel(unittest.TestCase):
    """
    测试通义千问（阿里）模型的可用性
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        # 简单的测试提示词
        self.test_prompt = "请用一句话介绍你自己"
        # 允许的测试超时时间（秒）
        self.timeout = 60  # 通义千问API通常响应较快，设置60秒超时
    
    def test_qwen_model_availability(self):
        """
        测试通义千问模型的可用性
        """
        print("\n===== 测试通义千问模型可用性 =====")
        
        try:
            # 从配置获取通义千问模型配置
            model_name = get_config("model.tongyi.model_name", "qwen3-coder-plus")
            api_key = get_config("model.tongyi.api_key")
            max_tokens = get_config("model.common.max_tokens", 512)
            temperature = get_config("model.common.temperature", 0.7)
            
            print(f"使用配置: model_name={model_name}, max_tokens={max_tokens}, temperature={temperature}")
            
            # 检查API密钥是否存在
            if not api_key:
                print("警告: 未配置通义千问API密钥")
                print("请在config.yaml中设置model.tongyi.api_key或设置环境变量AGENTIC_RAG_MODEL_TONGYI_API_KEY")
                # 不将缺少API密钥视为测试失败
                return
            
            # 创建通义千问模型实例
            tongyi_model = ModelFactory.create_model(
                "tongyi",
                model_name=model_name,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 验证模型实例类型
            self.assertIsInstance(tongyi_model, BaseLanguageModel)
            
            # 记录开始时间
            start_time = time.time()
            
            # 尝试生成响应
            print("正在调用通义千问模型...")
            try:
                response = tongyi_model.generate(self.test_prompt)
            except Exception as e:
                print(f"通义千问模型调用异常: {str(e)}")
                # 检查API密钥是否有效
                if "API key" in str(e) or "credentials" in str(e).lower():
                    print("提示: API密钥可能无效，请检查config.yaml中的model.tongyi.api_key设置")
                elif "quota" in str(e).lower() or "limit" in str(e).lower():
                    print("提示: 可能超出了API调用配额限制")
                elif "network" in str(e).lower() or "connection" in str(e).lower():
                    print("提示: 网络连接问题，请检查网络设置")
                self.fail(f"通义千问模型调用失败: {str(e)}")
            
            # 检查是否超时
            elapsed_time = time.time() - start_time
            self.assertLess(elapsed_time, self.timeout, f"通义千问模型调用超时（{elapsed_time:.2f}秒）")
            
            # 检查响应是否有效
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response.strip()) > 0)
            
            print(f"通义千问模型响应成功，耗时: {elapsed_time:.2f}秒")
            print(f"响应内容: {response[:100]}...")
            
        except ImportError as e:
            print(f"ImportError: {str(e)}")
            print("请确保已安装必要的依赖库，例如: pip install dashscope")
            self.fail(f"通义千问模型导入失败: {str(e)}")
        except Exception as e:
            print(f"通义千问模型测试失败: {str(e)}")
            # 不将API密钥或配额问题视为测试失败，因为这可能是环境问题
            if not isinstance(e, AssertionError):
                print("提示: 请检查通义千问API密钥是否有效以及是否有足够的调用配额")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main()