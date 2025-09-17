import unittest
from src.model.language_model import ModelFactory, BaseLanguageModel
from src.config import get_config
import time
import requests


class TestOllamaModel(unittest.TestCase):
    """
    测试Ollama本地模型的可用性
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        # 简单的测试提示词
        self.test_prompt = "请用一句话介绍你自己"
        # 允许的测试超时时间（秒）
        self.timeout = 120  # 增加超时时间到120秒，给Ollama模型足够的响应时间
    
    def test_ollama_model_availability(self):
        """
        测试Ollama模型的可用性
        """
        print("\n===== 测试Ollama模型可用性 =====")
        
        try:
            # 从配置获取Ollama模型配置
            model_name = get_config("model.ollama.model_name", "qwen3:4b")
            base_url = get_config("model.ollama.base_url", "http://localhost:11434")
            max_tokens = get_config("model.common.max_tokens", 512)
            temperature = get_config("model.common.temperature", 0.7)
            
            print(f"使用配置: model_name={model_name}, base_url={base_url}, max_tokens={max_tokens}, temperature={temperature}")
            
            # 首先检查Ollama服务是否可用
            self._check_ollama_service_health(base_url)
            
            # 创建Ollama模型实例
            ollama_model = ModelFactory.create_model(
                "ollama",
                model_name=model_name,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 验证模型实例类型
            self.assertIsInstance(ollama_model, BaseLanguageModel)
            
            # 记录开始时间
            start_time = time.time()
            
            # 尝试生成响应
            print("正在调用Ollama模型...")
            try:
                response = ollama_model.generate(self.test_prompt)
            except Exception as e:
                print(f"Ollama模型调用异常: {str(e)}")
                # 检查是否有正在运行的Ollama服务
                try:
                    response = requests.get(f"{base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json()
                        print(f"Ollama服务运行中，已安装模型: {[model['name'] for model in models.get('models', [])]}")
                    else:
                        print(f"Ollama服务返回非200状态码: {response.status_code}")
                except Exception as req_e:
                    print(f"无法连接到Ollama服务: {str(req_e)}")
                self.fail(f"Ollama模型调用失败: {str(e)}")
            
            # 检查是否超时
            elapsed_time = time.time() - start_time
            self.assertLess(elapsed_time, self.timeout, f"Ollama模型调用超时（{elapsed_time:.2f}秒）")
            
            # 检查响应是否有效
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response.strip()) > 0)
            
            # 清理响应内容中的特殊标记
            clean_response = response
            if "</think>" in response:
                # 移除思考过程标记
                clean_response = response.split("</think>")[-1].strip()
                print("注意: 响应中包含模型思考过程，已清理")
            
            print(f"Ollama模型响应成功，耗时: {elapsed_time:.2f}秒")
            print(f"响应内容: {clean_response[:100]}...")
            
        except ImportError as e:
            print(f"ImportError: {str(e)}")
            print("请确保已安装必要的依赖库，例如: pip install requests")
            self.fail(f"Ollama模型导入失败: {str(e)}")
        except Exception as e:
            print(f"Ollama模型测试失败: {str(e)}")
            # 不将连接错误视为测试失败，因为这可能是环境问题
            if not isinstance(e, AssertionError):
                print("提示: 请确保Ollama服务正在运行，并且已下载qwen3:4b模型")
                print("可以使用命令 'ollama run qwen3:4b' 下载并启动模型")
    
    def _check_ollama_service_health(self, base_url):
        """
        检查Ollama服务的健康状态
        
        @param base_url: Ollama服务的基础URL
        """
        try:
            print("正在检查Ollama服务健康状态...")
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                print(f"Ollama服务健康，已安装模型数量: {len(model_names)}")
                print(f"已安装模型列表: {', '.join(model_names)}")
                
                # 检查配置的模型是否已安装
                config_model = get_config("model.ollama.model_name", "qwen3:4b")
                if config_model not in model_names:
                    print(f"警告: 配置的模型 '{config_model}' 未在Ollama服务中安装")
                    print(f"请使用命令 'ollama pull {config_model}' 下载该模型")
            else:
                print(f"Ollama服务返回非200状态码: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"无法连接到Ollama服务: {base_url}")
            print("请确保Ollama服务正在运行")
        except Exception as e:
            print(f"检查Ollama服务状态时发生错误: {str(e)}")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main()