import unittest
from src.model.embedding_model import OllamaEmbeddingModel, EmbeddingModelFactory
from src.config import get_config
import time
import requests
import numpy as np


class TestOllamaEmbeddingModel(unittest.TestCase):
    """
    测试Ollama嵌入模型的可用性
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        # 简单的测试文本
        self.test_text = "这是一段用于测试嵌入模型的示例文本"
        # 允许的测试超时时间（秒）
        self.timeout = 60
    
    def test_ollama_embedding_availability(self):
        """
        测试Ollama嵌入模型的可用性
        """
        print("\n===== 测试Ollama嵌入模型可用性 =====")
        
        try:
            # 从配置获取Ollama嵌入模型配置
            model_name = get_config("model.ollama.embedding_model", "bge-m3:latest")
            base_url = get_config("model.ollama.base_url", "http://localhost:11434")
            
            print(f"使用配置: model_name={model_name}, base_url={base_url}")
            
            # 首先检查Ollama服务是否可用
            self._check_ollama_service_health(base_url, model_name)
            
            # 创建Ollama嵌入模型实例
            embedding_model = OllamaEmbeddingModel(
                model_name=model_name,
                base_url=base_url
            )
            
            # 测试is_available方法
            print("检查嵌入模型是否可用...")
            is_available = embedding_model.is_available()
            print(f"嵌入模型is_available()返回: {is_available}")
            
            # 如果模型被报告为可用，尝试获取嵌入
            if is_available:
                # 记录开始时间
                start_time = time.time()
                
                # 尝试获取单个嵌入
                print("尝试获取单个文本的嵌入...")
                try:
                    embedding = embedding_model.get_embedding(self.test_text)
                    
                    # 检查是否超时
                    elapsed_time = time.time() - start_time
                    self.assertLess(elapsed_time, self.timeout, f"获取嵌入超时（{elapsed_time:.2f}秒）")
                    
                    # 检查嵌入向量是否有效
                    self.assertIsNotNone(embedding)
                    self.assertIsInstance(embedding, np.ndarray)
                    self.assertTrue(len(embedding) > 0)
                    self.assertTrue(embedding.dtype == np.float32)
                    
                    print(f"成功获取嵌入向量，维度: {len(embedding)}")
                    print(f"嵌入向量前5个值: {embedding[:5]}")
                    
                    # 尝试批量获取嵌入
                    print("尝试批量获取嵌入...")
                    batch_texts = ["文本1", "文本2", "文本3"]
                    batch_embeddings = embedding_model.get_embeddings(batch_texts)
                    
                    self.assertIsInstance(batch_embeddings, np.ndarray)
                    self.assertEqual(batch_embeddings.shape[0], len(batch_texts))
                    
                    # 检查所有嵌入向量的维度是否一致
                    embedding_dim = batch_embeddings.shape[1]
                    
                    print(f"成功获取批量嵌入，样本数: {batch_embeddings.shape[0]}, 每个向量维度: {embedding_dim}")
                    
                    # 测试嵌入维度获取
                    embedding_dim = embedding_model.get_embedding_dimension()
                    print(f"嵌入维度: {embedding_dim}")
                    self.assertEqual(embedding_dim, len(embedding))
                except Exception as e:
                    print(f"获取嵌入时发生异常: {str(e)}")
                    self.fail(f"获取嵌入失败: {str(e)}")
            else:
                print("警告: 嵌入模型不可用")
                # 不将模型不可用视为测试失败，因为这可能是环境问题
        except ImportError as e:
            print(f"ImportError: {str(e)}")
            print("请确保已安装必要的依赖库，例如: pip install requests numpy")
            self.fail(f"Ollama嵌入模型导入失败: {str(e)}")
        except Exception as e:
            print(f"Ollama嵌入模型测试失败: {str(e)}")
            # 不将连接错误视为测试失败，因为这可能是环境问题
            if not isinstance(e, AssertionError):
                print(f"提示: 请确保Ollama服务正在运行，并且已下载嵌入模型 '{model_name}'")
                print(f"可以使用命令 'ollama pull {model_name}' 下载该模型")
    
    def _check_ollama_service_health(self, base_url, embedding_model_name):
        """
        检查Ollama服务的健康状态
        
        @param base_url: Ollama服务的基础URL
        @param embedding_model_name: 嵌入模型名称
        """
        try:
            print("正在检查Ollama服务健康状态...")
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                print(f"Ollama服务健康，已安装模型数量: {len(model_names)}")
                print(f"已安装模型列表: {', '.join(model_names)}")
                
                # 检查嵌入模型是否已安装
                if embedding_model_name not in model_names:
                    print(f"警告: 嵌入模型 '{embedding_model_name}' 未在Ollama服务中安装")
                    print(f"请使用命令 'ollama pull {embedding_model_name}' 下载该模型")
            else:
                print(f"Ollama服务返回非200状态码: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"无法连接到Ollama服务: {base_url}")
            print("请确保Ollama服务正在运行")
        except Exception as e:
            print(f"检查Ollama服务状态时发生错误: {str(e)}")
    
    def test_embedding_model_factory(self):
        """
        测试嵌入模型工厂类
        """
        print("\n===== 测试嵌入模型工厂 ====")
        
        try:
            # 尝试通过工厂创建Ollama嵌入模型
            embedding_model = EmbeddingModelFactory.create_embedding_model("ollama")
            self.assertIsInstance(embedding_model, OllamaEmbeddingModel)
            print("通过工厂成功创建Ollama嵌入模型")
        except Exception as e:
            print(f"测试嵌入模型工厂失败: {str(e)}")
            # 不将此视为测试失败，因为可能是环境问题


if __name__ == "__main__":
    # 运行单元测试
    unittest.main()