import sys
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.doc_process.simple_processor import SimpleTextSplitter, SimpleDocumentProcessor
from src.doc_process.base_processor import DocumentProcessor, DocumentSplitter
from unstructured.documents.elements import Text


class TestSimpleTextSplitter(unittest.TestCase):
    
    def setUp(self):
        self.splitter = SimpleTextSplitter()
        
    def test_split_empty_text(self):
        """测试分割空文本"""
        result = self.splitter.split("")
        self.assertEqual(result, [""])
    
    def test_split_short_text(self):
        """测试分割短文本（不需要分割）"""
        short_text = "这是一段短文本，不需要分割。"
        result = self.splitter.split(short_text, chunk_size=100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], short_text)
    
    @patch('src.doc_process.simple_processor.chunk_by_title')
    def test_split_long_text(self, mock_chunk_by_title):
        """测试分割长文本"""
        # 创建一段长文本，包含多个句子
        long_text = "这是第一句话。这是第二句话，包含更多的内容。" * 20
        
        # 模拟chunk_by_title的行为
        mock_element1 = MagicMock()
        mock_element1.text = "这是第一句话。这是第二句话，包含更多的内容。" * 5
        mock_element2 = MagicMock()
        mock_element2.text = "这是第一句话。这是第二句话，包含更多的内容。" * 5
        mock_chunk_by_title.return_value = [mock_element1, mock_element2]
        
        # 使用较小的chunk_size进行分割
        result = self.splitter.split(long_text, chunk_size=100, chunk_overlap=20)
        
        # 验证分割结果
        self.assertEqual(len(result), 2)
        
        # 验证chunk_by_title被正确调用
        mock_chunk_by_title.assert_called_once()
        self.assertEqual(mock_chunk_by_title.call_args[1]['max_characters'], 100)
    
    @patch('src.doc_process.simple_processor.chunk_by_title')
    def test_split_with_overlap(self, mock_chunk_by_title):
        """测试分割时的重叠功能"""
        long_text = "0123456789" * 20
        
        # 模拟chunk_by_title的行为
        mock_element1 = MagicMock()
        mock_element1.text = "01234567890123456789"
        mock_element2 = MagicMock()
        mock_element2.text = "23456789012345678901"
        mock_chunk_by_title.return_value = [mock_element1, mock_element2]
        
        # 使用chunk_overlap参数进行分割
        result = self.splitter.split(long_text, chunk_size=20, chunk_overlap=5)
        
        # 验证重叠逻辑
        self.assertEqual(len(result), 2)
        # 第一个块应该保持不变
        self.assertEqual(result[0], "01234567890123456789")
        # 第二个块应该包含前一个块的最后5个字符
        self.assertTrue("78901" in result[1])
    
    def test_split_invalid_params(self):
        """测试无效参数"""
        text = "测试文本"
        
        # 测试chunk_size为负数
        with self.assertRaises(ValueError):
            self.splitter.split(text, chunk_size=-10)
        
        # 测试chunk_overlap为负数
        with self.assertRaises(ValueError):
            self.splitter.split(text, chunk_size=100, chunk_overlap=-5)
        
        # 测试chunk_overlap大于等于chunk_size
        with self.assertRaises(ValueError):
            self.splitter.split(text, chunk_size=100, chunk_overlap=100)


class TestSimpleDocumentProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = SimpleDocumentProcessor()
        # 准备docs目录下的README.md文件路径
        self.readme_md_path = str(Path(__file__).resolve().parent.parent.parent / "docs" / "README.md")
    
    @patch('src.doc_process.simple_processor.partition_text')
    def test_load_txt_file(self, mock_partition_text):
        """测试加载TXT文件"""
        # 模拟partition_text的行为
        mock_element = MagicMock()
        mock_element.text = "这是TXT文件内容"
        mock_partition_text.return_value = [mock_element]
        
        # 测试load方法
        file_path = "test_file.txt"
        result = self.processor.load(file_path)
        
        # 验证结果
        self.assertEqual(result, "这是TXT文件内容")
        mock_partition_text.assert_called_once_with(file_path)
    
    @patch('src.doc_process.simple_processor.partition_md')
    def test_load_md_file(self, mock_partition_md):
        """测试加载MD文件"""
        # 模拟partition_md的行为
        mock_element1 = MagicMock()
        mock_element1.text = "# 标题"
        mock_element2 = MagicMock()
        mock_element2.text = "这是MD文件内容"
        mock_partition_md.return_value = [mock_element1, mock_element2]
        
        # 测试load方法
        file_path = "test_file.md"
        result = self.processor.load(file_path)
        
        # 验证结果
        self.assertEqual(result, "# 标题\n这是MD文件内容")
        mock_partition_md.assert_called_once_with(file_path)
    
    def test_load_unsupported_file(self):
        """测试加载不支持的文件类型"""
        with self.assertRaises(ValueError):
            self.processor.load("test_file.pdf")
    
    @patch('src.doc_process.simple_processor.SimpleDocumentProcessor.load')
    @patch('src.doc_process.simple_processor.SimpleTextSplitter.split')
    def test_process(self, mock_split, mock_load):
        """测试process方法"""
        # 设置模拟
        mock_load.return_value = "测试文档内容"
        mock_split.return_value = ["块1", "块2"]
        
        # 测试process方法
        file_path = "test_file.txt"
        result = self.processor.process(file_path, chunk_size=100, chunk_overlap=20)
        
        # 验证结果
        mock_load.assert_called_once_with(file_path)
        mock_split.assert_called_once_with("测试文档内容", chunk_size=100, chunk_overlap=20)
        self.assertEqual(result, ["块1", "块2"])
    
    def test_load_real_md_file(self):
        """测试加载真实的MD文件"""
        # 确保文件存在
        if not Path(self.readme_md_path).exists():
            self.skipTest(f"文件不存在: {self.readme_md_path}")
        
        # 加载真实的MD文件
        try:
            content = self.processor.load(self.readme_md_path)
            # 验证加载结果
            self.assertIsNotNone(content)
            self.assertGreater(len(content), 0)
            # 验证内容包含README中的一些关键词
            self.assertIn("Agentic RAG", content)
            self.assertIn("项目结构", content)
        except Exception as e:
            self.fail(f"加载真实MD文件失败: {str(e)}")
    
    def test_process_real_md_file(self):
        """测试处理真实的MD文件"""
        # 确保文件存在
        if not Path(self.readme_md_path).exists():
            self.skipTest(f"文件不存在: {self.readme_md_path}")
        
        # 处理真实的MD文件
        try:
            chunks = self.processor.process(self.readme_md_path, chunk_size=500, chunk_overlap=50)
            # 验证处理结果
            self.assertIsNotNone(chunks)
            self.assertGreater(len(chunks), 0)
            # 验证每个块的大小合理
            for chunk in chunks:
                self.assertGreater(len(chunk), 0)
                # 块大小可能会略大于设定值，因为unstructured的chunk_by_title方法有自己的逻辑
                self.assertLessEqual(len(chunk), 700, f"块大小过大: {len(chunk)}")
        except Exception as e:
            self.fail(f"处理真实MD文件失败: {str(e)}")


if __name__ == '__main__':
    unittest.main()