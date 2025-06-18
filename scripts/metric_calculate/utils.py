#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本分析工具模块

本模块提供文本分析的核心工具函数，包括：
1. 统一的文本预处理（jieba分词 + 停用词过滤）
2. 基于BGE(sentence-transformers)的文本向量化和相似度计算
3. 基于句子长度的文本可读性计算
4. 基于情感词典的净语调和负语调计算
"""

import re
import jieba
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union, Set
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器类
    
    提供统一的文本清洗功能，包括jieba分词和停用词过滤。
    """
    
    def __init__(self, stopwords_path: str = "data/processed_data/stop_words.txt"):
        """初始化文本预处理器
        
        Args:
            stopwords_path (str): 停用词文件路径
        """
        self.stopwords_path = Path(stopwords_path)
        self.stopwords: Set[str] = set()
        self._load_stopwords()
        
        # 设置jieba日志级别，减少输出噪音
        jieba.setLogLevel(logging.WARNING)
    
    def _load_stopwords(self) -> None:
        """加载停用词"""
        try:
            if self.stopwords_path.exists():
                with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            self.stopwords.add(word)
                logger.info(f"成功加载停用词，共 {len(self.stopwords)} 个")
            else:
                logger.warning(f"停用词文件不存在: {self.stopwords_path}")
                logger.info("将不进行停用词过滤")
        except Exception as e:
            logger.error(f"加载停用词时发生错误: {e}")
            logger.info("将不进行停用词过滤")
    
    def clean_text(self, text: str, keep_original_for_readability: bool = False) -> Union[str, Tuple[str, List[str]]]:
        """统一的文本清洗函数
        
        Args:
            text (str): 原始文本
            keep_original_for_readability (bool): 是否保留原文用于可读性计算
            
        Returns:
            Union[str, Tuple[str, List[str]]]: 
                - 如果keep_original_for_readability=False: 返回清洗后的文本字符串
                - 如果keep_original_for_readability=True: 返回(原文, 分词列表)
        """
        try:
            if not text or not text.strip():
                if keep_original_for_readability:
                    return "", []
                return ""
            
            # 基础文本清理
            cleaned_text = text.strip()
            
            # 去除多余的空白字符和特殊字符（但保留基本标点用于可读性计算）
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            # 使用jieba进行分词
            words = list(jieba.cut(cleaned_text))
            
            # 过滤停用词和空白词
            filtered_words = []
            for word in words:
                word = word.strip()
                if (word and 
                    len(word) > 0 and 
                    word not in self.stopwords and
                    not word.isspace()):
                    filtered_words.append(word)
            
            if keep_original_for_readability:
                return cleaned_text, filtered_words
            else:
                # 返回用空格连接的分词结果
                return ' '.join(filtered_words)
                
        except Exception as e:
            logger.error(f"文本清洗时发生错误: {e}")
            if keep_original_for_readability:
                return text, []
            return text
    
    def get_word_list(self, text: str) -> List[str]:
        """获取分词列表
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 分词列表
        """
        result = self.clean_text(text, keep_original_for_readability=True)
        if isinstance(result, tuple):
            _, words = result
            return words
        else:
            # 这种情况不应该发生，但为了类型安全
            return []


class BGEVectorizer:
    """BGE文本向量化器类
    
    使用sentence-transformers的BGE模型进行文本向量化，支持中文文本相似度计算。
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", 
                 device: Optional[str] = None, 
                 normalize_embeddings: bool = True,
                 use_instruction: bool = False,
                 instruction: str = "为这个句子生成表示以用于检索相关文章：",
                 preprocessor: Optional[TextPreprocessor] = None):
        """初始化BGE向量化器
        
        Args:
            model_name (str): BGE模型名称，默认为bge-large-zh-v1.5
            device (str, optional): 计算设备，如果为None则自动选择
            normalize_embeddings (bool): 是否归一化嵌入向量，默认True
            use_instruction (bool): 是否使用指令前缀，默认False
            instruction (str): 指令前缀文本
            preprocessor (Optional[TextPreprocessor]): 文本预处理器
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_instruction = use_instruction
        self.instruction = instruction
        self.preprocessor = preprocessor or TextPreprocessor()
        self.model = None   

        # 初始化模型
        self.SentenceTransformer = SentenceTransformer
        self._load_model()
    
    def _load_model(self) -> None:
        """加载BGE模型"""
        try:
            logger.info(f"正在加载BGE模型: {self.model_name}")
            
            # 创建模型配置
            model_kwargs = {}
            if self.device is not None:
                model_kwargs['device'] = self.device
            
            # 加载模型
            self.model = self.SentenceTransformer(
                self.model_name,
                **model_kwargs
            )
            
            logger.info(f"BGE模型加载完成")
            logger.info(f"模型信息: {self.model_name}")
            logger.info(f"最大序列长度: {self.model.max_seq_length}")
            logger.info(f"嵌入维度: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"加载BGE模型时发生错误: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        # 使用统一的预处理器，明确指定需要字符串结果
        cleaned_result = self.preprocessor.clean_text(text)
        
        # 检查返回类型，确保得到的是字符串
        if isinstance(cleaned_result, tuple):
            # 如果返回了元组（当keep_original_for_readability=True时），取第一个元素（原始文本）
            # 但在我们的使用场景中，这不应该发生，因为clean_text默认返回字符串
            logger.warning("clean_text返回了元组，但预期是字符串。使用原始文本。")
            cleaned_text = cleaned_result[0]
        else:
            cleaned_text = cleaned_result
        
        # 如果启用指令前缀，则添加指令
        if self.use_instruction:
            cleaned_text = self.instruction + cleaned_text
        
        return cleaned_text
    
    def get_text_vector(self, text: str) -> Optional[np.ndarray]:
        """获取单个文本的向量表示
        
        Args:
            text (str): 输入文本
            
        Returns:
            Optional[np.ndarray]: 文本向量，如果失败则返回None
        """
        if self.model is None:
            logger.error("模型尚未加载")
            return None
        
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            if not processed_text.strip():
                logger.warning("预处理后的文本为空")
                return None
            
            # 生成嵌入 - 禁用进度条
            embeddings = self.model.encode(
                [processed_text],
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False  # 禁用进度条
            )
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"获取文本向量时发生错误: {e}")
            return None
    
    def get_text_vectors(self, texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """批量获取文本向量表示
        
        Args:
            texts (List[str]): 文本列表
            batch_size (int): 批处理大小，默认32
            
        Returns:
            List[Optional[np.ndarray]]: 文本向量列表
        """
        if self.model is None:
            logger.error("模型尚未加载")
            return [None] * len(texts)
        
        try:
            # 预处理所有文本
            processed_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                processed_text = self._preprocess_text(text)
                if processed_text.strip():
                    processed_texts.append(processed_text)
                    valid_indices.append(i)
                else:
                    logger.warning(f"第{i}个文本预处理后为空，跳过")
            
            if not processed_texts:
                logger.warning("所有文本预处理后都为空")
                return [None] * len(texts)
            
            # 批量生成嵌入 - 禁用进度条
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False  # 禁用进度条
            )
            
            # 重建完整的结果列表
            results: List[Optional[np.ndarray]] = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_index = valid_indices[i]
                results[original_index] = embedding
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取文本向量时发生错误: {e}")
            return [None] * len(texts)


class SimilarityCalculator:
    """相似度计算器类
    
    基于BGE文档向量计算余弦相似度。
    """
    
    @staticmethod
    def cosine_similarity_vectors(vector1: Optional[np.ndarray], vector2: Optional[np.ndarray]) -> float:
        """计算两个向量的余弦相似度
        
        基于公式：Sim = (v1 · v2) / (||v1|| * ||v2||)
        如果向量已经归一化，则可以直接使用点积计算
        
        Args:
            vector1 (np.ndarray): 第一个向量
            vector2 (np.ndarray): 第二个向量
            
        Returns:
            float: 余弦相似度，范围[-1, 1]
        """
        try:
            # 检查向量是否有效
            if vector1 is None or vector2 is None:
                logger.warning("输入向量包含None值")
                return 0.0
            
            # 确保向量为numpy数组
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 检查向量维度是否匹配
            if v1.shape != v2.shape:
                logger.error(f"向量维度不匹配: {v1.shape} vs {v2.shape}")
                return 0.0
            
            # 检查向量是否为零向量
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                logger.warning("存在零向量")
                return 0.0
            
            # 计算余弦相似度
            # 如果向量已经归一化，可以直接使用点积
            if np.allclose(np.linalg.norm(v1), 1.0) and np.allclose(np.linalg.norm(v2), 1.0):
                # 向量已归一化，直接计算点积
                similarity = np.dot(v1, v2)
            else:
                # 向量未归一化，使用sklearn的cosine_similarity
                v1 = v1.reshape(1, -1)
                v2 = v2.reshape(1, -1)
                similarity = cosine_similarity(v1, v2)[0, 0]
            
            # 处理可能的NaN值
            if np.isnan(similarity):
                logger.warning("相似度计算结果为NaN")
                return 0.0
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算余弦相似度时发生错误: {e}")
            return 0.0
    
    @staticmethod
    def text_similarity(text1: str, text2: str, vectorizer: BGEVectorizer) -> float:
        """计算两个文本的相似度
        
        Args:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            vectorizer (BGEVectorizer): BGE向量化器
            
        Returns:
            float: 文本相似度，范围[-1, 1]
        """
        try:
            # 获取文档向量
            vector1 = vectorizer.get_text_vector(text1)
            vector2 = vectorizer.get_text_vector(text2)
            
            # 计算相似度
            return SimilarityCalculator.cosine_similarity_vectors(vector1, vector2)
            
        except Exception as e:
            logger.error(f"计算文本相似度时发生错误: {e}")
            return 0.0
    
    @staticmethod
    def batch_text_similarity(texts1: List[str], texts2: List[str], 
                            vectorizer: BGEVectorizer) -> List[List[float]]:
        """批量计算文本相似度矩阵
        
        Args:
            texts1 (List[str]): 第一组文本列表
            texts2 (List[str]): 第二组文本列表
            vectorizer (BGEVectorizer): BGE向量化器
            
        Returns:
            List[List[float]]: 相似度矩阵，形状为[len(texts1), len(texts2)]
        """
        try:
            # 批量获取向量
            vectors1 = vectorizer.get_text_vectors(texts1)
            vectors2 = vectorizer.get_text_vectors(texts2)
            
            # 计算相似度矩阵
            similarity_matrix = []
            for i, v1 in enumerate(vectors1):
                row = []
                for j, v2 in enumerate(vectors2):
                    similarity = SimilarityCalculator.cosine_similarity_vectors(v1, v2)
                    row.append(similarity)
                similarity_matrix.append(row)
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"批量计算文本相似度时发生错误: {e}")
            return [[0.0] * len(texts2) for _ in range(len(texts1))]


class ReadabilityCalculator:
    """可读性计算器类
    
    基于平均句子长度计算文本可读性。
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """初始化可读性计算器
        
        Args:
            preprocessor (Optional[TextPreprocessor]): 文本预处理器
        """
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def calculate_readability(self, text: str, use_character_count: bool = True) -> float:
        """计算文本可读性
        
        基于方法：可读性 = 文本总字数 / 文本句子总数
        
        Args:
            text (str): 输入文本
            use_character_count (bool): True使用字符数计算，False使用词数计算
            
        Returns:
            float: 平均句子长度
        """
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                return 0.0
            
            # 获取清洗后的文本和分词结果
            result = self.preprocessor.clean_text(text, keep_original_for_readability=True)
            if isinstance(result, tuple):
                original_text, words = result
            else:
                # 这种情况不应该发生，但为了类型安全
                original_text = text
                words = []
            
            if use_character_count:
                # 使用字符数计算
                # 清理文本，去除多余空白字符
                cleaned_text = re.sub(r'\s+', '', original_text)
                total_units = len(cleaned_text)
            else:
                # 使用词数计算（考虑分词后的效果）
                total_units = len(words)
            
            if total_units == 0:
                return 0.0
            
            # 使用正则表达式查找句子结束标识
            # 匹配句号、感叹号、问号（包括中英文）
            sentence_endings = re.findall(r'[。！？.!?]', original_text)
            sentence_count = len(sentence_endings)
            
            # 如果没有找到句子结束标识，则认为整个文本是一个句子
            if sentence_count == 0:
                sentence_count = 1
            
            # 计算平均句子长度
            avg_sentence_length = total_units / sentence_count
            
            logger.debug(f"文本总单位数: {total_units}, 句子数: {sentence_count}, 平均句子长度: {avg_sentence_length}")
            
            return avg_sentence_length
            
        except Exception as e:
            logger.error(f"计算文本可读性时发生错误: {e}")
            return 0.0


class SentimentCalculator:
    """情感计算器类
    
    基于情感词典计算文本的净语调和负语调指标。
    """
    
    def __init__(self, positive_words: List[str], negative_words: List[str], 
                 preprocessor: Optional[TextPreprocessor] = None):
        """初始化情感计算器
        
        Args:
            positive_words (List[str]): 积极词列表
            negative_words (List[str]): 消极词列表
            preprocessor (Optional[TextPreprocessor]): 文本预处理器
        """
        self.positive_words = set(positive_words) if positive_words else set()
        self.negative_words = set(negative_words) if negative_words else set()
        self.preprocessor = preprocessor or TextPreprocessor()
        
        logger.info(f"情感计算器初始化完成，积极词: {len(self.positive_words)}个, 消极词: {len(self.negative_words)}个")
    
    def count_sentiment_words(self, text: str) -> Tuple[int, int]:
        """统计文本中的情感词数量
        
        Args:
            text (str): 输入文本
            
        Returns:
            Tuple[int, int]: (积极词数量, 消极词数量)
        """
        try:
            if not text or not text.strip():
                return 0, 0
            
            # 获取分词后的词汇列表
            words = self.preprocessor.get_word_list(text)
            
            if not words:
                return 0, 0
            
            # 统计积极词和消极词出现次数
            positive_count = 0
            negative_count = 0
            
            # 创建词汇计数字典以提高效率
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # 统计积极词
            for word in self.positive_words:
                if word in word_counts:
                    positive_count += word_counts[word]
            
            # 统计消极词
            for word in self.negative_words:
                if word in word_counts:
                    negative_count += word_counts[word]
            
            return positive_count, negative_count
            
        except Exception as e:
            logger.error(f"统计情感词数量时发生错误: {e}")
            return 0, 0
    
    def calculate_tone(self, text: str) -> float:
        """计算净语调指标（TONE）
        
        基于公式：TONE = (Pos - Neg) / (Pos + Neg)
        范围：[-1, 1]
        
        Args:
            text (str): 输入文本
            
        Returns:
            float: 净语调指标
        """
        try:
            positive_count, negative_count = self.count_sentiment_words(text)
            
            # 如果积极词和消极词都为0，返回0
            if positive_count + negative_count == 0:
                return 0.0
            
            # 计算净语调
            tone = (positive_count - negative_count) / (positive_count + negative_count)
            
            logger.debug(f"积极词数: {positive_count}, 消极词数: {negative_count}, 净语调: {tone}")
            
            return tone
            
        except Exception as e:
            logger.error(f"计算净语调时发生错误: {e}")
            return 0.0
    
    def calculate_negative_tone(self, text: str) -> float:
        """计算负语调指标（NTONE）
        
        基于公式：NTONE = Neg / (Pos + Neg)
        范围：[0, 1]
        
        Args:
            text (str): 输入文本
            
        Returns:
            float: 负语调指标
        """
        try:
            positive_count, negative_count = self.count_sentiment_words(text)
            
            # 如果积极词和消极词都为0，返回0
            if positive_count + negative_count == 0:
                return 0.0
            
            # 计算负语调
            negative_tone = negative_count / (positive_count + negative_count)
            
            logger.debug(f"积极词数: {positive_count}, 消极词数: {negative_count}, 负语调: {negative_tone}")
            
            return negative_tone
            
        except Exception as e:
            logger.error(f"计算负语调时发生错误: {e}")
            return 0.0
    
    def calculate_all_metrics(self, text: str) -> Dict[str, Union[float, int]]:
        """计算所有情感指标
        
        Args:
            text (str): 输入文本
            
        Returns:
            Dict[str, Union[float, int]]: 包含所有情感指标的字典
        """
        try:
            positive_count, negative_count = self.count_sentiment_words(text)
            
            # 计算所有指标
            results = {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_sentiment_words': positive_count + negative_count,
                'tone': self.calculate_tone(text),
                'negative_tone': self.calculate_negative_tone(text)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"计算所有情感指标时发生错误: {e}")
            return {
                'positive_count': 0,
                'negative_count': 0,
                'total_sentiment_words': 0,
                'tone': 0.0,
                'negative_tone': 0.0
            }


def create_sentiment_calculator_from_file(emotion_dict_path: str, 
                                        preprocessor: Optional[TextPreprocessor] = None) -> Optional[SentimentCalculator]:
    """从情感词典文件创建情感计算器的便利函数
    
    Args:
        emotion_dict_path (str): 情感词典CSV文件路径
        preprocessor (Optional[TextPreprocessor]): 文本预处理器
        
    Returns:
        Optional[SentimentCalculator]: 情感计算器实例，失败时返回None
    """
    try:
        import pandas as pd
        
        # 读取情感词典
        df = pd.read_csv(emotion_dict_path, encoding='utf-8')
        
        # 提取积极词和消极词，去除NaN值
        positive_words = df['positive'].dropna().tolist()
        negative_words = df['negative'].dropna().tolist()
        
        # 创建情感计算器
        calculator = SentimentCalculator(positive_words, negative_words, preprocessor)
        
        logger.info(f"从文件 {emotion_dict_path} 成功创建情感计算器")
        return calculator
        
    except Exception as e:
        logger.error(f"从文件创建情感计算器时发生错误: {e}")
        return None


def create_bge_vectorizer(model_name: str = "BAAI/bge-large-zh-v1.5",
                         device: Optional[str] = None,
                         normalize_embeddings: bool = True,
                         use_instruction: bool = False,
                         preprocessor: Optional[TextPreprocessor] = None) -> Optional[BGEVectorizer]:
    """创建BGE向量化器的便利函数
    
    Args:
        model_name (str): BGE模型名称
        device (Optional[str]): 计算设备
        normalize_embeddings (bool): 是否归一化嵌入向量
        use_instruction (bool): 是否使用指令前缀
        preprocessor (Optional[TextPreprocessor]): 文本预处理器
        
    Returns:
        Optional[BGEVectorizer]: BGE向量化器实例，失败时返回None
    """
    try:
        vectorizer = BGEVectorizer(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            use_instruction=use_instruction,
            preprocessor=preprocessor
        )
        
        logger.info(f"成功创建BGE向量化器: {model_name}")
        return vectorizer
        
    except Exception as e:
        logger.error(f"创建BGE向量化器时发生错误: {e}")
        return None