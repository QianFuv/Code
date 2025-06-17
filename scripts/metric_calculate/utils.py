#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本分析工具模块

本模块提供文本分析的核心工具函数，包括：
1. 统一的文本预处理（jieba分词 + 停用词过滤）
2. 基于Doc2vec的文本向量化和相似度计算
3. 基于句子长度的文本可读性计算
4. 基于情感词典的净语调和负语调计算
"""

import re
import jieba
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union, Set
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

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


class TextVectorizer:
    """文本向量化器类
    
    使用Doc2vec模型进行文本向量化，支持训练和推理。
    """
    
    def __init__(self, vector_size: int = 350, window: int = 5, min_count: int = 1, 
                 workers: int = 4, epochs: int = 10, preprocessor: Optional[TextPreprocessor] = None):
        """初始化文本向量化器
        
        Args:
            vector_size (int): 向量维度，默认350
            window (int): 窗口大小，默认5
            min_count (int): 最小词频，默认1
            workers (int): 线程数，默认4
            epochs (int): 训练轮数，默认10
            preprocessor (Optional[TextPreprocessor]): 文本预处理器
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Optional[Doc2Vec] = None
        self.preprocessor = preprocessor or TextPreprocessor()
        
    def _preprocess_text(self, text: str) -> List[str]:
        """预处理文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 预处理后的词汇列表
        """
        # 使用统一的预处理器
        words = self.preprocessor.get_word_list(text)
        
        # 如果分词结果为空，使用gensim的simple_preprocess作为备选
        if not words:
            words = simple_preprocess(text, deacc=True, min_len=1, max_len=15)
        
        return words
    
    def train_model(self, documents: List[str], document_ids: Optional[List[str]] = None) -> None:
        """训练Doc2vec模型
        
        Args:
            documents (List[str]): 文档列表
            document_ids (Optional[List[str]]): 文档ID列表，如果不提供则自动生成
        """
        try:
            # 如果没有提供文档ID，自动生成
            if document_ids is None:
                document_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 预处理文档并创建TaggedDocument
            tagged_docs = []
            for i, doc in enumerate(documents):
                if doc and doc.strip():  # 确保文档不为空
                    processed_words = self._preprocess_text(doc)
                    if processed_words:  # 确保预处理后的词汇不为空
                        tagged_docs.append(TaggedDocument(words=processed_words, tags=[document_ids[i]]))
            
            if not tagged_docs:
                raise ValueError("没有有效的文档可用于训练")
            
            # 初始化并训练Doc2vec模型
            self.model = Doc2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                epochs=self.epochs,
                dm=1  # 使用PV-DM算法
            )
            
            # 构建词汇表
            self.model.build_vocab(tagged_docs)
            
            # 训练模型
            self.model.train(tagged_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            
            logger.info(f"Doc2vec模型训练完成，文档数量: {len(tagged_docs)}, 向量维度: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"训练Doc2vec模型时发生错误: {e}")
            raise
    
    def get_document_vector(self, text: str) -> Optional[np.ndarray]:
        """获取文档向量
        
        Args:
            text (str): 输入文本
            
        Returns:
            Optional[np.ndarray]: 文档向量，如果失败则返回None
        """
        if self.model is None:
            logger.error("模型尚未训练，请先调用train_model方法")
            return None
        
        try:
            # 预处理文本
            processed_words = self._preprocess_text(text)
            
            if not processed_words:
                logger.warning("预处理后的文本为空")
                return None
            
            # 推理文档向量
            vector = self.model.infer_vector(processed_words)
            return vector
            
        except Exception as e:
            logger.error(f"获取文档向量时发生错误: {e}")
            return None


class SimilarityCalculator:
    """相似度计算器类
    
    基于文档向量计算余弦相似度。
    """
    
    @staticmethod
    def cosine_similarity_vectors(vector1: Optional[np.ndarray], vector2: Optional[np.ndarray]) -> float:
        """计算两个向量的余弦相似度
        
        基于公式：Sim = (v1 · v2) / (||v1|| * ||v2||)
        
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
            v1 = np.array(vector1).reshape(1, -1)
            v2 = np.array(vector2).reshape(1, -1)
            
            # 检查向量维度是否匹配
            if v1.shape[1] != v2.shape[1]:
                logger.error(f"向量维度不匹配: {v1.shape[1]} vs {v2.shape[1]}")
                return 0.0
            
            # 计算余弦相似度
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
    def text_similarity(text1: str, text2: str, vectorizer: TextVectorizer) -> float:
        """计算两个文本的相似度
        
        Args:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            vectorizer (TextVectorizer): 已训练的向量化器
            
        Returns:
            float: 文本相似度，范围[-1, 1]
        """
        try:
            # 获取文档向量
            vector1 = vectorizer.get_document_vector(text1)
            vector2 = vectorizer.get_document_vector(text2)
            
            # 计算相似度
            return SimilarityCalculator.cosine_similarity_vectors(vector1, vector2)
            
        except Exception as e:
            logger.error(f"计算文本相似度时发生错误: {e}")
            return 0.0


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