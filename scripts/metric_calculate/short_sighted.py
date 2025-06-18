#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管理层短视指标计算模块

本模块用于计算上市公司年报管理层文本的短视指标，包括：
1. 基于TF-IDF的单个短视词语指标（43个）
2. 综合短视指标（1个）

实现方法：
- 采用TF-IDF方法对每个短视词语进行量化表示
- 为避免词频为0导致分子为0，计算时将所有词语词频加1
- 计算全部短视词汇总词频占文本全部词频的比例作为综合指标
"""

import os
import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter
from math import log
from tqdm import tqdm

# 导入工具函数
from utils import TextPreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShortSightedCalculator:
    """管理层短视指标计算器
    
    负责计算基于TF-IDF的短视词语指标和综合短视指标。
    """
    
    def __init__(self, 
                 short_sighted_dict_path: str = "data/original_data/short_sighted/dict.csv",
                 stopwords_path: str = "data/processed_data/stop_words.txt",
                 input_data_path: str = "data/processed_data/origin_with_textmetric.csv",
                 output_data_path: str = "data/processed_data/origin_with_textmetric_short_sighted.csv"):
        """初始化短视指标计算器
        
        Args:
            short_sighted_dict_path (str): 短视词典文件路径
            stopwords_path (str): 停用词文件路径
            input_data_path (str): 输入数据文件路径
            output_data_path (str): 输出数据文件路径
        """
        self.short_sighted_dict_path = Path(short_sighted_dict_path)
        self.stopwords_path = Path(stopwords_path)
        self.input_data_path = Path(input_data_path)
        self.output_data_path = Path(output_data_path)
        
        # 初始化组件
        self.preprocessor: Optional[TextPreprocessor] = None
        self.short_sighted_words: List[str] = []
        self.data: Optional[pd.DataFrame] = None
        self.management_texts: Dict[Tuple[str, int], str] = {}
        
        # 确保输出目录存在
        self.output_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("管理层短视指标计算器初始化完成")
    
    def initialize_components(self) -> None:
        """初始化计算组件"""
        try:
            logger.info("正在初始化计算组件...")
            
            # 初始化文本预处理器
            self.preprocessor = TextPreprocessor(str(self.stopwords_path))
            logger.info("✅ 文本预处理器初始化完成")
            
            logger.info("🎉 所有计算组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 初始化计算组件时发生错误: {e}")
            raise
    
    def load_short_sighted_dict(self) -> List[str]:
        """加载短视词典
        
        Returns:
            List[str]: 短视词汇列表
        """
        try:
            logger.info(f"正在加载短视词典: {self.short_sighted_dict_path}")
            
            if not self.short_sighted_dict_path.exists():
                raise FileNotFoundError(f"短视词典文件不存在: {self.short_sighted_dict_path}")
            
            # 读取CSV文件
            dict_df = pd.read_csv(self.short_sighted_dict_path, encoding='utf-8')
            
            # 检查列名
            if 'short_sighted' not in dict_df.columns:
                raise ValueError(f"短视词典文件中没有找到 'short_sighted' 列")
            
            # 提取短视词汇，去除空值
            self.short_sighted_words = dict_df['short_sighted'].dropna().tolist()
            
            # 清理词汇（去除空白字符）
            self.short_sighted_words = [word.strip() for word in self.short_sighted_words if word.strip()]
            
            logger.info(f"✅ 短视词典加载完成，共 {len(self.short_sighted_words)} 个词汇")
            logger.info(f"短视词汇样例: {self.short_sighted_words[:10]}")
            
            return self.short_sighted_words
            
        except Exception as e:
            logger.error(f"❌ 加载短视词典时发生错误: {e}")
            raise
    
    def load_input_data(self) -> pd.DataFrame:
        """加载输入数据
        
        Returns:
            pd.DataFrame: 输入数据
        """
        try:
            logger.info(f"正在加载输入数据: {self.input_data_path}")
            
            if not self.input_data_path.exists():
                raise FileNotFoundError(f"输入数据文件不存在: {self.input_data_path}")
            
            # 读取CSV文件
            self.data = pd.read_csv(self.input_data_path, encoding='utf-8')
            
            logger.info(f"✅ 输入数据加载完成，共 {len(self.data)} 行，{len(self.data.columns)} 列")
            
            return self.data
            
        except Exception as e:
            logger.error(f"❌ 加载输入数据时发生错误: {e}")
            raise
    
    def load_management_texts(self) -> Dict[Tuple[str, int], str]:
        """加载管理层文本数据
        
        Returns:
            Dict[Tuple[str, int], str]: 以(股票代码, 年份)为键的管理层文本字典
        """
        try:
            logger.info("开始加载管理层文本数据...")
            
            mgmt_file_path = Path("data/original_data/text_data/2001-2020年中国上市公司.管理层讨论与分析.年报文本.xlsx")
            if not mgmt_file_path.exists():
                logger.error(f"管理层文本文件不存在: {mgmt_file_path}")
                return {}
            
            # 读取管理层文本数据
            mgmt_data = pd.read_excel(mgmt_file_path, engine='openpyxl')
            logger.info(f"管理层文本数据加载完成，共 {len(mgmt_data)} 行")
            
            # 过滤2001-2020年的数据
            valid_data = mgmt_data[(mgmt_data['会计年度'] >= 2001) & (mgmt_data['会计年度'] <= 2020)]
            valid_data = valid_data.dropna(subset=['股票代码', '会计年度', '经营讨论与分析内容'])
            valid_data = valid_data[valid_data['经营讨论与分析内容'].astype(str).str.strip() != '']
            valid_data = valid_data[valid_data['经营讨论与分析内容'].astype(str) != 'nan']
            
            logger.info(f"有效的管理层文本数据: {len(valid_data)} 条")
            
            # 整理文本数据
            with tqdm(enumerate(valid_data.iterrows()), 
                     desc="整理管理层文本", 
                     unit="条", 
                     total=len(valid_data)) as pbar:
                
                for row_num, (idx, row) in pbar:
                    try:
                        stock_code = str(row['股票代码']).strip()
                        year = int(row['会计年度'])
                        text_content = str(row['经营讨论与分析内容']).strip()
                        
                        pbar.set_postfix(code=stock_code[:6], year=year)
                        
                        self.management_texts[(stock_code, year)] = text_content
                        
                    except Exception as e:
                        logger.error(f"整理管理层文本第 {row_num + 1} 行时发生错误: {e}")
                        continue
            
            logger.info(f"🎉 管理层文本整理完成，共 {len(self.management_texts)} 条记录")
            return self.management_texts
            
        except Exception as e:
            logger.error(f"❌ 加载管理层文本时发生错误: {e}")
            return {}
    
    def calculate_tf_idf_for_documents(self, 
                                     documents: Dict[Tuple[str, int], str]) -> Dict[Tuple[str, int], Dict[str, float]]:
        """计算所有文档的TF-IDF指标
        
        Args:
            documents (Dict[Tuple[str, int], str]): 文档字典
            
        Returns:
            Dict[Tuple[str, int], Dict[str, float]]: TF-IDF指标字典
        """
        try:
            logger.info("开始计算TF-IDF指标...")
            
            if self.preprocessor is None:
                raise ValueError("文本预处理器未初始化")
            
            # 第一步：预处理所有文档，获取词频统计
            logger.info("第一步：预处理文档并统计词频...")
            document_word_counts = {}  # 每个文档的词频统计
            document_total_words = {}  # 每个文档的总词数
            global_word_doc_counts = Counter()  # 全局词汇在多少个文档中出现
            
            with tqdm(documents.items(), desc="预处理文档", unit="文档") as pbar:
                for doc_key, text in pbar:
                    stock_code, year = doc_key
                    pbar.set_postfix(code=stock_code[:6], year=year)
                    
                    # 获取分词结果
                    words = self.preprocessor.get_word_list(text)
                    
                    if not words:
                        document_word_counts[doc_key] = {}
                        document_total_words[doc_key] = 0
                        continue
                    
                    # 统计词频
                    word_counts = Counter(words)
                    document_word_counts[doc_key] = word_counts
                    document_total_words[doc_key] = len(words)
                    
                    # 统计包含每个词的文档数量
                    for word in word_counts:
                        global_word_doc_counts[word] += 1
            
            logger.info(f"预处理完成，共处理 {len(documents)} 个文档")
            logger.info(f"全局词汇总数: {len(global_word_doc_counts)}")
            
            # 第二步：计算每个文档的TF-IDF指标
            logger.info("第二步：计算TF-IDF指标...")
            total_documents = len(documents)
            tf_idf_results = {}
            
            with tqdm(documents.items(), desc="计算TF-IDF", unit="文档") as pbar:
                for doc_key, text in pbar:
                    stock_code, year = doc_key
                    pbar.set_postfix(code=stock_code[:6], year=year)
                    
                    doc_word_counts = document_word_counts.get(doc_key, {})
                    doc_total_words = document_total_words.get(doc_key, 0)
                    
                    if doc_total_words == 0:
                        # 空文档，所有指标为0
                        tf_idf_results[doc_key] = {word: 0.0 for word in self.short_sighted_words}
                        continue
                    
                    # 计算每个短视词的TF-IDF
                    doc_tf_idf = {}
                    
                    for word in self.short_sighted_words:
                        # 计算TF（词频加1避免0值）
                        word_count = doc_word_counts.get(word, 0)
                        tf = (word_count + 1) / doc_total_words
                        
                        # 计算IDF
                        docs_containing_word = global_word_doc_counts.get(word, 0)
                        idf = log(total_documents / (docs_containing_word + 1))
                        
                        # 计算TF-IDF
                        tf_idf = tf * idf
                        doc_tf_idf[word] = tf_idf
                    
                    tf_idf_results[doc_key] = doc_tf_idf
            
            logger.info("🎉 TF-IDF计算完成")
            return tf_idf_results
            
        except Exception as e:
            logger.error(f"❌ 计算TF-IDF时发生错误: {e}")
            raise
    
    def calculate_comprehensive_short_sighted_indicator(self, 
                                                      documents: Dict[Tuple[str, int], str]) -> Dict[Tuple[str, int], float]:
        """计算综合短视指标
        
        Args:
            documents (Dict[Tuple[str, int], str]): 文档字典
            
        Returns:
            Dict[Tuple[str, int], float]: 综合短视指标字典
        """
        try:
            logger.info("开始计算综合短视指标...")
            
            if self.preprocessor is None:
                raise ValueError("文本预处理器未初始化")
            
            comprehensive_indicators = {}
            
            with tqdm(documents.items(), desc="计算综合短视指标", unit="文档") as pbar:
                for doc_key, text in pbar:
                    stock_code, year = doc_key
                    pbar.set_postfix(code=stock_code[:6], year=year)
                    
                    # 获取分词结果
                    words = self.preprocessor.get_word_list(text)
                    
                    if not words:
                        comprehensive_indicators[doc_key] = 0.0
                        continue
                    
                    # 统计词频
                    word_counts = Counter(words)
                    total_words = len(words)
                    
                    # 统计所有短视词的总出现次数
                    total_short_sighted_count = 0
                    for word in self.short_sighted_words:
                        total_short_sighted_count += word_counts.get(word, 0)
                    
                    # 计算综合指标（短视词总频次 / 文本总词数 * 100）
                    comprehensive_indicator = (total_short_sighted_count / total_words) * 100
                    comprehensive_indicators[doc_key] = comprehensive_indicator
            
            logger.info("🎉 综合短视指标计算完成")
            return comprehensive_indicators
            
        except Exception as e:
            logger.error(f"❌ 计算综合短视指标时发生错误: {e}")
            raise
    
    def merge_short_sighted_indicators_with_data(self, 
                                               tf_idf_results: Dict[Tuple[str, int], Dict[str, float]],
                                               comprehensive_indicators: Dict[Tuple[str, int], float]) -> pd.DataFrame:
        """将短视指标与现有数据合并
        
        Args:
            tf_idf_results: TF-IDF指标结果
            comprehensive_indicators: 综合短视指标结果
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        try:
            logger.info("开始合并短视指标与现有数据...")
            
            if self.data is None:
                raise ValueError("输入数据未加载")
            
            # 复制现有数据
            merged_data = self.data.copy()
            
            # 初始化短视指标列（43个单词指标 + 1个综合指标）
            # 单词指标列名
            word_columns = [f"短视_{word}" for word in self.short_sighted_words]
            # 综合指标列名
            comprehensive_column = "短视_综合指标"
            
            # 初始化所有列为NaN
            for col in word_columns + [comprehensive_column]:
                merged_data[col] = np.nan
            
            # 合并指标
            logger.info("正在合并短视指标...")
            matched_count = 0
            
            with tqdm(merged_data.iterrows(), 
                     desc="合并短视指标", 
                     unit="行", 
                     total=len(merged_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        # 获取股票代码和年份
                        stock_code = str(row['股票代码']).strip()
                        
                        if pd.isna(row['统计截止日期_年份']):
                            continue
                        
                        year = int(row['统计截止日期_年份'])
                        pbar.set_postfix(code=stock_code[:6], year=year)
                        
                        # 查找匹配的短视指标
                        doc_key = (stock_code, year)
                        
                        # 合并TF-IDF指标
                        if doc_key in tf_idf_results:
                            tf_idf_scores = tf_idf_results[doc_key]
                            for i, word in enumerate(self.short_sighted_words):
                                col_name = word_columns[i]
                                merged_data.at[idx, col_name] = tf_idf_scores.get(word, 0.0)
                        
                        # 合并综合指标
                        if doc_key in comprehensive_indicators:
                            merged_data.at[idx, comprehensive_column] = comprehensive_indicators[doc_key]
                            matched_count += 1
                            
                    except Exception as e:
                        logger.warning(f"合并短视指标第 {idx} 行时发生错误: {e}")
                        continue
            
            logger.info(f"短视指标匹配成功: {matched_count} 条记录")
            
            # 统计合并结果
            logger.info("合并结果统计:")
            
            # 统计单词指标
            for i, word in enumerate(self.short_sighted_words[:5]):  # 只显示前5个作为示例
                col_name = word_columns[i]
                non_na_count = merged_data[col_name].notna().sum()
                total_count = len(merged_data)
                coverage = non_na_count / total_count * 100
                logger.info(f"{col_name}: {non_na_count}/{total_count} ({coverage:.1f}%)")
            
            # 统计综合指标
            comp_non_na_count = merged_data[comprehensive_column].notna().sum()
            comp_coverage = comp_non_na_count / len(merged_data) * 100
            logger.info(f"{comprehensive_column}: {comp_non_na_count}/{len(merged_data)} ({comp_coverage:.1f}%)")
            
            logger.info("🎉 短视指标与现有数据合并完成")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 合并短视指标与现有数据时发生错误: {e}")
            raise
    
    def save_results(self, merged_data: pd.DataFrame) -> None:
        """保存合并后的结果
        
        Args:
            merged_data (pd.DataFrame): 合并后的数据
        """
        try:
            logger.info(f"正在保存结果到: {self.output_data_path}")
            
            # 保存为CSV文件
            merged_data.to_csv(self.output_data_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ 结果保存完成")
            logger.info(f"文件大小: {self.output_data_path.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info(f"数据形状: {merged_data.shape}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果时发生错误: {e}")
            raise
    
    def run_complete_analysis(self) -> pd.DataFrame:
        """运行完整的短视指标分析流程
        
        Returns:
            pd.DataFrame: 包含短视指标的完整数据
        """
        try:
            logger.info("🚀 开始运行完整的短视指标分析流程")
            
            # 1. 初始化组件
            self.initialize_components()
            
            # 2. 加载短视词典
            self.load_short_sighted_dict()
            
            # 3. 加载输入数据
            self.load_input_data()
            
            # 4. 加载管理层文本数据
            management_texts = self.load_management_texts()
            
            if not management_texts:
                logger.warning("未加载到管理层文本数据，将创建空的短视指标列")
                # 创建空的指标结果
                tf_idf_results = {}
                comprehensive_indicators = {}
            else:
                # 5. 计算TF-IDF指标
                tf_idf_results = self.calculate_tf_idf_for_documents(management_texts)
                
                # 6. 计算综合短视指标
                comprehensive_indicators = self.calculate_comprehensive_short_sighted_indicator(management_texts)
            
            # 7. 合并指标与现有数据
            merged_data = self.merge_short_sighted_indicators_with_data(
                tf_idf_results, comprehensive_indicators
            )
            
            # 8. 保存结果
            self.save_results(merged_data)
            
            logger.info("🎉 短视指标分析流程完成！")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 运行短视指标分析流程时发生错误: {e}")
            raise


def main():
    """主函数 - 用于直接运行此模块时的测试"""
    try:
        calculator = ShortSightedCalculator()
        result_data = calculator.run_complete_analysis()
        
        print(f"\n📊 分析结果概览:")
        print(f"数据形状: {result_data.shape}")
        
        # 显示短视指标的统计信息
        short_sighted_columns = [col for col in result_data.columns if col.startswith('短视_')]
        
        print(f"\n短视指标列统计:")
        print(f"短视指标列数: {len(short_sighted_columns)}")
        
        # 显示部分列的统计信息
        for col in short_sighted_columns[:5]:  # 显示前5个作为示例
            non_na_count = result_data[col].notna().sum()
            mean_val = result_data[col].mean()
            print(f"{col}: {non_na_count} 条有效数据, 均值: {mean_val:.6f}")
        
        # 显示综合指标
        comp_col = "短视_综合指标"
        if comp_col in result_data.columns:
            comp_non_na_count = result_data[comp_col].notna().sum()
            comp_mean_val = result_data[comp_col].mean()
            print(f"\n{comp_col}: {comp_non_na_count} 条有效数据, 均值: {comp_mean_val:.6f}")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        raise


if __name__ == "__main__":
    main()