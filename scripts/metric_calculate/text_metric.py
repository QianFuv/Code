#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本指标计算模块

本模块用于计算央行、政府和管理层文本的各项指标，包括：
1. 净语调 (TONE)
2. 负语调 (NTONE)  
3. 相似度 (SIMILARITY)
4. 可读性 (READABILITY)

处理的文本类型：
- 央行文本：季度货币政策执行报告
- 政府文本：省级政府工作报告
- 管理层文本：上市公司管理层讨论与分析

作者：根据论文要求实现
日期：2025年
"""

import os
import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm

# 导入工具函数
from utils import (
    TextPreprocessor,
    BGEVectorizer,
    SimilarityCalculator,
    ReadabilityCalculator,
    create_bge_vectorizer,
    create_sentiment_calculator_from_file
)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextMetricCalculator:
    """文本指标计算器
    
    负责计算央行、政府和管理层文本的各项指标，并与数值数据进行匹配。
    """
    
    def __init__(self, 
                 numeric_data_path: str = "data/original_data/numeric_data/2001-2020年制造业数值数据.xlsx",
                 emotion_dict_path: str = "data/processed_data/emo_dict.csv",
                 stopwords_path: str = "data/processed_data/stop_words.txt",
                 bge_model_name: str = "BAAI/bge-large-zh-v1.5"):
        """初始化文本指标计算器
        
        Args:
            numeric_data_path (str): 数值数据文件路径
            emotion_dict_path (str): 情感词典文件路径  
            stopwords_path (str): 停用词文件路径
            bge_model_name (str): BGE模型名称
        """
        self.numeric_data_path = Path(numeric_data_path)
        self.emotion_dict_path = Path(emotion_dict_path)
        self.stopwords_path = Path(stopwords_path)
        self.bge_model_name = bge_model_name
        
        # 初始化工具组件
        self.preprocessor = None
        self.vectorizer = None
        self.sentiment_calculator = None
        self.readability_calculator = None
        
        # 数据存储
        self.numeric_data = None
        self.text_metrics_data = None
        
        # 相似度计算的基准文本（用于央行和政府文本）
        self.baseline_texts = {
            'central_bank': "央行实施稳健货币政策，保持流动性合理充裕，支持实体经济发展。",
            'government': "政府坚持高质量发展，深化供给侧结构性改革，推进经济转型升级。"
        }
        
        logger.info("文本指标计算器初始化完成")
    
    def initialize_components(self) -> None:
        """初始化所有计算组件"""
        try:
            logger.info("正在初始化计算组件...")
            
            # 初始化文本预处理器
            self.preprocessor = TextPreprocessor(str(self.stopwords_path))
            logger.info("✅ 文本预处理器初始化完成")
            
            # 初始化BGE向量化器
            self.vectorizer = create_bge_vectorizer(
                model_name=self.bge_model_name,
                normalize_embeddings=True,
                use_instruction=False,
                preprocessor=self.preprocessor
            )
            if self.vectorizer is None:
                raise ValueError("BGE向量化器初始化失败")
            logger.info("✅ BGE向量化器初始化完成")
            
            # 初始化情感计算器
            self.sentiment_calculator = create_sentiment_calculator_from_file(
                str(self.emotion_dict_path), 
                self.preprocessor
            )
            if self.sentiment_calculator is None:
                raise ValueError("情感计算器初始化失败")
            logger.info("✅ 情感计算器初始化完成")
            
            # 初始化可读性计算器
            self.readability_calculator = ReadabilityCalculator(self.preprocessor)
            logger.info("✅ 可读性计算器初始化完成")
            
            logger.info("🎉 所有计算组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 初始化计算组件时发生错误: {e}")
            raise
    
    def load_numeric_data(self) -> pd.DataFrame:
        """加载数值数据
        
        Returns:
            pd.DataFrame: 数值数据
        """
        try:
            logger.info(f"正在加载数值数据: {self.numeric_data_path}")
            
            if not self.numeric_data_path.exists():
                raise FileNotFoundError(f"数值数据文件不存在: {self.numeric_data_path}")
            
            # 读取Excel文件
            self.numeric_data = pd.read_excel(self.numeric_data_path, engine='openpyxl')
            
            logger.info(f"✅ 数值数据加载完成，共 {len(self.numeric_data)} 行")
            logger.info(f"数据列名: {list(self.numeric_data.columns)}")
            
            return self.numeric_data
            
        except Exception as e:
            logger.error(f"❌ 加载数值数据时发生错误: {e}")
            raise
    
    def calculate_text_metrics(self, text: str, baseline_text: Optional[str] = None) -> Dict[str, float]:
        """计算单个文本的所有指标
        
        Args:
            text (str): 待分析文本
            baseline_text (Optional[str]): 用于相似度计算的基准文本
            
        Returns:
            Dict[str, float]: 包含所有指标的字典
        """
        try:
            if not text or not text.strip():
                return {
                    'tone': 0.0,
                    'negative_tone': 0.0,
                    'similarity': 0.0,
                    'readability': 0.0
                }
            
            # 计算情感指标
            sentiment_metrics = self.sentiment_calculator.calculate_all_metrics(text)
            tone = sentiment_metrics['tone']
            negative_tone = sentiment_metrics['negative_tone']
            
            # 计算相似度
            if baseline_text:
                similarity = SimilarityCalculator.text_similarity(
                    text, baseline_text, self.vectorizer
                )
            else:
                similarity = 0.0
            
            # 计算可读性（使用字符级别）
            readability = self.readability_calculator.calculate_readability(
                text, use_character_count=True
            )
            
            return {
                'tone': tone,
                'negative_tone': negative_tone,
                'similarity': similarity,
                'readability': readability
            }
            
        except Exception as e:
            logger.error(f"计算文本指标时发生错误: {e}")
            return {
                'tone': 0.0,
                'negative_tone': 0.0,
                'similarity': 0.0,
                'readability': 0.0
            }
    
    def process_central_bank_texts(self) -> Dict[int, Dict[str, float]]:
        """处理央行文本数据
        
        Returns:
            Dict[int, Dict[str, float]]: 以年份为键的央行文本指标字典
        """
        try:
            logger.info("开始处理央行文本数据...")
            
            central_bank_dir = Path("data/original_data/text_data/央行文本")
            if not central_bank_dir.exists():
                logger.error(f"央行文本目录不存在: {central_bank_dir}")
                return {}
            
            # 获取所有文本文件
            txt_files = list(central_bank_dir.glob("*.txt"))
            
            # 过滤2001-2020年的文件
            valid_files = []
            for txt_file in txt_files:
                filename = txt_file.stem
                year_match = re.match(r'(\d{4})第四季度', filename)
                if year_match:
                    year = int(year_match.group(1))
                    if 2001 <= year <= 2020:
                        valid_files.append((txt_file, year))
            
            if not valid_files:
                logger.warning("未找到有效的央行文本文件")
                return {}
            
            central_bank_metrics = {}
            baseline_text = self.baseline_texts['central_bank']
            
            # 使用tqdm显示进度
            with tqdm(valid_files, desc="处理央行文本", unit="文件") as pbar:
                for txt_file, year in pbar:
                    try:
                        pbar.set_postfix(year=year)
                        
                        # 读取文本内容
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text_content = f.read().strip()
                        
                        if not text_content:
                            logger.warning(f"央行文本文件为空: {txt_file}")
                            continue
                        
                        # 计算文本指标
                        metrics = self.calculate_text_metrics(text_content, baseline_text)
                        central_bank_metrics[year] = metrics
                        
                    except Exception as e:
                        logger.error(f"处理央行文本文件 {txt_file} 时发生错误: {e}")
                        continue
            
            logger.info(f"🎉 央行文本处理完成，共处理 {len(central_bank_metrics)} 年的数据")
            return central_bank_metrics
            
        except Exception as e:
            logger.error(f"❌ 处理央行文本时发生错误: {e}")
            return {}
    
    def process_government_texts(self) -> Dict[Tuple[str, int], Dict[str, float]]:
        """处理政府文本数据
        
        Returns:
            Dict[Tuple[str, int], Dict[str, float]]: 以(省份, 年份)为键的政府文本指标字典
        """
        try:
            logger.info("开始处理政府文本数据...")
            
            gov_file_path = Path("data/original_data/text_data/2001-2022省级工作政府报告.xlsx")
            if not gov_file_path.exists():
                logger.error(f"政府文本文件不存在: {gov_file_path}")
                return {}
            
            # 读取政府文本数据
            gov_data = pd.read_excel(gov_file_path, engine='openpyxl')
            logger.info(f"政府文本数据加载完成，共 {len(gov_data)} 行")
            
            # 过滤2001-2020年的数据
            valid_data = gov_data[(gov_data['会计年'] >= 2001) & (gov_data['会计年'] <= 2020)]
            valid_data = valid_data.dropna(subset=['省份名称', '会计年', '政府报告'])
            valid_data = valid_data[valid_data['政府报告'].str.strip() != '']
            
            logger.info(f"有效的政府文本数据: {len(valid_data)} 条")
            
            government_metrics = {}
            baseline_text = self.baseline_texts['government']
            
            # 使用tqdm显示进度
            with tqdm(valid_data.iterrows(), 
                     desc="处理政府文本", 
                     unit="条", 
                     total=len(valid_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        province = str(row['省份名称']).strip()
                        year = int(row['会计年'])
                        text_content = str(row['政府报告']).strip()
                        
                        pbar.set_postfix(province=province[:4], year=year)
                        
                        if not text_content or text_content == 'nan':
                            continue
                        
                        # 计算文本指标
                        metrics = self.calculate_text_metrics(text_content, baseline_text)
                        government_metrics[(province, year)] = metrics
                        
                    except Exception as e:
                        logger.error(f"处理政府文本第 {idx + 1} 行时发生错误: {e}")
                        continue
            
            logger.info(f"🎉 政府文本处理完成，共处理 {len(government_metrics)} 条数据")
            return government_metrics
            
        except Exception as e:
            logger.error(f"❌ 处理政府文本时发生错误: {e}")
            return {}
    
    def process_management_texts(self) -> Dict[Tuple[str, int], Dict[str, float]]:
        """处理管理层文本数据
        
        计算管理层相似度：同一公司本年度与上一年度文本的相似度
        
        Returns:
            Dict[Tuple[str, int], Dict[str, float]]: 以(股票代码, 年份)为键的管理层文本指标字典
        """
        try:
            logger.info("开始处理管理层文本数据...")
            
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
            
            # 先整理数据，按公司和年份组织
            logger.info("正在整理文本数据...")
            company_texts = {}  # {股票代码: {年份: 文本内容}}
            
            with tqdm(valid_data.iterrows(), 
                     desc="整理管理层文本", 
                     unit="条", 
                     total=len(valid_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        stock_code = str(row['股票代码']).strip()
                        year = int(row['会计年度'])
                        text_content = str(row['经营讨论与分析内容']).strip()
                        
                        pbar.set_postfix(code=stock_code[:6], year=year)
                        
                        if stock_code not in company_texts:
                            company_texts[stock_code] = {}
                        
                        company_texts[stock_code][year] = text_content
                        
                    except Exception as e:
                        logger.error(f"整理管理层文本第 {idx + 1} 行时发生错误: {e}")
                        continue
            
            logger.info(f"整理完成，共 {len(company_texts)} 家公司的文本数据")
            
            # 计算指标
            management_metrics = {}
            similarity_count = 0
            
            # 计算总的处理条目数
            total_items = sum(len(year_texts) for year_texts in company_texts.values())
            
            with tqdm(total=total_items, 
                     desc="计算管理层指标", 
                     unit="条") as pbar:
                
                for stock_code, year_texts in company_texts.items():
                    try:
                        for year, current_text in year_texts.items():
                            pbar.set_postfix(code=stock_code[:6], year=year)
                            
                            # 计算基本的情感和可读性指标
                            sentiment_metrics = self.sentiment_calculator.calculate_all_metrics(current_text)
                            tone = sentiment_metrics['tone']
                            negative_tone = sentiment_metrics['negative_tone']
                            
                            readability = self.readability_calculator.calculate_readability(
                                current_text, use_character_count=True
                            )
                            
                            # 计算与前一年的相似度
                            similarity = 0.0
                            previous_year = year - 1
                            
                            if previous_year in year_texts:
                                previous_text = year_texts[previous_year]
                                if previous_text and previous_text.strip():
                                    similarity = SimilarityCalculator.text_similarity(
                                        current_text, previous_text, self.vectorizer
                                    )
                                    if similarity > 0:
                                        similarity_count += 1
                            
                            # 存储指标
                            management_metrics[(stock_code, year)] = {
                                'tone': tone,
                                'negative_tone': negative_tone,
                                'similarity': similarity,
                                'readability': readability
                            }
                            
                            pbar.update(1)
                            
                    except Exception as e:
                        logger.error(f"处理公司 {stock_code} 管理层文本时发生错误: {e}")
                        # 仍然更新进度条
                        remaining_items = len(year_texts)
                        pbar.update(remaining_items)
                        continue
            
            logger.info(f"🎉 管理层文本处理完成，共处理 {len(management_metrics)} 条数据")
            logger.info(f"成功计算相似度的记录数: {similarity_count}/{len(management_metrics)}")
            
            return management_metrics
            
        except Exception as e:
            logger.error(f"❌ 处理管理层文本时发生错误: {e}")
            return {}
    
    def normalize_province_name(self, province_name: str) -> str:
        """标准化省份名称
        
        Args:
            province_name (str): 原始省份名称
            
        Returns:
            str: 标准化后的省份名称
        """
        if not province_name:
            return ""
        
        province_name = province_name.strip()
        
        # 如果省份名称不以"省"结尾，且不是直辖市或特别行政区，则添加"省"
        special_regions = {'北京', '上海', '天津', '重庆', '香港', '澳门', '台湾', 
                          '内蒙古', '广西', '西藏', '宁夏', '新疆'}
        
        if province_name not in special_regions and not province_name.endswith('省'):
            province_name += '省'
        
        return province_name
    
    def merge_text_metrics_with_numeric_data(self, 
                                           central_bank_metrics: Dict[int, Dict[str, float]],
                                           government_metrics: Dict[Tuple[str, int], Dict[str, float]],
                                           management_metrics: Dict[Tuple[str, int], Dict[str, float]]) -> pd.DataFrame:
        """将文本指标与数值数据合并
        
        Args:
            central_bank_metrics: 央行文本指标
            government_metrics: 政府文本指标  
            management_metrics: 管理层文本指标
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        try:
            logger.info("开始合并文本指标与数值数据...")
            
            # 复制数值数据
            merged_data = self.numeric_data.copy()
            
            # 检查必要的列是否存在
            required_columns = ['统计截止日期_年份', '所属省份', '股票代码']
            missing_columns = [col for col in required_columns if col not in merged_data.columns]
            
            if missing_columns:
                logger.error(f"数值数据中缺少必要的列: {missing_columns}")
                logger.info(f"现有列名: {list(merged_data.columns)}")
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 初始化文本指标列
            text_metric_columns = [
                # 央行指标
                '央行_净语调', '央行_负语调', '央行_相似度', '央行_可读性',
                # 政府指标
                '政府_净语调', '政府_负语调', '政府_相似度', '政府_可读性',
                # 管理层指标
                '管理层_净语调', '管理层_负语调', '管理层_相似度', '管理层_可读性'
            ]
            
            for col in text_metric_columns:
                merged_data[col] = np.nan
            
            # 合并央行指标
            logger.info("正在合并央行指标...")
            central_bank_matched = 0
            
            with tqdm(merged_data.iterrows(), 
                     desc="合并央行指标", 
                     unit="行", 
                     total=len(merged_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        # 从"统计截止日期_年份"提取年份
                        if pd.isna(row['统计截止日期_年份']):
                            continue
                        
                        year = int(row['统计截止日期_年份'])
                        pbar.set_postfix(year=year)
                        
                        if year in central_bank_metrics:
                            metrics = central_bank_metrics[year]
                            merged_data.loc[idx, '央行_净语调'] = metrics['tone']
                            merged_data.loc[idx, '央行_负语调'] = metrics['negative_tone']
                            merged_data.loc[idx, '央行_相似度'] = metrics['similarity']
                            merged_data.loc[idx, '央行_可读性'] = metrics['readability']
                            central_bank_matched += 1
                            
                    except Exception as e:
                        logger.warning(f"合并央行指标第 {idx} 行时发生错误: {e}")
                        continue
            
            logger.info(f"央行指标匹配成功: {central_bank_matched} 条记录")
            
            # 合并政府指标
            logger.info("正在合并政府指标...")
            government_matched = 0
            
            with tqdm(merged_data.iterrows(), 
                     desc="合并政府指标", 
                     unit="行", 
                     total=len(merged_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        # 获取省份和年份
                        province = str(row['所属省份']).strip()
                        
                        if pd.isna(row['统计截止日期_年份']):
                            continue
                        
                        year = int(row['统计截止日期_年份'])
                        pbar.set_postfix(province=province[:4], year=year)
                        
                        # 从省份名称中移除"省"字进行匹配
                        province_key = province.replace('省', '') if province.endswith('省') else province
                        
                        # 查找匹配的政府指标
                        gov_key = (province_key, year)
                        if gov_key in government_metrics:
                            metrics = government_metrics[gov_key]
                            merged_data.loc[idx, '政府_净语调'] = metrics['tone']
                            merged_data.loc[idx, '政府_负语调'] = metrics['negative_tone']
                            merged_data.loc[idx, '政府_相似度'] = metrics['similarity']
                            merged_data.loc[idx, '政府_可读性'] = metrics['readability']
                            government_matched += 1
                            
                    except Exception as e:
                        logger.warning(f"合并政府指标第 {idx} 行时发生错误: {e}")
                        continue
            
            logger.info(f"政府指标匹配成功: {government_matched} 条记录")
            
            # 合并管理层指标
            logger.info("正在合并管理层指标...")
            management_matched = 0
            
            with tqdm(merged_data.iterrows(), 
                     desc="合并管理层指标", 
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
                        
                        # 查找匹配的管理层指标
                        mgmt_key = (stock_code, year)
                        if mgmt_key in management_metrics:
                            metrics = management_metrics[mgmt_key]
                            merged_data.loc[idx, '管理层_净语调'] = metrics['tone']
                            merged_data.loc[idx, '管理层_负语调'] = metrics['negative_tone']
                            merged_data.loc[idx, '管理层_相似度'] = metrics['similarity']
                            merged_data.loc[idx, '管理层_可读性'] = metrics['readability']
                            management_matched += 1
                            
                    except Exception as e:
                        logger.warning(f"合并管理层指标第 {idx} 行时发生错误: {e}")
                        continue
            
            logger.info(f"管理层指标匹配成功: {management_matched} 条记录")
            
            # 统计合并结果
            logger.info("合并结果统计:")
            for col in text_metric_columns:
                non_na_count = merged_data[col].notna().sum()
                total_count = len(merged_data)
                coverage = non_na_count / total_count * 100
                logger.info(f"{col}: {non_na_count}/{total_count} ({coverage:.1f}%)")
            
            # 特别统计管理层相似度
            mgmt_similarity_count = merged_data['管理层_相似度'].notna().sum()
            mgmt_similarity_positive = (merged_data['管理层_相似度'] > 0).sum()
            logger.info(f"管理层相似度统计: 总计{mgmt_similarity_count}条, 其中{mgmt_similarity_positive}条相似度>0")
            
            logger.info("🎉 文本指标与数值数据合并完成")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 合并文本指标与数值数据时发生错误: {e}")
            raise
    
    def save_results(self, merged_data: pd.DataFrame, 
                    output_path: str = "data/processed_data/origin_with_textmetric.csv") -> None:
        """保存合并后的结果
        
        Args:
            merged_data (pd.DataFrame): 合并后的数据
            output_path (str): 输出文件路径
        """
        try:
            logger.info(f"正在保存结果到: {output_path}")
            
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为CSV文件
            merged_data.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ 结果保存完成")
            logger.info(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info(f"数据形状: {merged_data.shape}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果时发生错误: {e}")
            raise
    
    def run_complete_analysis(self) -> pd.DataFrame:
        """运行完整的文本指标分析流程
        
        Returns:
            pd.DataFrame: 包含文本指标的完整数据
        """
        try:
            logger.info("🚀 开始运行完整的文本指标分析流程")
            
            # 1. 初始化组件
            self.initialize_components()
            
            # 2. 加载数值数据
            self.load_numeric_data()
            
            # 3. 处理各类文本数据
            central_bank_metrics = self.process_central_bank_texts()
            government_metrics = self.process_government_texts()
            management_metrics = self.process_management_texts()
            
            # 4. 合并文本指标与数值数据
            merged_data = self.merge_text_metrics_with_numeric_data(
                central_bank_metrics, government_metrics, management_metrics
            )
            
            # 5. 保存结果
            self.save_results(merged_data)
            
            logger.info("🎉 文本指标分析流程完成！")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 运行文本指标分析流程时发生错误: {e}")
            raise


def main():
    """主函数 - 用于直接运行此模块时的测试"""
    try:
        calculator = TextMetricCalculator()
        result_data = calculator.run_complete_analysis()
        
        print(f"\n📊 分析结果概览:")
        print(f"数据形状: {result_data.shape}")
        print(f"\n列名: {list(result_data.columns)}")
        
        # 显示文本指标的统计信息
        text_columns = [col for col in result_data.columns 
                       if any(keyword in col for keyword in ['央行', '政府', '管理层'])]
        
        print(f"\n文本指标列统计:")
        for col in text_columns:
            non_na_count = result_data[col].notna().sum()
            mean_val = result_data[col].mean()
            print(f"{col}: {non_na_count} 条有效数据, 均值: {mean_val:.4f}")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        raise


if __name__ == "__main__":
    main()