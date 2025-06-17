#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感词典合并处理模块

本模块用于合并RFS词典和中文金融情感词典，去除重复词汇并保存为统一格式。
基于论文中提到的综合情感词典构建方法实现。

Authors: 论文作者团队
Date: 2024
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmotionDictProcessor:
    """情感词典处理器类
    
    用于合并和处理多个情感词典，生成统一的情感词典文件。
    """
    
    def __init__(self, lm_dict_path: str, rfs_dict_path: str, output_path: str):
        """初始化情感词典处理器
        
        Args:
            lm_dict_path (str): 中文金融情感词典文件路径
            rfs_dict_path (str): RFS词表文件路径  
            output_path (str): 输出文件路径
        """
        self.lm_dict_path = Path(lm_dict_path)
        self.rfs_dict_path = Path(rfs_dict_path)
        self.output_path = Path(output_path)
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _validate_input_files(self) -> bool:
        """验证输入文件是否存在
        
        Returns:
            bool: 所有输入文件都存在返回True，否则返回False
        """
        if not self.lm_dict_path.exists():
            logger.error(f"中文金融情感词典文件不存在: {self.lm_dict_path}")
            return False
        
        if not self.rfs_dict_path.exists():
            logger.error(f"RFS词表文件不存在: {self.rfs_dict_path}")
            return False
        
        return True
    
    def _load_lm_dict(self) -> Tuple[pd.Series, pd.Series]:
        """加载中文金融情感词典
        
        Returns:
            Tuple[pd.Series, pd.Series]: (积极词Series, 消极词Series)
        """
        try:
            # 读取Excel文件的所有工作表
            excel_file = pd.ExcelFile(self.lm_dict_path)
            logger.info(f"Excel文件包含工作表: {excel_file.sheet_names}")
            
            # 初始化积极词和消极词列表
            pos_words = pd.Series(dtype=str)
            neg_words = pd.Series(dtype=str)
            
            # 查找并读取积极词工作表
            positive_sheet_names = ['positive', 'Positive', 'POSITIVE', '积极词', '正面词']
            for sheet_name in excel_file.sheet_names:
                if any(pos_name in sheet_name.lower() for pos_name in ['positive', 'pos']): # type: ignore
                    df_pos = pd.read_excel(self.lm_dict_path, sheet_name=sheet_name)
                    # 获取第一列数据作为积极词
                    if len(df_pos.columns) > 0:
                        pos_words = df_pos.iloc[:, 0].dropna()
                        logger.info(f"从工作表 '{sheet_name}' 加载积极词: {len(pos_words)}个")
                    break
            
            # 查找并读取消极词工作表  
            negative_sheet_names = ['negative', 'Negative', 'NEGATIVE', '消极词', '负面词']
            for sheet_name in excel_file.sheet_names:
                if any(neg_name in sheet_name.lower() for neg_name in ['negative', 'neg']): # type: ignore
                    df_neg = pd.read_excel(self.lm_dict_path, sheet_name=sheet_name)
                    # 获取第一列数据作为消极词
                    if len(df_neg.columns) > 0:
                        neg_words = df_neg.iloc[:, 0].dropna()
                        logger.info(f"从工作表 '{sheet_name}' 加载消极词: {len(neg_words)}个")
                    break
            
            # 如果没有找到对应的工作表，尝试读取第一个非介绍性工作表
            if len(pos_words) == 0 and len(neg_words) == 0:
                logger.warning("未找到明确的积极词/消极词工作表，尝试读取所有工作表")
                
                for sheet_name in excel_file.sheet_names:
                    # 跳过可能的介绍工作表
                    if any(intro in sheet_name.lower() for intro in ['介绍', 'intro', 'readme', '说明']): # type: ignore
                        continue
                    
                    df = pd.read_excel(self.lm_dict_path, sheet_name=sheet_name)
                    if len(df.columns) > 0 and len(df) > 0:
                        # 检查列名来判断是积极词还是消极词
                        first_col_name = str(df.columns[0]).lower()
                        if any(pos_name in first_col_name for pos_name in ['positive', 'pos', '积极', '正面']):
                            pos_words = df.iloc[:, 0].dropna()
                            logger.info(f"从工作表 '{sheet_name}' 加载积极词: {len(pos_words)}个")
                        elif any(neg_name in first_col_name for neg_name in ['negative', 'neg', '消极', '负面']):
                            neg_words = df.iloc[:, 0].dropna()
                            logger.info(f"从工作表 '{sheet_name}' 加载消极词: {len(neg_words)}个")
            
            if len(pos_words) == 0:
                logger.warning("未找到积极词数据")
            if len(neg_words) == 0:
                logger.warning("未找到消极词数据")
                
            logger.info(f"中文金融情感词典加载完成，积极词: {len(pos_words)}个，消极词: {len(neg_words)}个")
            return pos_words, neg_words
            
        except Exception as e:
            logger.error(f"加载中文金融情感词典失败: {e}")
            raise
    
    def _load_rfs_dict(self) -> Tuple[pd.Series, pd.Series]:
        """加载RFS词表
        
        Returns:
            Tuple[pd.Series, pd.Series]: (积极词Series, 消极词Series)
        """
        try:
            # 读取CSV文件，第一列为积极词(pos)，第二列为消极词(neg)
            df = pd.read_csv(self.rfs_dict_path, encoding='utf-8')
            
            # 获取积极词和消极词，去除NaN值
            pos_words = df['pos'].dropna()
            neg_words = df['neg'].dropna()
            
            logger.info(f"RFS词表加载完成，积极词: {len(pos_words)}个，消极词: {len(neg_words)}个")
            return pos_words, neg_words
            
        except Exception as e:
            logger.error(f"加载RFS词表失败: {e}")
            raise
    
    def _clean_and_merge_words(self, *word_series: pd.Series) -> pd.Series:
        """清理和合并词汇序列
        
        Args:
            *word_series: 多个pandas Series对象
            
        Returns:
            pd.Series: 清理和去重后的词汇序列
        """
        # 合并所有Series
        merged_words = pd.concat(word_series, ignore_index=True)
        
        # 去除换行符和空白字符
        merged_words = merged_words.astype(str).str.replace('\n', '', regex=True)
        merged_words = merged_words.str.strip()
        
        # 去除空值和重复值
        merged_words = merged_words[merged_words != ''].drop_duplicates().reset_index(drop=True)
        
        return merged_words
    
    def process_emotion_dict(self) -> Optional[str]:
        """处理情感词典合并
        
        Returns:
            Optional[str]: 成功返回输出文件路径，失败返回None
        """
        logger.info("开始处理情感词典合并...")
        
        # 验证输入文件
        if not self._validate_input_files():
            return None
        
        try:
            # 加载两个词典
            lm_pos, lm_neg = self._load_lm_dict()
            rfs_pos, rfs_neg = self._load_rfs_dict()
            
            # 检查是否成功加载到数据
            if len(lm_pos) == 0 and len(lm_neg) == 0:
                logger.error("中文金融情感词典未加载到任何数据")
                return None
            
            if len(rfs_pos) == 0 and len(rfs_neg) == 0:
                logger.error("RFS词典未加载到任何数据")
                return None
            
            # 合并积极词和消极词
            pos_merged = self._clean_and_merge_words(lm_pos, rfs_pos)
            neg_merged = self._clean_and_merge_words(lm_neg, rfs_neg)
            
            # 检查合并结果
            if len(pos_merged) == 0 and len(neg_merged) == 0:
                logger.error("词典合并后未得到任何有效数据")
                return None
            
            # 创建输出数据框
            max_len = max(len(pos_merged), len(neg_merged))
            
            # 补齐长度不足的序列
            pos_padded = pd.concat([pos_merged, pd.Series([None] * (max_len - len(pos_merged)))], ignore_index=True)
            neg_padded = pd.concat([neg_merged, pd.Series([None] * (max_len - len(neg_merged)))], ignore_index=True)
            
            output_df = pd.DataFrame({
                'positive': pos_padded,
                'negative': neg_padded
            })
            
            # 保存为CSV文件
            output_df.to_csv(self.output_path, index=False, encoding='utf-8')
            
            logger.info(f"情感词典合并完成，积极词: {len(pos_merged)}个，消极词: {len(neg_merged)}个")
            logger.info(f"合并后的词典已保存至: {self.output_path}")
            
            return str(self.output_path)
            
        except Exception as e:
            logger.error(f"处理情感词典时发生错误: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None


def merge_emotion_dicts(lm_dict_path: str = "data/original_data/emo_dict/lm_dict/中文金融情感词典_姜富伟等(2020).xlsx",
                       rfs_dict_path: str = "data/original_data/emo_dict/rfs_dict/RFS词表.csv",
                       output_path: str = "data/processed_data/emo_dict.csv") -> Optional[str]:
    """合并情感词典的便利函数
    
    Args:
        lm_dict_path (str): 中文金融情感词典文件路径
        rfs_dict_path (str): RFS词表文件路径
        output_path (str): 输出文件路径
        
    Returns:
        Optional[str]: 成功返回输出文件路径，失败返回None
    """
    processor = EmotionDictProcessor(lm_dict_path, rfs_dict_path, output_path)
    return processor.process_emotion_dict()


if __name__ == "__main__":
    # 测试代码
    result = merge_emotion_dicts()
    if result:
        print(f"情感词典合并成功，输出文件: {result}")
    else:
        print("情感词典合并失败")