#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
停用词合并处理模块

本模块用于合并多个停用词文件，去除重复词汇并保存为统一格式。
基于论文中使用的综合停用词处理方法实现。

Authors: 论文作者团队
Date: 2024
"""

import os
from pathlib import Path
from typing import List, Set, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StopWordsProcessor:
    """停用词处理器类
    
    用于合并和处理多个停用词文件，生成统一的停用词文件。
    """
    
    def __init__(self, input_dir: str, output_path: str):
        """初始化停用词处理器
        
        Args:
            input_dir (str): 停用词文件所在目录
            output_path (str): 输出文件路径
        """
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 预定义需要合并的停用词文件名
        self.stopword_files = [
            "baidu_stopwords.txt",
            "cn_stopwords.txt", 
            "hit_stopwords.txt",
            "scu_stopwords.txt"
        ]
    
    def _validate_input_files(self) -> List[Path]:
        """验证输入文件是否存在
        
        Returns:
            List[Path]: 存在的文件路径列表
        """
        existing_files = []
        missing_files = []
        
        for filename in self.stopword_files:
            file_path = self.input_dir / filename
            if file_path.exists():
                existing_files.append(file_path)
                logger.info(f"找到停用词文件: {file_path}")
            else:
                missing_files.append(file_path)
                logger.warning(f"停用词文件不存在: {file_path}")
        
        if missing_files:
            logger.warning(f"以下文件不存在，将跳过: {[str(f) for f in missing_files]}")
        
        if not existing_files:
            logger.error("未找到任何停用词文件")
            raise FileNotFoundError("未找到任何停用词文件")
        
        return existing_files
    
    def _load_stopwords_from_file(self, file_path: Path) -> Set[str]:
        """从单个文件加载停用词
        
        Args:
            file_path (Path): 停用词文件路径
            
        Returns:
            Set[str]: 停用词集合
        """
        stopwords = set()
        
        try:
            # 尝试多种编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            # 去除行首行尾空白字符
                            word = line.strip()
                            # 跳过空行和注释行
                            if word and not word.startswith('#'):
                                stopwords.add(word)
                    logger.info(f"成功加载停用词文件 {file_path} (编码: {encoding})，词汇数: {len(stopwords)}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"无法读取文件 {file_path}，尝试了所有编码方式")
                
        except Exception as e:
            logger.error(f"加载停用词文件 {file_path} 时发生错误: {e}")
        
        return stopwords
    
    def _merge_all_stopwords(self, file_paths: List[Path]) -> Set[str]:
        """合并所有停用词文件
        
        Args:
            file_paths (List[Path]): 停用词文件路径列表
            
        Returns:
            Set[str]: 合并后的停用词集合
        """
        merged_stopwords = set()
        
        for file_path in file_paths:
            file_stopwords = self._load_stopwords_from_file(file_path)
            merged_stopwords.update(file_stopwords)
            logger.info(f"已合并文件 {file_path.name}，当前总词数: {len(merged_stopwords)}")
        
        return merged_stopwords
    
    def _clean_stopwords(self, stopwords: Set[str]) -> Set[str]:
        """清理停用词
        
        Args:
            stopwords (Set[str]): 原始停用词集合
            
        Returns:
            Set[str]: 清理后的停用词集合
        """
        cleaned_stopwords = set()
        
        for word in stopwords:
            # 去除多余的空白字符
            cleaned_word = word.strip()
            
            # 跳过空字符串和单纯的标点符号
            if cleaned_word and len(cleaned_word) > 0:
                cleaned_stopwords.add(cleaned_word)
        
        logger.info(f"停用词清理完成，清理前: {len(stopwords)}个，清理后: {len(cleaned_stopwords)}个")
        return cleaned_stopwords
    
    def _save_stopwords(self, stopwords: Set[str]) -> None:
        """保存停用词到文件
        
        Args:
            stopwords (Set[str]): 停用词集合
        """
        try:
            # 按字母顺序排序
            sorted_stopwords = sorted(stopwords)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for word in sorted_stopwords:
                    f.write(word + '\n')
            
            logger.info(f"停用词文件已保存至: {self.output_path}，共 {len(stopwords)} 个词")
            
        except Exception as e:
            logger.error(f"保存停用词文件时发生错误: {e}")
            raise
    
    def process_stopwords(self) -> Optional[str]:
        """处理停用词合并
        
        Returns:
            Optional[str]: 成功返回输出文件路径，失败返回None
        """
        logger.info("开始处理停用词合并...")
        
        try:
            # 验证输入文件
            existing_files = self._validate_input_files()
            
            # 合并所有停用词
            merged_stopwords = self._merge_all_stopwords(existing_files)
            
            # 清理停用词
            cleaned_stopwords = self._clean_stopwords(merged_stopwords)
            
            # 保存到文件
            self._save_stopwords(cleaned_stopwords)
            
            logger.info("停用词合并处理完成")
            return str(self.output_path)
            
        except Exception as e:
            logger.error(f"处理停用词时发生错误: {e}")
            return None


def merge_stopwords(input_dir: str = "data/original_data/stop_words",
                   output_path: str = "data/processed_data/stop_words.txt") -> Optional[str]:
    """合并停用词的便利函数
    
    Args:
        input_dir (str): 停用词文件所在目录
        output_path (str): 输出文件路径
        
    Returns:
        Optional[str]: 成功返回输出文件路径，失败返回None
    """
    processor = StopWordsProcessor(input_dir, output_path)
    return processor.process_stopwords()


def load_merged_stopwords(stopwords_path: str = "data/processed_data/stop_words.txt") -> Set[str]:
    """加载合并后的停用词
    
    Args:
        stopwords_path (str): 停用词文件路径
        
    Returns:
        Set[str]: 停用词集合
    """
    stopwords = set()
    
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        
        logger.info(f"成功加载停用词，共 {len(stopwords)} 个词")
        
    except FileNotFoundError:
        logger.error(f"停用词文件不存在: {stopwords_path}")
        logger.info("请先运行 merge_stopwords() 函数生成停用词文件")
        
    except Exception as e:
        logger.error(f"加载停用词时发生错误: {e}")
    
    return stopwords


if __name__ == "__main__":
    # 测试代码
    result = merge_stopwords()
    if result:
        print(f"停用词合并成功，输出文件: {result}")
        
        # 测试加载
        stopwords = load_merged_stopwords()
        print(f"加载停用词成功，共 {len(stopwords)} 个词")
    else:
        print("停用词合并失败")