#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
竞争策略指标计算模块

本模块用于处理竞争策略数据，将cost和diff指标与现有数据进行匹配合并。
主要功能包括：
1. 读取ATT00002.bin文件中的竞争策略数据
2. 提取cost和diff指标
3. 与现有数据按股票代码和年份进行匹配
4. 保存合并后的完整数据
"""

import os
import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompetitiveStrategyCalculator:
    """竞争策略指标计算器
    
    负责加载和处理竞争策略数据，将cost和diff指标与现有数据进行匹配合并。
    """
    
    def __init__(self, 
                 competitive_data_path: str = "data/original_data/competitive_strategy/ATT00002.bin",
                 input_data_path: str = "data/processed_data/origin_with_textmetric_short_sighted.csv",
                 output_data_path: str = "data/processed_data/unfilled_full.csv"):
        """初始化竞争策略指标计算器
        
        Args:
            competitive_data_path (str): 竞争策略数据文件路径（ATT00002.bin）
            input_data_path (str): 输入数据文件路径
            output_data_path (str): 输出数据文件路径
        """
        self.competitive_data_path = Path(competitive_data_path)
        self.input_data_path = Path(input_data_path)
        self.output_data_path = Path(output_data_path)
        
        # 数据存储
        self.competitive_data: Optional[pd.DataFrame] = None
        self.input_data: Optional[pd.DataFrame] = None
        
        # 确保输出目录存在
        self.output_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("竞争策略指标计算器初始化完成")
    
    def load_competitive_data(self) -> pd.DataFrame:
        """加载竞争策略数据
        
        从ATT00002.bin文件中读取竞争策略数据，该文件实际为CSV格式。
        
        Returns:
            pd.DataFrame: 竞争策略数据
        """
        try:
            logger.info(f"正在加载竞争策略数据: {self.competitive_data_path}")
            
            if not self.competitive_data_path.exists():
                raise FileNotFoundError(f"竞争策略数据文件不存在: {self.competitive_data_path}")
            
            # 尝试读取文件（.bin文件实际为CSV格式）
            try:
                # 先尝试UTF-8编码
                self.competitive_data = pd.read_csv(self.competitive_data_path, encoding='utf-8')
                logger.info("使用UTF-8编码成功读取文件")
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试GBK编码
                self.competitive_data = pd.read_csv(self.competitive_data_path, encoding='gbk')
                logger.info("使用GBK编码成功读取文件")
            except Exception:
                # 最后尝试ISO-8859-1编码
                self.competitive_data = pd.read_csv(self.competitive_data_path, encoding='iso-8859-1')
                logger.info("使用ISO-8859-1编码成功读取文件")
            
            logger.info(f"✅ 竞争策略数据加载完成，共 {len(self.competitive_data)} 行，{len(self.competitive_data.columns)} 列")
            logger.info(f"数据列名: {list(self.competitive_data.columns)}")
            
            # 检查必要的列是否存在
            required_columns = ['security_code', 'rep_period', 'cost', 'diff']
            missing_columns = [col for col in required_columns if col not in self.competitive_data.columns]
            
            if missing_columns:
                logger.error(f"竞争策略数据中缺少必要的列: {missing_columns}")
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 显示数据样例
            logger.info("数据样例（前5行）:")
            logger.info(f"\n{self.competitive_data.head()}")
            
            return self.competitive_data
            
        except Exception as e:
            logger.error(f"❌ 加载竞争策略数据时发生错误: {e}")
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
            self.input_data = pd.read_csv(self.input_data_path, encoding='utf-8')
            
            logger.info(f"✅ 输入数据加载完成，共 {len(self.input_data)} 行，{len(self.input_data.columns)} 列")
            
            # 检查必要的列是否存在
            required_columns = ['股票代码', '统计截止日期_年份']
            missing_columns = [col for col in required_columns if col not in self.input_data.columns]
            
            if missing_columns:
                logger.error(f"输入数据中缺少必要的列: {missing_columns}")
                logger.info(f"现有列名: {list(self.input_data.columns)}")
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            return self.input_data
            
        except Exception as e:
            logger.error(f"❌ 加载输入数据时发生错误: {e}")
            raise
    
    def extract_year_from_rep_period(self, rep_period: str) -> Optional[int]:
        """从报告期间字符串中提取年份
        
        Args:
            rep_period (str): 报告期间字符串，格式如 "2001-12-31"
            
        Returns:
            Optional[int]: 提取的年份，失败时返回None
        """
        try:
            if not rep_period or pd.isna(rep_period):
                return None
            
            # 使用正则表达式提取年份（前4位数字）
            year_match = re.match(r'(\d{4})', str(rep_period))
            if year_match:
                return int(year_match.group(1))
            
            # 如果正则表达式匹配失败，尝试解析日期
            try:
                date_obj = pd.to_datetime(rep_period)
                return date_obj.year
            except:
                pass
            
            logger.warning(f"无法从报告期间 '{rep_period}' 中提取年份")
            return None
            
        except Exception as e:
            logger.warning(f"提取年份时发生错误: {e}, rep_period: {rep_period}")
            return None
    
    def normalize_security_code(self, security_code: Union[str, int, float]) -> str:
        """标准化证券代码
        
        Args:
            security_code: 原始证券代码（可能是字符串、整数或浮点数）
            
        Returns:
            str: 标准化后的证券代码字符串
        """
        try:
            if pd.isna(security_code):
                return ""
            
            # 转换为字符串并去除空白字符
            code_str = str(security_code).strip()
            
            # 如果是浮点数格式（如 "1.0"），转换为整数字符串
            if '.' in code_str:
                try:
                    code_float = float(code_str)
                    if code_float.is_integer():
                        code_str = str(int(code_float))
                except:
                    pass
            
            # 确保代码长度为6位，不足的前面补0
            if code_str.isdigit() and len(code_str) <= 6:
                code_str = code_str.zfill(6)
            
            return code_str
            
        except Exception as e:
            logger.warning(f"标准化证券代码时发生错误: {e}, security_code: {security_code}")
            return str(security_code) if not pd.isna(security_code) else ""
    
    def preprocess_competitive_data(self) -> Dict[Tuple[str, int], Dict[str, float]]:
        """预处理竞争策略数据
        
        将竞争策略数据转换为以(证券代码, 年份)为键的字典格式，便于后续匹配。
        
        Returns:
            Dict[Tuple[str, int], Dict[str, float]]: 竞争策略指标字典
        """
        try:
            logger.info("开始预处理竞争策略数据...")
            
            if self.competitive_data is None:
                raise ValueError("竞争策略数据未加载")
            
            competitive_metrics = {}
            processed_count = 0
            skipped_count = 0
            
            with tqdm(self.competitive_data.iterrows(), 
                     desc="预处理竞争策略数据", 
                     unit="行", 
                     total=len(self.competitive_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        # 提取和标准化证券代码
                        security_code = self.normalize_security_code(row['security_code'])
                        
                        # 提取年份
                        year = self.extract_year_from_rep_period(row['rep_period'])
                        
                        if not security_code or year is None:
                            skipped_count += 1
                            pbar.set_postfix(processed=processed_count, skipped=skipped_count)
                            continue
                        
                        # 提取cost和diff指标
                        cost = row.get('cost', np.nan)
                        diff = row.get('diff', np.nan)
                        
                        # 转换为数值类型
                        try:
                            cost = float(cost) if not pd.isna(cost) else np.nan
                        except:
                            cost = np.nan
                        
                        try:
                            diff = float(diff) if not pd.isna(diff) else np.nan
                        except:
                            diff = np.nan
                        
                        # 创建键值对
                        key = (security_code, year)
                        
                        # 如果同一个公司同一年有多条记录，保留最后一条
                        competitive_metrics[key] = {
                            'cost': cost,
                            'diff': diff
                        }
                        
                        processed_count += 1
                        pbar.set_postfix(processed=processed_count, skipped=skipped_count)
                        
                    except Exception as e:
                        logger.warning(f"预处理第 {idx} 行时发生错误: {e}")
                        skipped_count += 1
                        continue
            
            logger.info(f"竞争策略数据预处理完成，成功处理: {processed_count} 条，跳过: {skipped_count} 条")
            logger.info(f"唯一的(证券代码, 年份)组合数: {len(competitive_metrics)}")
            
            # 显示一些统计信息
            if competitive_metrics:
                # 统计有效的cost和diff值
                cost_valid_count = sum(1 for metrics in competitive_metrics.values() 
                                     if not pd.isna(metrics['cost']))
                diff_valid_count = sum(1 for metrics in competitive_metrics.values() 
                                     if not pd.isna(metrics['diff']))
                
                logger.info(f"有效的cost值数量: {cost_valid_count}")
                logger.info(f"有效的diff值数量: {diff_valid_count}")
                
                # 显示前几个样例
                sample_keys = list(competitive_metrics.keys())[:5]
                logger.info("预处理结果样例:")
                for key in sample_keys:
                    code, year = key
                    metrics = competitive_metrics[key]
                    logger.info(f"  {code}_{year}: cost={metrics['cost']:.6f}, diff={metrics['diff']:.6f}")
            
            return competitive_metrics
            
        except Exception as e:
            logger.error(f"❌ 预处理竞争策略数据时发生错误: {e}")
            raise
    
    def merge_competitive_metrics_with_data(self, 
                                          competitive_metrics: Dict[Tuple[str, int], Dict[str, float]]) -> pd.DataFrame:
        """将竞争策略指标与现有数据合并
        
        Args:
            competitive_metrics: 竞争策略指标字典
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        try:
            logger.info("开始合并竞争策略指标与现有数据...")
            
            if self.input_data is None:
                raise ValueError("输入数据未加载")
            
            # 复制现有数据
            merged_data = self.input_data.copy()
            
            # 初始化竞争策略指标列
            competitive_columns = ['cost', 'diff']
            
            # 初始化所有列为NaN
            for col in competitive_columns:
                merged_data[col] = np.nan
            
            # 合并指标
            logger.info("正在合并竞争策略指标...")
            matched_count = 0
            
            with tqdm(merged_data.iterrows(), 
                     desc="合并竞争策略指标", 
                     unit="行", 
                     total=len(merged_data)) as pbar:
                
                for idx, row in pbar:
                    try:
                        # 获取股票代码和年份
                        stock_code = self.normalize_security_code(row['股票代码'])
                        
                        if pd.isna(row['统计截止日期_年份']):
                            continue
                        
                        year = int(row['统计截止日期_年份'])
                        pbar.set_postfix(code=stock_code[:6], year=year)
                        
                        # 查找匹配的竞争策略指标
                        metric_key = (stock_code, year)
                        
                        if metric_key in competitive_metrics:
                            metrics = competitive_metrics[metric_key]
                            
                            # 合并cost指标
                            if not pd.isna(metrics['cost']):
                                merged_data.at[idx, 'cost'] = metrics['cost']
                            
                            # 合并diff指标
                            if not pd.isna(metrics['diff']):
                                merged_data.at[idx, 'diff'] = metrics['diff']
                            
                            matched_count += 1
                            
                    except Exception as e:
                        logger.warning(f"合并竞争策略指标第 {idx} 行时发生错误: {e}")
                        continue
            
            logger.info(f"竞争策略指标匹配成功: {matched_count} 条记录")
            
            # 统计合并结果
            logger.info("合并结果统计:")
            for col in competitive_columns:
                non_na_count = merged_data[col].notna().sum()
                total_count = len(merged_data)
                coverage = non_na_count / total_count * 100
                
                if non_na_count > 0:
                    mean_val = merged_data[col].mean()
                    std_val = merged_data[col].std()
                    min_val = merged_data[col].min()
                    max_val = merged_data[col].max()
                    
                    logger.info(f"{col}: {non_na_count}/{total_count} ({coverage:.1f}%)")
                    logger.info(f"  均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
                    logger.info(f"  最小值: {min_val:.6f}, 最大值: {max_val:.6f}")
                else:
                    logger.info(f"{col}: {non_na_count}/{total_count} ({coverage:.1f}%)")
            
            logger.info("🎉 竞争策略指标与现有数据合并完成")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 合并竞争策略指标与现有数据时发生错误: {e}")
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
        """运行完整的竞争策略指标分析流程
        
        Returns:
            pd.DataFrame: 包含竞争策略指标的完整数据
        """
        try:
            logger.info("🚀 开始运行完整的竞争策略指标分析流程")
            
            # 1. 加载竞争策略数据
            self.load_competitive_data()
            
            # 2. 加载输入数据
            self.load_input_data()
            
            # 3. 预处理竞争策略数据
            competitive_metrics = self.preprocess_competitive_data()
            
            # 4. 合并指标与现有数据
            merged_data = self.merge_competitive_metrics_with_data(competitive_metrics)
            
            # 5. 保存结果
            self.save_results(merged_data)
            
            logger.info("🎉 竞争策略指标分析流程完成！")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ 运行竞争策略指标分析流程时发生错误: {e}")
            raise


def main():
    """主函数 - 用于直接运行此模块时的测试"""
    try:
        calculator = CompetitiveStrategyCalculator()
        result_data = calculator.run_complete_analysis()
        
        print(f"\n📊 分析结果概览:")
        print(f"数据形状: {result_data.shape}")
        
        # 显示竞争策略指标的统计信息
        competitive_columns = ['cost', 'diff']
        
        print(f"\n竞争策略指标统计:")
        for col in competitive_columns:
            non_na_count = result_data[col].notna().sum()
            if non_na_count > 0:
                mean_val = result_data[col].mean()
                std_val = result_data[col].std()
                print(f"{col}: {non_na_count} 条有效数据")
                print(f"  均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
            else:
                print(f"{col}: {non_na_count} 条有效数据")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        raise


if __name__ == "__main__":
    main()