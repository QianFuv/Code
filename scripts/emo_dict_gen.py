#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感词典生成器
用于合并中文金融情感词典和RFS词表，生成综合的情绪词典

作者: Nolan
创建时间: 2025年6月6日
文件描述: 将中文金融情感词典_姜富伟等(2020).xlsx和RFS词表.csv两个文件进行合并
"""

import pandas as pd
import os
import sys
import argparse
from typing import Dict, List, Set, Tuple


def read_lm_dict(file_path: str) -> Tuple[Set[str], Set[str]]:
    """
    读取中文金融情感词典Excel文件
    
    Args:
        file_path (str): Excel文件路径
        
    Returns:
        Tuple[Set[str], Set[str]]: 返回正面词汇集合和负面词汇集合
    """
    try:
        # 读取Excel文件的所有工作表
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        positive_words = set()
        negative_words = set()
        
        # 遍历所有工作表
        for sheet_name, df in excel_data.items():
            print(f"正在处理工作表: {sheet_name}")
            
            # 根据列名识别正面和负面词汇
            for column in df.columns:
                column_lower = column.lower()
                if any(keyword in column_lower for keyword in ['正面', 'positive', 'pos']):
                    # 处理正面词汇列
                    words = df[column].dropna().astype(str).str.strip()
                    positive_words.update(words[words != ''])
                elif any(keyword in column_lower for keyword in ['负面', 'negative', 'neg']):
                    # 处理负面词汇列
                    words = df[column].dropna().astype(str).str.strip()
                    negative_words.update(words[words != ''])
        
        print(f"从LM词典中读取到正面词汇: {len(positive_words)} 个")
        print(f"从LM词典中读取到负面词汇: {len(negative_words)} 个")
        
        return positive_words, negative_words
        
    except Exception as e:
        print(f"读取LM词典时发生错误: {e}")
        return set(), set()


def read_rfs_dict(file_path: str) -> Tuple[Set[str], Set[str]]:
    """
    读取RFS词表CSV文件
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        Tuple[Set[str], Set[str]]: 返回正面词汇集合和负面词汇集合
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 提取正面和负面词汇
        positive_words = set()
        negative_words = set()
        
        if 'pos' in df.columns:
            pos_words = df['pos'].dropna().astype(str).str.strip()
            positive_words.update(pos_words[pos_words != ''])
        
        if 'neg' in df.columns:
            neg_words = df['neg'].dropna().astype(str).str.strip()
            negative_words.update(neg_words[neg_words != ''])
        
        print(f"从RFS词典中读取到正面词汇: {len(positive_words)} 个")
        print(f"从RFS词典中读取到负面词汇: {len(negative_words)} 个")
        
        return positive_words, negative_words
        
    except Exception as e:
        print(f"读取RFS词典时发生错误: {e}")
        return set(), set()


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='合并中文金融情感词典和RFS词表生成综合情感词典',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python emo_dict_gen.py --lm-dict path/to/lm.xlsx --rfs-dict path/to/rfs.csv --output-dir path/to/emo_dict
  python emo_dict_gen.py --lm-dict lm.xlsx --rfs-dict rfs.csv --output-dir ./emo_dict --verbose
        """
    )
    
    parser.add_argument(
        '--lm-dict',
        type=str,
        required=True,
        help='LM金融情感词典Excel文件路径（必需）'
    )
    
    parser.add_argument(
        '--rfs-dict', 
        type=str,
        required=True,
        help='RFS词表CSV文件路径（必需）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录路径（必需）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细处理信息'
    )
    
    parser.add_argument(
        '--no-conflict-removal',
        action='store_true',
        help='不移除冲突词汇（同时出现在正面和负面词典中的词汇）'
    )
    
    return parser.parse_args()


def merge_dictionaries(lm_pos: Set[str], lm_neg: Set[str], 
                      rfs_pos: Set[str], rfs_neg: Set[str],
                      remove_conflicts: bool = True) -> Tuple[Set[str], Set[str]]:
    """
    合并两个情感词典
    
    Args:
        lm_pos (Set[str]): LM词典正面词汇
        lm_neg (Set[str]): LM词典负面词汇
        rfs_pos (Set[str]): RFS词典正面词汇
        rfs_neg (Set[str]): RFS词典负面词汇
        remove_conflicts (bool): 是否移除冲突词汇
        
    Returns:
        Tuple[Set[str], Set[str]]: 合并后的正面词汇集合和负面词汇集合
    """
    # 合并正面词汇
    merged_positive = lm_pos.union(rfs_pos)
    
    # 合并负面词汇
    merged_negative = lm_neg.union(rfs_neg)
    
    # 检查冲突词汇（同时出现在正面和负面词典中的词汇）
    conflict_words = merged_positive.intersection(merged_negative)
    
    if conflict_words:
        print(f"发现冲突词汇 {len(conflict_words)} 个:")
        for word in sorted(list(conflict_words)[:10]):  # 只显示前10个
            print(f"  - {word}")
        if len(conflict_words) > 10:
            print(f"  ... 还有 {len(conflict_words) - 10} 个冲突词汇")
        
        if remove_conflicts:
            # 从两个集合中移除冲突词汇
            merged_positive -= conflict_words
            merged_negative -= conflict_words
            print("已将冲突词汇从词典中移除")
        else:
            print("保留冲突词汇（词汇将同时出现在正面和负面词典中）")
    
    print(f"合并后正面词汇总数: {len(merged_positive)} 个")
    print(f"合并后负面词汇总数: {len(merged_negative)} 个")
    
    return merged_positive, merged_negative


def save_merged_dict(positive_words: Set[str], negative_words: Set[str], 
                    output_dir: str) -> None:
    """
    保存合并后的情感词典为单个CSV文件
    
    Args:
        positive_words (Set[str]): 正面词汇集合
        negative_words (Set[str]): 负面词汇集合
        output_dir (str): 输出目录路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV格式
        csv_path = os.path.join(output_dir, "emo_dict.csv")
        
        # 将词汇转换为列表并排序
        pos_list = sorted(list(positive_words))
        neg_list = sorted(list(negative_words))
        
        # 创建DataFrame，处理长度不一致的情况
        max_length = max(len(pos_list), len(neg_list))
        
        # 用空字符串填充较短的列表
        pos_list.extend([''] * (max_length - len(pos_list)))
        neg_list.extend([''] * (max_length - len(neg_list)))
        
        df = pd.DataFrame({
            'positive': pos_list,
            'negative': neg_list
        })
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"合并后的词典已保存为: {csv_path}")
        
    except Exception as e:
        print(f"保存词典时发生错误: {e}")


def print_statistics(positive_words: Set[str], negative_words: Set[str]) -> None:
    """
    打印词典统计信息
    
    Args:
        positive_words (Set[str]): 正面词汇集合
        negative_words (Set[str]): 负面词汇集合
    """
    print("\n" + "=" * 30)
    print("词典统计信息")
    print("=" * 30)
    print(f"正面词汇数量: {len(positive_words):,} 个")
    print(f"负面词汇数量: {len(negative_words):,} 个") 
    print(f"总词汇数量: {len(positive_words) + len(negative_words):,} 个")
    
    # 显示一些示例词汇
    print("\n正面词汇示例 (前10个):")
    for word in sorted(list(positive_words))[:10]:
        print(f"  - {word}")
    
    print("\n负面词汇示例 (前10个):")
    for word in sorted(list(negative_words))[:10]:
        print(f"  - {word}")


def main():
    """
    主函数：执行情感词典合并流程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 50)
    print("情感词典合并工具")
    print("=" * 50)
    
    # 使用命令行参数
    lm_dict_path = args.lm_dict
    rfs_dict_path = args.rfs_dict
    output_dir = args.output_dir
    
    if args.verbose:
        print(f"LM词典路径: {lm_dict_path}")
        print(f"RFS词典路径: {rfs_dict_path}")
        print(f"输出目录: {output_dir}")
    
    # 检查输入文件是否存在
    if not os.path.exists(lm_dict_path):
        print(f"错误: 找不到LM词典文件: {lm_dict_path}")
        sys.exit(1)
    
    if not os.path.exists(rfs_dict_path):
        print(f"错误: 找不到RFS词典文件: {rfs_dict_path}")
        sys.exit(1)
    
    print(f"正在读取LM词典: {os.path.basename(lm_dict_path)}")
    lm_pos, lm_neg = read_lm_dict(lm_dict_path)
    
    print(f"正在读取RFS词典: {os.path.basename(rfs_dict_path)}")
    rfs_pos, rfs_neg = read_rfs_dict(rfs_dict_path)
    
    print("正在合并词典...")
    merged_pos, merged_neg = merge_dictionaries(
        lm_pos, lm_neg, rfs_pos, rfs_neg, 
        remove_conflicts=not args.no_conflict_removal
    )
    
    print("正在保存合并后的词典...")
    save_merged_dict(merged_pos, merged_neg, output_dir)
    
    # 打印统计信息
    print_statistics(merged_pos, merged_neg)
    
    print("=" * 50)
    print("词典合并完成!")
    print(f"输出文件: {os.path.join(output_dir, 'emo_dict.csv')}")
    print("=" * 50)


if __name__ == "__main__":
    main()