"""
统一输出管理器 - Unified Output Manager

解决问题：
1. 各实验脚本输出文件重复
2. 文件命名不一致
3. 输出目录散乱

设计原则：
1. 单一职责：每种类型的输出只生成一次
2. 统一命名：所有文件遵循规范化命名
3. 分层输出：主结果/中间文件/日志分开

@author: ABM Research Team
@date: 2026-01-07
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt


class OutputManager:
    """
    统一输出管理器
    
    用法：
        manager = OutputManager(
            base_dir=Path("results"),
            experiment_type="integrated_workflow",
            experiment_id="20260107_120000"
        )
        
        # 保存数据
        manager.save_csv(df, "simulation_data", category="main")
        manager.save_figure(fig, "time_evolution", category="figures")
        manager.save_config(config)
        
        # 生成报告
        manager.generate_summary_report()
    """
    
    # 输出类别定义
    CATEGORIES = {
        'main': '',                      # 主结果放根目录
        'figures': 'figures/',           # 图表
        'data': 'data/',                 # 数据文件
        'intermediate': 'intermediate/', # 中间结果
        'logs': 'logs/',                 # 日志
        'reports': 'reports/'            # 报告
    }
    
    # 标准文件命名
    STANDARD_FILES = {
        # 核心数据
        'simulation_data': 'simulation_data_{years}years.csv',
        'scenario_comparison': 'scenario_comparison.csv',
        'key_timepoints': 'key_timepoints_summary.csv',
        
        # 核心图表
        'time_evolution': 'time_evolution_{years}years.png',
        'scenario_comparison_fig': 'scenario_comparison.png',
        'key_metrics_dashboard': 'key_metrics_dashboard.png',
        
        # 配置和报告
        'config': 'experiment_config.yaml',
        'summary_report': 'EXPERIMENT_SUMMARY.md'
    }
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        experiment_type: str,
        experiment_id: Optional[str] = None,
        clean_old_results: bool = False
    ):
        """
        初始化输出管理器
        
        Args:
            base_dir: 结果根目录
            experiment_type: 实验类型 (integrated, counterfactual, subsidy, policy, sensitivity)
            experiment_id: 实验ID（默认使用时间戳）
            clean_old_results: 是否清理旧结果目录
        """
        self.base_dir = Path(base_dir)
        self.experiment_type = experiment_type
        
        # 生成实验ID
        if experiment_id is None:
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_id = experiment_id
        
        # 创建输出目录
        self.output_dir = self.base_dir / f"run_{experiment_type}_{self.experiment_id}"
        
        # 创建子目录
        for category, subdir in self.CATEGORIES.items():
            if subdir:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 记录已生成的文件（避免重复）
        self._generated_files = set()
        
        # 清理旧结果
        if clean_old_results:
            self._clean_old_results()
    
    def _clean_old_results(self, keep_days: int = 7):
        """
        清理超过指定天数的旧结果目录
        
        Args:
            keep_days: 保留最近N天的结果
        """
        import time
        
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        pattern = f"run_{self.experiment_type}_*"
        
        for old_dir in self.base_dir.glob(pattern):
            if old_dir.is_dir() and old_dir != self.output_dir:
                try:
                    mtime = old_dir.stat().st_mtime
                    if mtime < cutoff_time:
                        shutil.rmtree(old_dir)
                        print(f"[清理] 已删除旧目录: {old_dir.name}")
                except Exception as e:
                    print(f"[警告] 无法删除 {old_dir.name}: {e}")
    
    def get_path(
        self,
        filename: str,
        category: str = 'main'
    ) -> Path:
        """
        获取输出文件路径
        
        Args:
            filename: 文件名
            category: 类别 (main/figures/data/intermediate/logs/reports)
        
        Returns:
            完整文件路径
        """
        subdir = self.CATEGORIES.get(category, '')
        return self.output_dir / subdir / filename
    
    def save_csv(
        self,
        df: pd.DataFrame,
        name: str,
        category: str = 'data',
        **kwargs
    ) -> Path:
        """
        保存DataFrame为CSV
        
        Args:
            df: 要保存的DataFrame
            name: 文件名（不含扩展名）
            category: 输出类别
            **kwargs: 传递给to_csv的参数
        
        Returns:
            保存的文件路径
        """
        # 避免重复生成
        file_key = f"csv:{name}"
        if file_key in self._generated_files:
            print(f"[跳过] {name}.csv 已存在")
            return self.get_path(f"{name}.csv", category)
        
        filepath = self.get_path(f"{name}.csv", category)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=kwargs.pop('index', False), **kwargs)
        self._generated_files.add(file_key)
        
        print(f"[保存] {filepath.relative_to(self.base_dir)}")
        return filepath
    
    def save_figure(
        self,
        fig,
        name: str,
        category: str = 'figures',
        dpi: int = 150,
        **kwargs
    ) -> Path:
        """
        保存matplotlib图表
        
        Args:
            fig: matplotlib figure对象
            name: 文件名（不含扩展名）
            category: 输出类别
            dpi: 分辨率
            **kwargs: 传递给savefig的参数
        
        Returns:
            保存的文件路径
        """
        # 避免重复生成
        file_key = f"fig:{name}"
        if file_key in self._generated_files:
            print(f"[跳过] {name}.png 已存在")
            plt.close(fig)
            return self.get_path(f"{name}.png", category)
        
        filepath = self.get_path(f"{name}.png", category)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
        plt.close(fig)
        self._generated_files.add(file_key)
        
        print(f"[保存] {filepath.relative_to(self.base_dir)}")
        return filepath
    
    def save_json(
        self,
        data: dict,
        name: str,
        category: str = 'data'
    ) -> Path:
        """
        保存JSON文件
        """
        filepath = self.get_path(f"{name}.json", category)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"[保存] {filepath.relative_to(self.base_dir)}")
        return filepath
    
    def save_config(self, config: dict) -> Path:
        """
        保存实验配置
        """
        filepath = self.get_path('experiment_config.yaml', 'main')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"[保存] {filepath.relative_to(self.base_dir)}")
        return filepath
    
    def save_report(
        self,
        content: str,
        name: str = 'EXPERIMENT_SUMMARY',
        format: str = 'md'
    ) -> Path:
        """
        保存报告文件
        """
        filepath = self.get_path(f"{name}.{format}", 'reports')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[保存] {filepath.relative_to(self.base_dir)}")
        return filepath
    
    def generate_summary_report(
        self,
        experiment_config: dict,
        key_results: dict,
        findings: List[str]
    ) -> Path:
        """
        生成实验摘要报告
        
        Args:
            experiment_config: 实验配置
            key_results: 关键结果指标
            findings: 主要发现列表
        
        Returns:
            报告文件路径
        """
        report = f"""# 实验结果报告

## 实验信息
- **类型**: {self.experiment_type}
- **ID**: {self.experiment_id}
- **时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验配置
```yaml
{yaml.dump(experiment_config, allow_unicode=True, default_flow_style=False)}
```

## 关键结果

| 指标 | 数值 |
|------|------|
"""
        for key, value in key_results.items():
            if isinstance(value, float):
                report += f"| {key} | {value:.4f} |\n"
            else:
                report += f"| {key} | {value} |\n"
        
        report += "\n## 主要发现\n\n"
        for i, finding in enumerate(findings, 1):
            report += f"{i}. {finding}\n"
        
        report += f"\n## 输出文件\n\n"
        for filepath in sorted(self.output_dir.rglob('*')):
            if filepath.is_file():
                rel_path = filepath.relative_to(self.output_dir)
                report += f"- `{rel_path}`\n"
        
        return self.save_report(report)
    
    def get_generated_files(self) -> List[str]:
        """
        获取已生成文件列表
        """
        return list(self._generated_files)


def clean_old_results(
    results_dir: Union[str, Path],
    keep_latest: int = 5,
    experiment_types: Optional[List[str]] = None
):
    """
    清理旧的结果目录，保留最近N个
    
    Args:
        results_dir: results目录路径
        keep_latest: 每种实验类型保留最近N个
        experiment_types: 要清理的实验类型列表
    """
    results_dir = Path(results_dir)
    
    if experiment_types is None:
        experiment_types = [
            'integrated_workflow',
            'counterfactual',
            'subsidy_evaluation',
            'policy_optimization',
            'sensitivity'
        ]
    
    for exp_type in experiment_types:
        pattern = f"run_{exp_type}_*"
        dirs = sorted(
            [d for d in results_dir.glob(pattern) if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # 保留最新的N个
        to_delete = dirs[keep_latest:]
        for old_dir in to_delete:
            try:
                shutil.rmtree(old_dir)
                print(f"[清理] 已删除: {old_dir.name}")
            except Exception as e:
                print(f"[警告] 无法删除 {old_dir.name}: {e}")


def get_latest_result_dir(
    results_dir: Union[str, Path],
    experiment_type: str
) -> Optional[Path]:
    """
    获取最新的结果目录
    
    Args:
        results_dir: results目录路径
        experiment_type: 实验类型
    
    Returns:
        最新结果目录路径，如果不存在返回None
    """
    results_dir = Path(results_dir)
    pattern = f"run_{experiment_type}_*"
    
    dirs = sorted(
        [d for d in results_dir.glob(pattern) if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return dirs[0] if dirs else None


if __name__ == '__main__':
    # 示例用法
    manager = OutputManager(
        base_dir=Path("results"),
        experiment_type="test",
        clean_old_results=False
    )
    
    print(f"输出目录: {manager.output_dir}")
    print(f"图表目录: {manager.get_path('test.png', 'figures')}")
    print(f"数据目录: {manager.get_path('test.csv', 'data')}")
