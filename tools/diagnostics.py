# -*- coding: utf-8 -*-
"""
诊断工具包 - 整合所有诊断功能的统一接口

本模块整合了项目中所有的诊断脚本功能，提供统一的命令行接口。
原始的独立诊断脚本已归档到 archive/legacy_scripts/ 目录。

使用方法:
    python tools/diagnostics.py --help
    python tools/diagnostics.py income --step 5
    python tools/diagnostics.py land --detailed
    python tools/diagnostics.py all

作者: 研究团队
日期: 2025-11-30
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DiagnosticsToolkit:
    """诊断工具包主类"""
    
    def __init__(self, model=None, config=None):
        """初始化诊断工具包
        
        Args:
            model: MainModel实例，如果为None则在需要时创建
            config: 配置字典，如果为None则从文件加载
        """
        self.model = model
        self.config = config
        self._setup_model()
    
    def _setup_model(self):
        """设置模型实例"""
        if self.model is None:
            try:
                import yaml
                from core.model_builder import ModelBuilder
                
                config_path = PROJECT_ROOT / "config" / "master_config.yaml"
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                
                # 使用较少的步数进行诊断
                self.config['experiment']['steps'] = 5
                
                builder = ModelBuilder(config=self.config)
                self.model = builder.build()
                print("[INFO] 模型已初始化")
            except Exception as e:
                print(f"[WARNING] 模型初始化失败: {e}")
                self.model = None
    
    def diagnose_income(self, step=5, detailed=False):
        """诊断收入相关问题
        
        Args:
            step: 运行到第几步进行诊断
            detailed: 是否输出详细信息
        """
        print("\n" + "="*60)
        print("收入诊断报告")
        print("="*60)
        
        if self.model is None:
            print("[ERROR] 模型未初始化")
            return
        
        # 运行模型
        for i in range(step):
            self.model.step()
        
        # 收集农户收入统计
        incomes = []
        for farmer in self.model.farmers:
            total_income = getattr(farmer, 'total_income', 0)
            agri_income = getattr(farmer, 'agricultural_income', 0)
            off_farm = getattr(farmer, 'off_farm_income', 0)
            subsidy = getattr(farmer, 'subsidy_income', 0)
            incomes.append({
                'id': farmer.unique_id,
                'total': total_income,
                'agricultural': agri_income,
                'off_farm': off_farm,
                'subsidy': subsidy,
                'strategy': farmer.livelihood_strategy.value
            })
        
        # 统计分析
        import numpy as np
        total_incomes = [i['total'] for i in incomes]
        
        print(f"\n第 {step} 步收入统计:")
        print(f"  农户数量: {len(incomes)}")
        print(f"  平均总收入: {np.mean(total_incomes):.2f} 元")
        print(f"  收入中位数: {np.median(total_incomes):.2f} 元")
        print(f"  收入标准差: {np.std(total_incomes):.2f} 元")
        print(f"  最低收入: {np.min(total_incomes):.2f} 元")
        print(f"  最高收入: {np.max(total_incomes):.2f} 元")
        
        # 零收入诊断
        zero_income_count = sum(1 for i in total_incomes if i <= 0)
        if zero_income_count > 0:
            print(f"\n[WARNING] 存在 {zero_income_count} 个零收入或负收入农户!")
        
        # 生计策略分布
        from collections import Counter
        strategies = [i['strategy'] for i in incomes]
        strategy_counts = Counter(strategies)
        print(f"\n生计策略分布:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} ({count/len(incomes)*100:.1f}%)")
        
        if detailed:
            print("\n详细收入明细 (前10个农户):")
            for i in incomes[:10]:
                print(f"  农户{i['id']}: 总收入={i['total']:.0f}, "
                      f"农业={i['agricultural']:.0f}, "
                      f"非农={i['off_farm']:.0f}, "
                      f"补贴={i['subsidy']:.0f}")
        
        return incomes
    
    def diagnose_land_transfer(self, step=5, detailed=False):
        """诊断土地流转相关问题
        
        Args:
            step: 运行到第几步进行诊断
            detailed: 是否输出详细信息
        """
        print("\n" + "="*60)
        print("土地流转诊断报告")
        print("="*60)
        
        if self.model is None:
            print("[ERROR] 模型未初始化")
            return
        
        # 运行模型
        for i in range(step):
            self.model.step()
        
        # 统计土地流转情况
        total_patches = len(self.model.land_patches)
        transferred_patches = 0
        total_area = 0
        transferred_area = 0
        
        for patch in self.model.land_patches:
            area = getattr(patch, 'area', 0)
            total_area += area
            if hasattr(patch, 'leased_to') and patch.leased_to is not None:
                transferred_patches += 1
                transferred_area += area
        
        transfer_rate = transferred_patches / total_patches if total_patches > 0 else 0
        
        print(f"\n第 {step} 步土地流转统计:")
        print(f"  总地块数: {total_patches}")
        print(f"  已流转地块: {transferred_patches}")
        print(f"  流转率: {transfer_rate*100:.2f}%")
        print(f"  总面积: {total_area:.2f} 亩")
        print(f"  流转面积: {transferred_area:.2f} 亩")
        
        # 企业租地情况
        print(f"\n企业租地情况:")
        for enterprise in self.model.agricultural_enterprises:
            leased_area = getattr(enterprise, 'total_leased_area', 0)
            print(f"  企业{enterprise.unique_id}: 租入 {leased_area:.2f} 亩")
        
        return {
            'transfer_rate': transfer_rate,
            'transferred_area': transferred_area,
            'total_area': total_area
        }
    
    def diagnose_market(self, step=5):
        """诊断市场相关问题
        
        Args:
            step: 运行到第几步进行诊断
        """
        print("\n" + "="*60)
        print("市场诊断报告")
        print("="*60)
        
        if self.model is None:
            print("[ERROR] 模型未初始化")
            return
        
        # 运行模型
        for i in range(step):
            self.model.step()
        
        env = self.model.environment
        
        print(f"\n第 {step} 步市场状态:")
        print(f"  粮食价格: {env.grain_price:.2f} 元/斤")
        print(f"  经作价格: {env.cash_crop_price:.2f} 元/斤")
        print(f"  粮食价格下限: {env.grain_price_floor:.2f} 元/斤")
        
        # 供需统计
        if hasattr(env, 'grain_supply'):
            print(f"  粮食供给: {env.grain_supply:.2f}")
        if hasattr(env, 'grain_demand'):
            print(f"  粮食需求: {env.grain_demand:.2f}")
        
        return {
            'grain_price': env.grain_price,
            'cash_crop_price': env.cash_crop_price
        }
    
    def diagnose_subsidy(self, step=5):
        """诊断补贴相关问题
        
        Args:
            step: 运行到第几步进行诊断
        """
        print("\n" + "="*60)
        print("补贴诊断报告")
        print("="*60)
        
        if self.model is None:
            print("[ERROR] 模型未初始化")
            return
        
        # 运行模型
        for i in range(step):
            self.model.step()
        
        # 统计补贴发放情况
        total_subsidy = 0
        subsidy_recipients = 0
        
        for farmer in self.model.farmers:
            subsidy = getattr(farmer, 'subsidy_income', 0)
            if subsidy > 0:
                total_subsidy += subsidy
                subsidy_recipients += 1
        
        for enterprise in self.model.agricultural_enterprises:
            subsidy = getattr(enterprise, 'subsidy_revenue', 0)
            if subsidy > 0:
                total_subsidy += subsidy
        
        print(f"\n第 {step} 步补贴统计:")
        print(f"  补贴发放总额: {total_subsidy:.2f} 元")
        print(f"  获得补贴的农户数: {subsidy_recipients}")
        print(f"  平均补贴: {total_subsidy/subsidy_recipients:.2f} 元/户" if subsidy_recipients > 0 else "  平均补贴: 0 元/户")
        
        # 政策参数
        env = self.model.environment
        print(f"\n当前补贴政策:")
        print(f"  补贴率: {getattr(env, 'grain_subsidies_rate', 0)}")
        print(f"  补贴类型: {getattr(env, 'subsidy_type', 'N/A')}")
        print(f"  补贴对象: {getattr(env, 'subsidy_target', 'N/A')}")
        
        return {
            'total_subsidy': total_subsidy,
            'subsidy_recipients': subsidy_recipients
        }
    
    def diagnose_all(self, step=5):
        """运行所有诊断
        
        Args:
            step: 运行到第几步进行诊断
        """
        print("\n" + "="*60)
        print("全面诊断报告")
        print("="*60)
        
        self.diagnose_income(step=step)
        self.diagnose_land_transfer(step=step)
        self.diagnose_market(step=step)
        self.diagnose_subsidy(step=step)
        
        print("\n" + "="*60)
        print("诊断完成")
        print("="*60)
    
    def run_model_check(self):
        """运行模型完整性检查"""
        print("\n" + "="*60)
        print("模型完整性检查")
        print("="*60)
        
        if self.model is None:
            print("[ERROR] 模型未初始化")
            return False
        
        checks = []
        
        # 检查农户
        checks.append(('农户数量', len(self.model.farmers) > 0))
        checks.append(('企业数量', len(self.model.agricultural_enterprises) > 0))
        checks.append(('地块数量', len(self.model.land_patches) > 0))
        checks.append(('政府智能体', self.model.government is not None))
        checks.append(('环境对象', self.model.environment is not None))
        checks.append(('数据收集器', self.model.datacollector is not None))
        
        # 检查市场
        env = self.model.environment
        checks.append(('粮食价格>0', env.grain_price > 0))
        checks.append(('经作价格>0', env.cash_crop_price > 0))
        
        # 输出结果
        all_passed = True
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  [{status}] {name}")
            if not passed:
                all_passed = False
        
        print(f"\n总体状态: {'通过' if all_passed else '存在问题'}")
        return all_passed


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="农业经济多智能体模型诊断工具包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python diagnostics.py income --step 10
  python diagnostics.py land --detailed
  python diagnostics.py market
  python diagnostics.py subsidy
  python diagnostics.py all
  python diagnostics.py check
        """
    )
    
    parser.add_argument('command', 
                        choices=['income', 'land', 'market', 'subsidy', 'all', 'check'],
                        help='诊断命令')
    parser.add_argument('--step', type=int, default=5,
                        help='运行模型步数 (默认: 5)')
    parser.add_argument('--detailed', action='store_true',
                        help='输出详细信息')
    
    args = parser.parse_args()
    
    # 创建诊断工具包
    toolkit = DiagnosticsToolkit()
    
    # 执行命令
    if args.command == 'income':
        toolkit.diagnose_income(step=args.step, detailed=args.detailed)
    elif args.command == 'land':
        toolkit.diagnose_land_transfer(step=args.step, detailed=args.detailed)
    elif args.command == 'market':
        toolkit.diagnose_market(step=args.step)
    elif args.command == 'subsidy':
        toolkit.diagnose_subsidy(step=args.step)
    elif args.command == 'all':
        toolkit.diagnose_all(step=args.step)
    elif args.command == 'check':
        toolkit.run_model_check()


if __name__ == '__main__':
    main()
