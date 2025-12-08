# -*- coding: utf-8 -*-
"""
全面诊断脚本

功能：
1. 诊断农户农业收入占比过高的问题
2. 诊断土地流转率低的问题
3. 检查整个模型流程
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
import numpy as np
from core.model_builder import ModelBuilder


def comprehensive_diagnosis(config_path="config/master_config.yaml", n_steps=10):
    """全面诊断"""
    print("=" * 80)
    print("全面诊断：农户收入占比和土地流转率")
    print("=" * 80)
    
    # @fix: 修复路径问题，支持相对路径和绝对路径
    if not Path(config_path).is_absolute():
        config_path = _project_root / config_path
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 构建模型
    builder = ModelBuilder(config)
    model = builder.build()
    
    # 运行几步
    print("\n运行模型...")
    for step in range(n_steps):
        model.step()
        if step % 2 == 0:
            print(f"  步骤 {step+1}/{n_steps}")
    
    print("\n" + "=" * 80)
    print("诊断结果")
    print("=" * 80)
    
    # 1. 农户收入分析
    print("\n1. 农户收入构成分析:")
    total_income_list = []
    agricultural_income_list = []
    off_farm_income_list = []
    rental_income_list = []
    subsidy_income_list = []
    income_ratio_list = []
    
    for farmer in model.farmers:
        if hasattr(farmer, 'total_income') and farmer.total_income > 0:
            total_income_list.append(farmer.total_income)
            agricultural_income_list.append(getattr(farmer, 'agricultural_income', 0))
            off_farm_income_list.append(getattr(farmer, 'off_farm_income', 0))
            rental_income_list.append(getattr(farmer, 'land_rental_income', 0))
            subsidy_income_list.append(getattr(farmer, 'subsidy_income', 0))
            
            ratio = farmer.agricultural_income / farmer.total_income if farmer.total_income > 0 else 0
            income_ratio_list.append(ratio)
    
    if total_income_list:
        print(f"   平均总收入: {np.mean(total_income_list):.2f}")
        print(f"   平均农业收入: {np.mean(agricultural_income_list):.2f} ({np.mean(agricultural_income_list)/np.mean(total_income_list)*100:.1f}%)")
        print(f"   平均非农收入: {np.mean(off_farm_income_list):.2f} ({np.mean(off_farm_income_list)/np.mean(total_income_list)*100:.1f}%)")
        print(f"   平均租金收入: {np.mean(rental_income_list):.2f} ({np.mean(rental_income_list)/np.mean(total_income_list)*100:.1f}%)")
        print(f"   平均补贴收入: {np.mean(subsidy_income_list):.2f} ({np.mean(subsidy_income_list)/np.mean(total_income_list)*100:.1f}%)")
        print(f"   平均农业收入占比: {np.mean(income_ratio_list):.3f} ({np.mean(income_ratio_list)*100:.1f}%)")
        print(f"   农业收入占比中位数: {np.median(income_ratio_list):.3f} ({np.median(income_ratio_list)*100:.1f}%)")
    
    # 2. 非农收入详细分析
    print("\n2. 非农收入详细分析:")
    off_farm_labor_list = []
    off_farm_wage_list = []
    for farmer in model.farmers:
        if hasattr(farmer, 'off_farm_labor'):
            off_farm_labor_list.append(farmer.off_farm_labor)
        if hasattr(farmer, 'off_farm_wage_level'):
            off_farm_wage_list.append(farmer.off_farm_wage_level)
    
    if off_farm_labor_list:
        print(f"   平均非农劳动分配: {np.mean(off_farm_labor_list):.2f}")
        print(f"   非农劳动分配范围: {np.min(off_farm_labor_list):.2f} - {np.max(off_farm_labor_list):.2f}")
    if off_farm_wage_list:
        print(f"   平均非农工资水平: {np.mean(off_farm_wage_list):.2f}")
        print(f"   非农工资水平范围: {np.min(off_farm_wage_list):.2f} - {np.max(off_farm_wage_list):.2f}")
    
    # 3. 土地流转分析
    print("\n3. 土地流转分析:")
    total_land_area = sum(p.area for p in model.land_patches)
    leased_area = sum(p.area for p in model.land_patches if p.leased_to is not None)
    transfer_rate = leased_area / total_land_area if total_land_area > 0 else 0
    
    print(f"   总土地面积: {total_land_area:.2f} 亩")
    print(f"   流转面积: {leased_area:.2f} 亩")
    print(f"   流转率: {transfer_rate:.4f} ({transfer_rate*100:.2f}%)")
    print(f"   目标流转率: 0.3673 (36.73%)")
    
    # 4. 地块状态分析
    print("\n4. 地块状态分析:")
    total_patches = len(model.land_patches)
    leased_patches = len([p for p in model.land_patches if p.leased_to is not None])
    owned_patches = len([p for p in model.land_patches if p.owner is not None])
    available_patches = len(model.get_available_patches())
    
    print(f"   总地块数: {total_patches}")
    print(f"   已流转地块数: {leased_patches}")
    print(f"   有所有者地块数: {owned_patches}")
    print(f"   可租赁地块数: {available_patches}")
    
    # 5. 企业参与分析
    print("\n5. 企业参与分析:")
    enterprises_with_land = 0
    total_enterprise_area = 0
    for enterprise in model.agricultural_enterprises:
        area = enterprise.get_operated_land_area()
        if area > 0:
            enterprises_with_land += 1
            total_enterprise_area += area
    
    print(f"   总企业数: {len(model.agricultural_enterprises)}")
    print(f"   有土地的企业数: {enterprises_with_land}")
    print(f"   企业总经营面积: {total_enterprise_area:.2f} 亩")
    print(f"   平均企业面积: {total_enterprise_area/len(model.agricultural_enterprises):.2f} 亩")
    
    # 6. 农户生计策略分布
    print("\n6. 农户生计策略分布:")
    strategy_count = {}
    for farmer in model.farmers:
        strategy = farmer.livelihood_strategy.value if hasattr(farmer.livelihood_strategy, 'value') else str(farmer.livelihood_strategy)
        strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
    
    for strategy, count in strategy_count.items():
        pct = count / len(model.farmers) * 100
        print(f"   {strategy}: {count} ({pct:.1f}%)")
    
    # 7. 问题诊断
    print("\n" + "=" * 80)
    print("问题诊断")
    print("=" * 80)
    
    issues = []
    
    # 问题1: 农业收入占比过高
    if income_ratio_list and np.mean(income_ratio_list) > 0.7:
        issues.append(f"[问题] 农业收入占比过高 ({np.mean(income_ratio_list)*100:.1f}%)，应该降低到50-60%")
    
    # 问题2: 非农收入过低
    if off_farm_income_list and np.mean(off_farm_income_list) < np.mean(agricultural_income_list) * 0.3:
        issues.append(f"[问题] 非农收入过低，平均只有 {np.mean(off_farm_income_list):.2f}，应该提高")
    
    # 问题3: 非农劳动分配不足
    if off_farm_labor_list and np.mean(off_farm_labor_list) < 100:
        issues.append(f"[问题] 非农劳动分配不足，平均只有 {np.mean(off_farm_labor_list):.2f}，应该提高")
    
    # 问题4: 土地流转率过低
    if transfer_rate < 0.1:
        issues.append(f"[问题] 土地流转率过低 ({transfer_rate*100:.2f}%)，远低于目标值36.73%")
    
    # 问题5: 可租赁地块不足
    if available_patches < total_patches * 0.3:
        issues.append(f"[问题] 可租赁地块不足 ({available_patches}/{total_patches})，可能影响流转")
    
    if issues:
        print("\n发现的问题:")
        for i, issue in enumerate(issues, 1):
            # @fix: 处理编码问题
            try:
                print(f"  {i}. {issue}")
            except UnicodeEncodeError:
                safe_issue = issue.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(f"  {i}. {safe_issue}")
    else:
        print("\n[OK] 未发现明显问题")
    
    return {
        'income_ratio': np.mean(income_ratio_list) if income_ratio_list else 0,
        'transfer_rate': transfer_rate,
        'issues': issues
    }


if __name__ == '__main__':
    comprehensive_diagnosis()

