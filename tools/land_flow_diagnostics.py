# -*- coding: utf-8 -*-
"""
土地流转诊断工具 - 诊断为什么农户退出后土地没有流转给规模户/企业
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_builder import ModelBuilder
import yaml

def run_land_flow_diagnostics():
    """诊断土地流转问题"""
    print("=" * 70)
    print("土地流转深度诊断")
    print("问题：为什么Non_Farmer退出后土地没有流转给规模户/企业？")
    print("=" * 70)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'config', 'master_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    builder = ModelBuilder(config)
    model = builder.build()
    
    # 初始状态
    print("\n[初始状态]")
    print_land_distribution(model)
    
    # 运行10步并跟踪
    for step in range(1, 11):
        model.step()
        
        if step in [1, 5, 10]:
            print(f"\n[第 {step} 步]")
            print_land_distribution(model)
            analyze_transfer_market(model)
    
    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)
    
    # 最终分析
    final_analysis(model)

def print_land_distribution(model):
    """打印土地分布"""
    from agents.farmer_agent import LivelihoodStrategy
    
    # 按生计策略统计土地
    strategy_land = {
        'Pure_Farmer': 0,
        'Pluriactive': 0, 
        'Non_Farmer': 0
    }
    strategy_count = {
        'Pure_Farmer': 0,
        'Pluriactive': 0,
        'Non_Farmer': 0
    }
    
    for farmer in model.farmers:
        strategy = farmer.livelihood_strategy.value
        area = farmer.get_operated_land_area()
        strategy_land[strategy] += area
        strategy_count[strategy] += 1
    
    # 企业土地
    enterprise_land = sum(e.get_operated_land_area() for e in model.agricultural_enterprises)
    
    # 总土地面积
    total_land = sum(p.area for p in model.land_patches)
    leased_land = sum(p.area for p in model.land_patches if p.leased_to is not None)
    abandoned_land = sum(p.area for p in model.land_patches if getattr(p, 'abandoned', False))
    
    print(f"  纯农户: {strategy_count['Pure_Farmer']}人, {strategy_land['Pure_Farmer']:.0f}亩")
    print(f"  兼业户: {strategy_count['Pluriactive']}人, {strategy_land['Pluriactive']:.0f}亩")
    print(f"  非农户: {strategy_count['Non_Farmer']}人, {strategy_land['Non_Farmer']:.0f}亩")
    print(f"  企业: {len(model.agricultural_enterprises)}家, {enterprise_land:.0f}亩")
    print(f"  总土地: {total_land:.0f}亩, 流转中: {leased_land:.0f}亩, 撂荒: {abandoned_land:.0f}亩")
    
    # 检查土地守恒
    accounted_land = strategy_land['Pure_Farmer'] + strategy_land['Pluriactive'] + strategy_land['Non_Farmer'] + enterprise_land
    unaccounted = total_land - accounted_land
    if abs(unaccounted) > 1:
        print(f"  ⚠️ 未被计入的土地: {unaccounted:.0f}亩 (可能是统计口径问题)")

def analyze_transfer_market(model):
    """分析土地流转市场"""
    stats = model.land_clearing_house.get_market_statistics()
    
    print(f"  土地市场: 交易{stats.get('transaction_count', 0)}笔, "
          f"面积{stats.get('total_area_transferred', 0):.0f}亩, "
          f"均价{stats.get('average_price', 0):.0f}元/亩")
    
    # 供需分析
    supply_count = getattr(model.land_clearing_house, 'last_supply_count', 0)
    demand_count = getattr(model.land_clearing_house, 'last_demand_count', 0)
    print(f"  供需情况: 供给{supply_count}笔, 需求{demand_count}笔")

def final_analysis(model):
    """最终分析"""
    from agents.farmer_agent import LivelihoodStrategy
    
    # 分析Non_Farmer的土地状态
    non_farmer_patches = []
    for farmer in model.farmers:
        if farmer.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            # 获取该农户名下的所有地块
            owned = [p for p in model.land_patches if p.owner == farmer]
            leased_out = [p for p in owned if p.leased_to is not None]
            self_operated = [p for p in owned if p.leased_to is None]
            
            owned_area = sum(p.area for p in owned)
            leased_out_area = sum(p.area for p in leased_out)
            self_operated_area = sum(p.area for p in self_operated)
            
            if owned_area > 0:
                non_farmer_patches.append({
                    'farmer': farmer,
                    'owned_area': owned_area,
                    'leased_out_area': leased_out_area,
                    'self_operated_area': self_operated_area
                })
    
    print(f"\nNon_Farmer土地状态分析 (共{len(non_farmer_patches)}户有地):")
    
    total_owned = sum(p['owned_area'] for p in non_farmer_patches)
    total_leased_out = sum(p['leased_out_area'] for p in non_farmer_patches)
    total_self_operated = sum(p['self_operated_area'] for p in non_farmer_patches)
    
    print(f"  Non_Farmer名下总土地: {total_owned:.0f}亩")
    print(f"  已流转出去: {total_leased_out:.0f}亩 ({total_leased_out/total_owned*100:.1f}%)")
    print(f"  仍自己经营: {total_self_operated:.0f}亩 ({total_self_operated/total_owned*100:.1f}%)")
    
    if total_self_operated > 100:
        print(f"\n⚠️ 问题发现: Non_Farmer仍持有{total_self_operated:.0f}亩土地未流转！")
        print("   这些土地既未流转也未正式撂荒，但不被计入粮食生产")
        print("   修复方案: 增强Non_Farmer的土地供给意愿，或强制流转机制")
    
    # 检查企业需求是否充足
    print(f"\n企业需求分析:")
    for i, ent in enumerate(model.agricultural_enterprises[:5]):
        current_area = ent.get_operated_land_area()
        print(f"  企业{i+1}: 当前经营{current_area:.0f}亩")
    
    # 检查可租赁地块
    available = model.get_available_patches()
    available_area = sum(p.area for p in available)
    print(f"\n当前可租赁地块: {len(available)}块, 共{available_area:.0f}亩")
    
    if available_area > 500:
        print(f"  ⚠️ 有大量可租赁土地({available_area:.0f}亩)但未被企业租入")
        print("   可能原因: 1.企业需求不足 2.价格不匹配 3.位置不佳")

if __name__ == "__main__":
    run_land_flow_diagnostics()
