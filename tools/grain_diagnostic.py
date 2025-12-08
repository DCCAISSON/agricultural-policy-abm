"""
诊断粮食供给下降的根本原因
"""
import sys
sys.path.insert(0, '.')

import yaml
from core.model_builder import ModelBuilder

# 使用 ModelBuilder 创建模型
builder = ModelBuilder()
m = builder.build()

print("=" * 60)
print("粮食供给诊断分析")
print("=" * 60)

for step in range(5):
    m.step()
    
    # 统计作物选择
    grain_farmers = []
    cash_farmers = []
    no_choice = []
    
    for f in m.farmers:
        if hasattr(f, 'last_crop_choice') and f.last_crop_choice:
            if f.last_crop_choice.value == 'Grain':
                grain_farmers.append(f)
            elif f.last_crop_choice.value == 'Cash_Crop':
                cash_farmers.append(f)
            else:
                no_choice.append(f)
        else:
            no_choice.append(f)
    
    # 计算粮食产量
    total_grain_yield = sum(getattr(f, 'last_yield', 0) for f in grain_farmers)
    total_cash_yield = sum(getattr(f, 'last_yield', 0) for f in cash_farmers)
    
    # 统计不同生计策略的作物选择
    pure_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'PURE_FARMER')
    pure_cash = sum(1 for f in cash_farmers if f.livelihood_strategy.name == 'PURE_FARMER')
    pluri_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'PLURIACTIVE')
    pluri_cash = sum(1 for f in cash_farmers if f.livelihood_strategy.name == 'PLURIACTIVE')
    non_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'NON_FARMER')
    non_cash = sum(1 for f in cash_farmers if f.livelihood_strategy.name == 'NON_FARMER')
    
    # 计算平均产量
    avg_grain_yield = total_grain_yield / len(grain_farmers) if grain_farmers else 0
    avg_cash_yield = total_cash_yield / len(cash_farmers) if cash_farmers else 0
    
    print(f"\n步骤 {step + 1}:")
    print(f"  种粮农户: {len(grain_farmers)}  |  种经作农户: {len(cash_farmers)}  |  无选择: {len(no_choice)}")
    print(f"  粮食总产量: {total_grain_yield/1000:.1f} 吨  |  经作总产量: {total_cash_yield/1000:.1f} 吨")
    print(f"  平均粮食产量: {avg_grain_yield:.1f} kg/户  |  平均经作产量: {avg_cash_yield:.1f} kg/户")
    print(f"  生计策略分布:")
    print(f"    纯农户: 种粮{pure_grain} / 种经作{pure_cash}")
    print(f"    兼业户: 种粮{pluri_grain} / 种经作{pluri_cash}")
    print(f"    非农户: 种粮{non_grain} / 种经作{non_cash}")
    
    # 抽样检查一些种粮农户的产量
    if step == 0:
        print(f"\n  前5个种粮农户详情:")
        for f in grain_farmers[:5]:
            area = getattr(f, 'get_operated_land_area', lambda: 0)()
            print(f"    ID={f.unique_id}, 策略={f.livelihood_strategy.name}, 产量={getattr(f, 'last_yield', 0):.1f}kg, 经营面积={area:.1f}亩")

print("\n" + "=" * 60)
print("诊断完成")
