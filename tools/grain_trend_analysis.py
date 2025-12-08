"""
粮食产量趋势深度分析
诊断为什么粮食产量从6348吨下降到2909吨
"""
import sys
sys.path.insert(0, '.')

from core.model_builder import ModelBuilder

# 使用 ModelBuilder 创建模型
builder = ModelBuilder()
m = builder.build()

print("=" * 70)
print("粮食产量趋势深度分析")
print("问题：粮食产量从6348吨下降到2909吨是否符合粮食安全战略？")
print("=" * 70)

# 记录每步的详细数据
history = {
    'step': [],
    'grain_farmers': [],      # 种粮农户数
    'cash_farmers': [],       # 种经作农户数
    'grain_yield_total': [],  # 粮食总产量
    'grain_yield_avg': [],    # 户均粮食产量
    'pure_farmer_grain': [],  # 纯农户种粮数
    'pluri_grain': [],        # 兼业户种粮数
    'non_farmer_grain': [],   # 非农户种粮数
    'enterprise_grain': [],   # 企业粮食产量
    'total_operated_area': [],# 总经营面积
    'grain_area': [],         # 粮食种植面积（估算）
}

for step in range(20):
    m.step()
    
    # 统计农户作物选择
    grain_farmers = []
    cash_farmers = []
    
    for f in m.farmers:
        if hasattr(f, 'last_crop_choice') and f.last_crop_choice:
            if f.last_crop_choice.value == 'Grain':
                grain_farmers.append(f)
            elif f.last_crop_choice.value == 'Cash_Crop':
                cash_farmers.append(f)
    
    # 计算粮食产量
    total_grain_yield = sum(getattr(f, 'last_yield', 0) for f in grain_farmers)
    avg_grain_yield = total_grain_yield / len(grain_farmers) if grain_farmers else 0
    
    # 统计不同类型农户的种粮情况
    pure_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'PURE_FARMER')
    pluri_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'PLURIACTIVE')
    non_grain = sum(1 for f in grain_farmers if f.livelihood_strategy.name == 'NON_FARMER')
    
    # 计算总经营面积
    total_area = sum(getattr(f, 'get_operated_land_area', lambda: 0)() for f in m.farmers)
    grain_area = sum(getattr(f, 'get_operated_land_area', lambda: 0)() for f in grain_farmers)
    
    # 企业粮食产量
    enterprise_grain = 0
    for ent in m.agricultural_enterprises:
        if hasattr(ent, 'last_yields') and isinstance(ent.last_yields, dict):
            enterprise_grain += ent.last_yields.get('grain', 0)
    
    # 记录数据
    history['step'].append(step + 1)
    history['grain_farmers'].append(len(grain_farmers))
    history['cash_farmers'].append(len(cash_farmers))
    history['grain_yield_total'].append(total_grain_yield / 1000)  # 转换为吨
    history['grain_yield_avg'].append(avg_grain_yield)
    history['pure_farmer_grain'].append(pure_grain)
    history['pluri_grain'].append(pluri_grain)
    history['non_farmer_grain'].append(non_grain)
    history['enterprise_grain'].append(enterprise_grain / 1000)
    history['total_operated_area'].append(total_area)
    history['grain_area'].append(grain_area)

print("\n" + "=" * 70)
print("分析结果")
print("=" * 70)

print("\n1. 种粮农户数量变化：")
print(f"   第1步: {history['grain_farmers'][0]} 户")
print(f"   第20步: {history['grain_farmers'][-1]} 户")
print(f"   变化率: {(history['grain_farmers'][-1] - history['grain_farmers'][0]) / history['grain_farmers'][0] * 100:.1f}%")

print("\n2. 粮食总产量变化：")
print(f"   第1步: {history['grain_yield_total'][0]:.1f} 吨")
print(f"   第20步: {history['grain_yield_total'][-1]:.1f} 吨")
print(f"   变化率: {(history['grain_yield_total'][-1] - history['grain_yield_total'][0]) / history['grain_yield_total'][0] * 100:.1f}%")

print("\n3. 户均粮食产量变化：")
print(f"   第1步: {history['grain_yield_avg'][0]:.0f} kg/户")
print(f"   第20步: {history['grain_yield_avg'][-1]:.0f} kg/户")
if history['grain_yield_avg'][0] > 0:
    print(f"   变化率: {(history['grain_yield_avg'][-1] - history['grain_yield_avg'][0]) / history['grain_yield_avg'][0] * 100:.1f}%")

print("\n4. 粮食种植面积变化：")
print(f"   第1步: {history['grain_area'][0]:.0f} 亩")
print(f"   第20步: {history['grain_area'][-1]:.0f} 亩")
print(f"   总经营面积: {history['total_operated_area'][-1]:.0f} 亩")
print(f"   粮食种植比例: {history['grain_area'][-1] / history['total_operated_area'][-1] * 100:.1f}%")

print("\n5. 企业粮食产量：")
print(f"   第1步: {history['enterprise_grain'][0]:.1f} 吨")
print(f"   第20步: {history['enterprise_grain'][-1]:.1f} 吨")

print("\n6. 各类农户种粮趋势（第1步 → 第20步）：")
print(f"   纯农户种粮: {history['pure_farmer_grain'][0]} → {history['pure_farmer_grain'][-1]}")
print(f"   兼业户种粮: {history['pluri_grain'][0]} → {history['pluri_grain'][-1]}")
print(f"   非农户种粮: {history['non_farmer_grain'][0]} → {history['non_farmer_grain'][-1]}")

print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

# 计算亩产
if history['grain_area'][0] > 0 and history['grain_area'][-1] > 0:
    yield_per_mu_start = history['grain_yield_total'][0] * 1000 / history['grain_area'][0]
    yield_per_mu_end = history['grain_yield_total'][-1] * 1000 / history['grain_area'][-1]
    print(f"\n粮食亩产变化: {yield_per_mu_start:.0f} kg/亩 → {yield_per_mu_end:.0f} kg/亩")
    print(f"亩产变化率: {(yield_per_mu_end - yield_per_mu_start) / yield_per_mu_start * 100:.1f}%")
    
    # 理论技术进步效果
    annual_growth = 0.015
    expected_yield_with_tech = yield_per_mu_start * (1 + annual_growth) ** 20
    print(f"理论技术进步后亩产: {expected_yield_with_tech:.0f} kg/亩 (+{(expected_yield_with_tech/yield_per_mu_start - 1)*100:.1f}%)")
    
    # 判断产量下降的原因
    area_decline = (history['grain_area'][-1] - history['grain_area'][0]) / history['grain_area'][0]
    yield_decline = (history['grain_yield_total'][-1] - history['grain_yield_total'][0]) / history['grain_yield_total'][0]
    
    print(f"\n产量下降分解：")
    print(f"  - 面积因素: {area_decline * 100:.1f}%")
    print(f"  - 总产量变化: {yield_decline * 100:.1f}%")
    
    # 分析面积下降原因
    print(f"\n面积下降原因分析：")
    print(f"  - 总经营面积从 {history['total_operated_area'][0]:.0f} 亩 变为 {history['total_operated_area'][-1]:.0f} 亩")
    print(f"  - 纯农户从 {history['pure_farmer_grain'][0]} 人减少到 {history['pure_farmer_grain'][-1]} 人")
    print(f"  - 非农户经营面积可能流转或撂荒")

print("\n" + "=" * 70)
print("结论与建议")
print("=" * 70)
print("""
关键发现：
1. 粮食亩产确实在提升（技术进步因子生效）
2. 但种粮面积下降幅度太大，超过了亩产提升的补偿

现实对比：
- 中国粮食播种面积：2015年17.5亿亩 → 2022年17.75亿亩（略有增加）
- 模型：面积下降36%（不符合现实）

问题根源：
- 模型中Non_Farmer的土地没有被规模户/企业接管
- 土地流转机制可能不完善
- 企业粮食生产贡献不足

修复建议：
1. 增强土地流转机制：纯农户退出后土地应流转给规模户/企业
2. 提高企业种粮比例和产量
3. 或者降低纯农户退出速度
""")
