# -*- coding: utf-8 -*-
"""
计量单位常量与转换函数
Unit Constants and Conversion Functions

本文件定义模型中使用的所有计量单位常量，防止硬编码和单位混乱。
所有模块在涉及单位转换时必须引用此文件中的常量。

版本: v1.0
日期: 2026-01-02
"""

# ============================================================================
# 土地单位 (Land Units)
# ============================================================================

MU_PER_HA = 15              # 1公顷 = 15亩
HA_PER_MU = 1 / 15          # 1亩 = 1/15公顷 ≈ 0.0667公顷
SQM_PER_MU = 666.67         # 1亩 = 666.67平方米

# ============================================================================
# 重量单位 (Weight Units)
# ============================================================================

KG_PER_TON = 1000           # 1吨 = 1000千克
TON_PER_KG = 0.001          # 1千克 = 0.001吨
KG_PER_WAN_TON = 10_000_000 # 1万吨 = 10^7千克
WAN_TON_PER_KG = 1e-7       # 1千克 = 10^-7万吨

JIN_PER_KG = 2              # 1千克 = 2斤 (中国传统单位)
KG_PER_JIN = 0.5            # 1斤 = 0.5千克

# ============================================================================
# 货币单位 (Currency Units)
# ============================================================================

YUAN_PER_WAN_YUAN = 10_000          # 1万元 = 10000元
YUAN_PER_YI_YUAN = 100_000_000      # 1亿元 = 10^8元
WAN_YUAN_PER_YI_YUAN = 10_000       # 1亿元 = 10000万元

# ============================================================================
# 时间单位 (Time Units)
# ============================================================================

DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12
WORKING_DAYS_PER_MONTH = 22  # 农业劳动日/月

# ============================================================================
# 模型默认参数 (Default Parameters)
# 警告: 仅用于初始化，实际应从配置文件读取
# ============================================================================

DEFAULT_GRAIN_YIELD_KG_PER_MU = 450     # 粮食单产 kg/亩
DEFAULT_CASH_CROP_YIELD_KG_PER_MU = 500 # 经济作物单产 kg/亩 (折算)
DEFAULT_GRAIN_PRICE_YUAN_PER_KG = 2.5   # 粮食价格 元/kg
DEFAULT_RENTAL_YUAN_PER_MU_YEAR = 500   # 土地租金 元/亩/年
DEFAULT_SUBSIDY_YUAN_PER_MU = 100       # 种粮补贴 元/亩
DEFAULT_DAILY_WAGE_YUAN = 150           # 日工资 元/天

# ============================================================================
# 合理值范围 (Valid Ranges) - 用于数据验证
# ============================================================================

VALID_RANGES = {
    'land_transfer_rate': (0.0, 100.0),          # 土地流转率 %
    'grain_price_yuan_per_kg': (1.0, 6.0),       # 粮价 元/kg
    'rental_yuan_per_mu': (50, 2000),            # 租金 元/亩/年
    'yield_kg_per_mu': (100, 1000),              # 单产 kg/亩
    'subsidy_yuan_per_mu': (0, 500),             # 补贴 元/亩
    'daily_wage_yuan': (50, 500),                # 日工资 元/天
    'farm_size_mu': (0.1, 10000),                # 经营规模 亩
}

# ============================================================================
# 单位转换函数 (Conversion Functions)
# ============================================================================

def kg_to_ton(kg: float) -> float:
    """千克转换为吨"""
    return kg * TON_PER_KG

def ton_to_kg(ton: float) -> float:
    """吨转换为千克"""
    return ton * KG_PER_TON

def kg_to_wan_ton(kg: float) -> float:
    """千克转换为万吨"""
    return kg * WAN_TON_PER_KG

def wan_ton_to_kg(wan_ton: float) -> float:
    """万吨转换为千克"""
    return wan_ton * KG_PER_WAN_TON

def yuan_to_yi_yuan(yuan: float) -> float:
    """元转换为亿元"""
    return yuan / YUAN_PER_YI_YUAN

def yi_yuan_to_yuan(yi_yuan: float) -> float:
    """亿元转换为元"""
    return yi_yuan * YUAN_PER_YI_YUAN

def mu_to_ha(mu: float) -> float:
    """亩转换为公顷"""
    return mu * HA_PER_MU

def ha_to_mu(ha: float) -> float:
    """公顷转换为亩"""
    return ha * MU_PER_HA

# ============================================================================
# 数据验证函数 (Validation Functions)
# ============================================================================

def validate_in_range(value: float, var_name: str, warn_only: bool = False) -> float:
    """
    验证数值是否在合理范围内
    
    Args:
        value: 待验证的值
        var_name: 变量名（用于查找合理范围）
        warn_only: 如果为True，仅发出警告而不抛出异常
    
    Returns:
        原始值（通过验证）
    
    Raises:
        ValueError: 如果值超出范围且warn_only=False
    """
    if var_name not in VALID_RANGES:
        return value
    
    min_val, max_val = VALID_RANGES[var_name]
    
    if not (min_val <= value <= max_val):
        msg = f"⚠️ {var_name}={value:.2f} 超出合理范围 [{min_val}, {max_val}]"
        if warn_only:
            import warnings
            warnings.warn(msg)
        else:
            raise ValueError(msg)
    
    return value

def validate_land_transfer_rate(rate: float, strict: bool = True) -> float:
    """
    专用：验证土地流转率在0-100%之间
    
    Args:
        rate: 土地流转率（百分比值）
        strict: 如果为True，超过100%时抛出异常；否则裁剪到100%
    
    Returns:
        验证/裁剪后的流转率
    """
    if rate < 0:
        raise ValueError(f"土地流转率不能为负: {rate:.2f}%")
    
    if rate > 100:
        if strict:
            raise ValueError(f"土地流转率超过100%: {rate:.2f}%，请检查计算逻辑")
        else:
            import warnings
            warnings.warn(f"土地流转率 {rate:.2f}% 被裁剪到 100%")
            return 100.0
    
    return rate

def validate_positive(value: float, name: str = "值") -> float:
    """验证数值为正数"""
    if value <= 0:
        raise ValueError(f"{name} 必须为正数，当前值: {value}")
    return value

def validate_non_negative(value: float, name: str = "值") -> float:
    """验证数值为非负数"""
    if value < 0:
        raise ValueError(f"{name} 不能为负数，当前值: {value}")
    return value

# ============================================================================
# 格式化显示函数 (Display Formatting)
# ============================================================================

def format_large_number(value: float, precision: int = 2) -> str:
    """
    大数值格式化为易读形式
    
    Args:
        value: 数值
        precision: 小数位数
    
    Returns:
        格式化字符串，如 "1.23亿" 或 "456.78万"
    """
    if abs(value) >= YUAN_PER_YI_YUAN:
        return f"{value / YUAN_PER_YI_YUAN:.{precision}f}亿"
    elif abs(value) >= YUAN_PER_WAN_YUAN:
        return f"{value / YUAN_PER_WAN_YUAN:.{precision}f}万"
    else:
        return f"{value:.{precision}f}"

def format_kg(kg: float, precision: int = 2) -> str:
    """
    千克格式化为易读形式
    
    Args:
        kg: 千克数
        precision: 小数位数
    
    Returns:
        格式化字符串，如 "1.23万吨" 或 "456吨"
    """
    if abs(kg) >= KG_PER_WAN_TON:
        return f"{kg / KG_PER_WAN_TON:.{precision}f}万吨"
    elif abs(kg) >= KG_PER_TON:
        return f"{kg / KG_PER_TON:.{precision}f}吨"
    else:
        return f"{kg:.{precision}f}kg"

# ============================================================================
# 模块自测试
# ============================================================================

if __name__ == "__main__":
    print("=== 单位常量验证 ===")
    print(f"1公顷 = {MU_PER_HA}亩")
    print(f"1吨 = {KG_PER_TON}kg")
    print(f"1万吨 = {KG_PER_WAN_TON}kg")
    
    print("\n=== 转换函数测试 ===")
    print(f"1000kg = {kg_to_ton(1000)}吨")
    print(f"5吨 = {ton_to_kg(5)}kg")
    print(f"10亿元 = {format_large_number(1e9)}")
    print(f"500万kg = {format_kg(5e6)}")
    
    print("\n=== 验证函数测试 ===")
    try:
        validate_land_transfer_rate(150, strict=True)
    except ValueError as e:
        print(f"预期异常: {e}")
    
    rate = validate_land_transfer_rate(150, strict=False)
    print(f"非严格模式裁剪: 150% -> {rate}%")
    
    print("\n✓ 所有测试通过")
