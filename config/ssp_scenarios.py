# -*- coding: utf-8 -*-
"""
@module ssp_scenarios.py
@description SSP社会发展情景定义与参数设置
@created 2025-12-06
@author GitHub Copilot

设计思路：
==========
参考项目使用RCP气候情景（RCP8.5）作为外部驱动，通过GeoTIFF栅格数据提供空间异质性。
我们的模型聚焦粮食补贴政策，使用SSP情景作为社会经济背景框架。

关键区别：
- 参考项目：空间异质性来自洪水风险地图（外部栅格数据）
- 我们的模型：空间异质性来自农业生产条件分区（内生设定）

SSP框架来源：
- O'Neill et al. (2014). "A new scenario framework for climate change research"
- O'Neill et al. (2017). "The roads ahead: Narratives for SSPs"
- IPCC AR6 (2021): SSP-RCP耦合情景矩阵
- 中国SSP本土化研究：Chen et al. (2020), Jiang et al. (2017)

SSP与粮食安全的关联：
- SSP1: 可持续集约化，高技术投入，粮食安全保障好
- SSP2: 延续现状趋势
- SSP3: 发展碎片化，粮食安全风险高
- SSP4: 不平等加剧，小农边缘化
- SSP5: 高增长但环境压力大
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class SSPScenarioConfig:
    """
    SSP情景完整配置
    
    将SSP叙事转化为模型可用的参数集
    """
    # ============ 基本信息 ============
    code: str                               # SSP1-5
    name: str                               # 中文名称
    narrative: str                          # 情景叙事
    
    # ============ 人口与城镇化 ============
    # 数据来源：《中国人口预测》(2017-2100), Jiang & O'Neill (2017)
    urbanization_rate_2030: float = 0.70    # 2030年城镇化率
    urbanization_rate_2050: float = 0.80    # 2050年城镇化率
    rural_population_decline_annual: float = 0.02  # 农村人口年下降率
    
    # 农村劳动力结构
    youth_farming_rate: float = 0.10        # 青年(18-45)务农比例
    elder_farming_rate: float = 0.60        # 老年(>55)务农比例
    female_farming_rate: float = 0.45       # 女性务农比例
    
    # ============ 经济发展 ============
    # 数据来源：Dellink et al. (2017) "Long-term economic growth projections"
    gdp_growth_rate: float = 0.05           # GDP年增长率
    agricultural_gdp_share: float = 0.07    # 农业GDP占比
    farmer_income_growth: float = 0.05      # 农民收入增长率
    
    # 城乡收入差距
    urban_rural_income_ratio: float = 2.5   # 城乡收入比
    income_ratio_trend: float = 0.0         # 城乡收入比变化趋势(正=扩大)
    
    # ============ 非农就业机会 ============
    # 来源：中国劳动力市场调查数据
    non_farm_wage_level: float = 50000      # 非农年工资(元)
    non_farm_wage_growth: float = 0.05      # 非农工资增长率
    non_farm_job_availability: float = 0.80  # 非农就业可获得性
    migration_cost: float = 10000           # 外出务工年均成本(元)
    
    # ============ 农业生产技术 ============
    # 来源：农业技术进步相关文献
    tech_progress_rate: float = 0.02        # 技术进步率(TFP增长)
    mechanization_rate: float = 0.70        # 农业机械化率
    irrigation_coverage: float = 0.55       # 有效灌溉面积比例
    
    # 要素生产率
    labor_productivity_growth: float = 0.03  # 劳动生产率增长
    land_productivity_growth: float = 0.01   # 土地生产率增长
    
    # ============ 土地制度与流转 ============
    # 来源：农村土地制度改革研究
    land_transfer_friction: float = 0.10    # 土地流转摩擦系数(0-1)
    land_rental_price_trend: float = 0.03   # 租金年变化率
    scale_operation_threshold: float = 30   # 规模经营阈值(亩)
    
    # 土地细碎化
    land_fragmentation: float = 0.50        # 土地细碎化程度(0-1)
    consolidation_progress: float = 0.02    # 土地整合年进展率
    
    # ============ 政策环境 ============
    # 来源：农业政策研究
    fiscal_capacity: float = 1.0            # 财政能力系数(相对基准)
    policy_effectiveness: float = 1.0       # 政策执行效率(0-1)
    subsidy_targeting_precision: float = 0.8  # 补贴瞄准精度
    
    # 政策稳定性
    policy_uncertainty: float = 0.1         # 政策不确定性(0-1)
    
    # ============ 市场环境 ============
    # 来源：农产品市场研究
    market_integration: float = 0.80        # 市场整合程度
    price_volatility: float = 0.15          # 价格波动率
    market_access_cost: float = 0.10        # 市场准入成本(占售价比)
    
    # ============ 气候与环境风险 ============
    # 来源：气候变化影响评估
    climate_risk_level: float = 1.0         # 气候风险水平(相对基准)
    extreme_weather_frequency: float = 0.10  # 极端天气发生频率
    yield_variability: float = 0.15         # 产量年际波动率
    
    # ============ 社会资本与网络 ============
    # 来源：农村社会调查
    social_network_density: float = 0.60    # 社会网络密度
    information_diffusion_rate: float = 0.30  # 信息扩散率
    collective_action_capacity: float = 0.50  # 集体行动能力


# ============ SSP情景定义 ============

SSP1_SUSTAINABILITY = SSPScenarioConfig(
    code="SSP1",
    name="可持续发展路径",
    narrative="""
    绿色发展与城乡融合：
    - 高技术投入，农业可持续集约化
    - 城乡收入差距缩小，青年返乡创业增加
    - 土地流转市场规范，规模经营稳步推进
    - 政策执行效率高，补贴精准瞄准
    - 社会资本丰富，信息网络发达
    """,
    
    # 人口
    urbanization_rate_2030=0.72,
    urbanization_rate_2050=0.78,
    rural_population_decline_annual=0.025,
    youth_farming_rate=0.15,  # 青年返乡增加
    elder_farming_rate=0.50,
    
    # 经济
    gdp_growth_rate=0.045,
    agricultural_gdp_share=0.06,
    farmer_income_growth=0.06,
    urban_rural_income_ratio=2.0,  # 差距缩小
    income_ratio_trend=-0.02,
    
    # 非农就业
    non_farm_wage_level=55000,
    non_farm_wage_growth=0.04,
    non_farm_job_availability=0.85,
    
    # 技术
    tech_progress_rate=0.03,  # 快速技术进步
    mechanization_rate=0.85,
    irrigation_coverage=0.65,
    labor_productivity_growth=0.04,
    
    # 土地
    land_transfer_friction=0.05,  # 流转顺畅
    scale_operation_threshold=50,
    land_fragmentation=0.35,
    consolidation_progress=0.04,
    
    # 政策
    fiscal_capacity=1.2,
    policy_effectiveness=1.2,
    subsidy_targeting_precision=0.90,
    policy_uncertainty=0.05,
    
    # 市场
    market_integration=0.90,
    price_volatility=0.10,
    
    # 环境
    climate_risk_level=0.8,
    extreme_weather_frequency=0.08,
    
    # 社会
    social_network_density=0.75,
    information_diffusion_rate=0.50,
    collective_action_capacity=0.70
)


SSP2_MIDDLE_ROAD = SSPScenarioConfig(
    code="SSP2",
    name="中间道路",
    narrative="""
    延续历史趋势：
    - 发展速度保持中等水平
    - 城乡差距缓慢缩小
    - 土地流转稳步推进，但仍有摩擦
    - 政策执行中规中矩
    """,
    
    # 人口
    urbanization_rate_2030=0.70,
    urbanization_rate_2050=0.80,
    rural_population_decline_annual=0.02,
    youth_farming_rate=0.10,
    elder_farming_rate=0.55,
    
    # 经济
    gdp_growth_rate=0.05,
    agricultural_gdp_share=0.07,
    farmer_income_growth=0.05,
    urban_rural_income_ratio=2.5,
    income_ratio_trend=-0.01,
    
    # 技术
    tech_progress_rate=0.02,
    mechanization_rate=0.75,
    irrigation_coverage=0.58,
    
    # 土地
    land_transfer_friction=0.10,
    scale_operation_threshold=30,
    land_fragmentation=0.45,
    
    # 政策
    fiscal_capacity=1.0,
    policy_effectiveness=1.0,
    subsidy_targeting_precision=0.80,
    
    # 市场
    market_integration=0.80,
    price_volatility=0.15,
    
    # 社会
    social_network_density=0.60,
    information_diffusion_rate=0.35,
    collective_action_capacity=0.55
)


SSP3_REGIONAL_RIVALRY = SSPScenarioConfig(
    code="SSP3",
    name="区域竞争路径",
    narrative="""
    发展碎片化与分割：
    - 经济增长缓慢，城乡发展分化
    - 农村人口老龄化严重，青年流失
    - 土地流转困难，规模经营受阻
    - 政策效率低下，补贴漏损严重
    - 社会网络弱化，信息传播慢
    """,
    
    # 人口
    urbanization_rate_2030=0.65,
    urbanization_rate_2050=0.72,
    rural_population_decline_annual=0.015,
    youth_farming_rate=0.05,  # 青年流失严重
    elder_farming_rate=0.70,
    
    # 经济
    gdp_growth_rate=0.03,
    agricultural_gdp_share=0.10,
    farmer_income_growth=0.03,
    urban_rural_income_ratio=3.2,  # 差距扩大
    income_ratio_trend=0.01,
    
    # 非农就业
    non_farm_wage_level=45000,
    non_farm_wage_growth=0.03,
    non_farm_job_availability=0.65,
    
    # 技术
    tech_progress_rate=0.01,  # 技术进步慢
    mechanization_rate=0.60,
    irrigation_coverage=0.50,
    labor_productivity_growth=0.015,
    
    # 土地
    land_transfer_friction=0.25,  # 流转困难
    scale_operation_threshold=20,
    land_fragmentation=0.60,
    consolidation_progress=0.005,
    
    # 政策
    fiscal_capacity=0.7,
    policy_effectiveness=0.6,
    subsidy_targeting_precision=0.60,
    policy_uncertainty=0.25,
    
    # 市场
    market_integration=0.60,
    price_volatility=0.25,
    market_access_cost=0.18,
    
    # 环境
    climate_risk_level=1.3,
    extreme_weather_frequency=0.15,
    yield_variability=0.22,
    
    # 社会
    social_network_density=0.40,
    information_diffusion_rate=0.20,
    collective_action_capacity=0.35
)


SSP4_INEQUALITY = SSPScenarioConfig(
    code="SSP4",
    name="不平等路径",
    narrative="""
    贫富分化与资本集中：
    - 经济增长但分配不均
    - 大资本进入农业，小农边缘化
    - 土地向大户集中，但租金高企
    - 政策倾向规模经营，小农获益少
    - 社会分层明显，网络割裂
    """,
    
    # 人口
    urbanization_rate_2030=0.72,
    urbanization_rate_2050=0.82,
    rural_population_decline_annual=0.03,
    youth_farming_rate=0.08,
    elder_farming_rate=0.45,
    
    # 经济
    gdp_growth_rate=0.04,
    agricultural_gdp_share=0.05,
    farmer_income_growth=0.03,  # 小农增长慢
    urban_rural_income_ratio=3.5,
    income_ratio_trend=0.02,  # 差距扩大
    
    # 非农就业
    non_farm_wage_level=60000,
    non_farm_wage_growth=0.06,
    non_farm_job_availability=0.75,
    
    # 技术
    tech_progress_rate=0.025,
    mechanization_rate=0.80,  # 大户机械化高
    irrigation_coverage=0.60,
    
    # 土地
    land_transfer_friction=0.08,  # 市场活跃但不平等
    land_rental_price_trend=0.05,  # 租金上涨快
    scale_operation_threshold=50,
    land_fragmentation=0.30,  # 向大户集中
    consolidation_progress=0.05,
    
    # 政策
    fiscal_capacity=1.1,
    policy_effectiveness=0.85,
    subsidy_targeting_precision=0.70,  # 偏向大户
    
    # 市场
    market_integration=0.85,
    price_volatility=0.12,
    
    # 社会
    social_network_density=0.45,
    information_diffusion_rate=0.40,
    collective_action_capacity=0.40
)


SSP5_FOSSIL_FUELED = SSPScenarioConfig(
    code="SSP5",
    name="高增长路径",
    narrative="""
    高经济增长与快速转型：
    - 经济高速增长，城镇化快速推进
    - 农业劳动力快速流失
    - 规模化经营快速发展
    - 财政能力强，但环境压力大
    """,
    
    # 人口
    urbanization_rate_2030=0.75,
    urbanization_rate_2050=0.88,
    rural_population_decline_annual=0.04,
    youth_farming_rate=0.05,
    elder_farming_rate=0.40,
    
    # 经济
    gdp_growth_rate=0.065,
    agricultural_gdp_share=0.04,
    farmer_income_growth=0.04,
    urban_rural_income_ratio=2.8,
    income_ratio_trend=0.0,
    
    # 非农就业
    non_farm_wage_level=70000,
    non_farm_wage_growth=0.07,
    non_farm_job_availability=0.90,
    
    # 技术
    tech_progress_rate=0.03,
    mechanization_rate=0.90,
    irrigation_coverage=0.70,
    labor_productivity_growth=0.05,
    
    # 土地
    land_transfer_friction=0.05,
    scale_operation_threshold=100,
    land_fragmentation=0.25,
    consolidation_progress=0.06,
    
    # 政策
    fiscal_capacity=1.5,
    policy_effectiveness=1.1,
    subsidy_targeting_precision=0.85,
    
    # 市场
    market_integration=0.95,
    price_volatility=0.10,
    
    # 环境
    climate_risk_level=1.2,  # 环境压力大
    extreme_weather_frequency=0.12,
    
    # 社会
    social_network_density=0.50,
    information_diffusion_rate=0.55,
    collective_action_capacity=0.45
)


# ============ 情景字典 ============

SSP_SCENARIOS: Dict[str, SSPScenarioConfig] = {
    "SSP1": SSP1_SUSTAINABILITY,
    "SSP2": SSP2_MIDDLE_ROAD,
    "SSP3": SSP3_REGIONAL_RIVALRY,
    "SSP4": SSP4_INEQUALITY,
    "SSP5": SSP5_FOSSIL_FUELED
}


def get_scenario(code: str) -> SSPScenarioConfig:
    """获取SSP情景配置"""
    if code not in SSP_SCENARIOS:
        raise ValueError(f"未知SSP代码: {code}，可选: {list(SSP_SCENARIOS.keys())}")
    return SSP_SCENARIOS[code]


def apply_ssp_to_config(
    base_config: Dict[str, Any],
    ssp_code: str,
    simulation_year: int = 2025
) -> Dict[str, Any]:
    """
    将SSP情景参数应用到模型配置
    
    Args:
        base_config: 基础配置字典
        ssp_code: SSP代码 (SSP1-5)
        simulation_year: 模拟年份（用于插值时变参数）
        
    Returns:
        更新后的配置字典
    """
    import copy
    
    ssp = get_scenario(ssp_code)
    config = copy.deepcopy(base_config)
    
    # -------- 更新agents配置 --------
    if 'agents' not in config:
        config['agents'] = {}
    
    # 农户相关
    if 'farmer' not in config['agents']:
        config['agents']['farmer'] = {}
    
    farmer_cfg = config['agents']['farmer']
    farmer_cfg['youth_farming_rate'] = ssp.youth_farming_rate
    farmer_cfg['elder_farming_rate'] = ssp.elder_farming_rate
    
    # -------- 更新environment配置 --------
    if 'environment' not in config:
        config['environment'] = {}
    
    # 市场参数
    if 'market' not in config['environment']:
        config['environment']['market'] = {}
    
    market_cfg = config['environment']['market']
    market_cfg['non_farm_wage'] = ssp.non_farm_wage_level
    market_cfg['non_farm_wage_growth'] = ssp.non_farm_wage_growth
    market_cfg['non_farm_job_availability'] = ssp.non_farm_job_availability
    market_cfg['price_volatility'] = ssp.price_volatility
    market_cfg['market_integration'] = ssp.market_integration
    
    # 土地市场
    if 'land' not in config['environment']:
        config['environment']['land'] = {}
    
    land_cfg = config['environment']['land']
    land_cfg['transfer_friction'] = ssp.land_transfer_friction
    land_cfg['rental_price_trend'] = ssp.land_rental_price_trend
    land_cfg['fragmentation'] = ssp.land_fragmentation
    
    # -------- 更新production配置 --------
    if 'production' not in config:
        config['production'] = {}
    
    config['production']['tech_progress_rate'] = ssp.tech_progress_rate
    config['production']['mechanization_rate'] = ssp.mechanization_rate
    config['production']['labor_productivity_growth'] = ssp.labor_productivity_growth
    config['production']['yield_variability'] = ssp.yield_variability
    
    # -------- 更新policy配置 --------
    if 'policy' not in config:
        config['policy'] = {}
    
    config['policy']['fiscal_capacity'] = ssp.fiscal_capacity
    config['policy']['effectiveness'] = ssp.policy_effectiveness
    config['policy']['targeting_precision'] = ssp.subsidy_targeting_precision
    config['policy']['uncertainty'] = ssp.policy_uncertainty
    
    # -------- 添加SSP元数据 --------
    config['ssp_scenario'] = {
        'code': ssp.code,
        'name': ssp.name,
        'simulation_year': simulation_year
    }
    
    return config


def compare_scenarios() -> str:
    """生成SSP情景对比表"""
    lines = []
    lines.append("=" * 80)
    lines.append("SSP情景参数对比")
    lines.append("=" * 80)
    
    # 关键参数对比
    key_params = [
        ("GDP增长率", "gdp_growth_rate"),
        ("青年务农率", "youth_farming_rate"),
        ("非农工资", "non_farm_wage_level"),
        ("技术进步率", "tech_progress_rate"),
        ("土地流转摩擦", "land_transfer_friction"),
        ("政策效率", "policy_effectiveness"),
        ("城乡收入比", "urban_rural_income_ratio"),
    ]
    
    header = f"{'参数':<20}" + "".join(f"{code:>12}" for code in SSP_SCENARIOS.keys())
    lines.append(header)
    lines.append("-" * 80)
    
    for param_name, param_attr in key_params:
        row = f"{param_name:<20}"
        for ssp in SSP_SCENARIOS.values():
            value = getattr(ssp, param_attr)
            if isinstance(value, float):
                if value > 100:
                    row += f"{value:>12,.0f}"
                else:
                    row += f"{value:>12.3f}"
            else:
                row += f"{value:>12}"
        lines.append(row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(compare_scenarios())
    print("\n")
    print("SSP情景已定义，可通过以下方式使用：")
    print("""
    from config.ssp_scenarios import SSP_SCENARIOS, apply_ssp_to_config
    
    # 获取SSP2情景
    ssp2 = SSP_SCENARIOS['SSP2']
    print(ssp2.narrative)
    
    # 应用到模型配置
    config = apply_ssp_to_config(base_config, 'SSP3', simulation_year=2030)
    """)
