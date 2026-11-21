# -*- coding: utf-8 -*-
"""
@theory Environmental Accountant
@created 2025-12-28

负责核算农业生产的环境外部性，包括：
1. 碳排放 (Carbon Footprint)
2. 氮流失 (Nitrogen Leaching)

@reference:
- Li et al. (2020, PNAS): Agro-environmental policies
- Deng et al. (2024, Nature Food): N-loss from fragmented croplands

@deprecated 2026-05: 碳排放和氮流失计算暂时禁用
原因: 当前公式使用硬编码线性假设，缺乏衢州本地实证数据支撑。
TODO: 待获取衢州本地碳排放系数和氮淋溶实测数据后重新实现。
"""


class EnvironmentalAccountant:
    def __init__(self, model):
        self.model = model
        self.history = []
        
        # TODO: 暂无可靠数据支撑以下排放因子，待获取衢州本地数据后重新校准
        # @theory 排放因子 (Emission Factors)
        # 基于中国农业生命周期评价(LCA)数据
        # self.emission_factors = {
        #     'grain': {
        #         'fertilizer_input': 18.0,  # kg N/mu (粮食作物平均氮肥投入)
        #         'carbon_intensity': 0.45,  # kg CO2e / kg yield (含化肥生产、运输、施用)
        #         'tillage_carbon': 40.0,    # kg CO2e / mu (耕作排放)
        #         'n_leaching_rate': 0.20    # 氮淋溶率 20%
        #     },
        #     'cash_crop': {
        #         'fertilizer_input': 35.0,  # kg N/mu (经济作物通常更高)
        #         'carbon_intensity': 0.70,  # kg CO2e / kg yield
        #         'tillage_carbon': 60.0,    # kg CO2e / mu
        #         'n_leaching_rate': 0.25    # 氮淋溶率 25% (过量施肥导致淋溶率上升)
        #     }
        # }
        self.emission_factors = None  # 禁用，等待本地数据
        
    def calculate_annual_footprint(self):
        """
        计算全域年度环境足迹

        @deprecated: 暂无可靠数据支撑碳排放和氮流失计算，返回零值占位。
        TODO: 待获取衢州本地碳排放系数（稻作/旱作分类）和氮淋溶实测数据后重新实现。
        原公式使用硬编码线性假设（如 carbon_intensity=0.45 kg CO2e/kg yield），
        缺乏区域差异性和作物品种差异性考量。
        
        Returns:
            dict: 包含零值的环境指标占位
        """
        # TODO: 暂无可靠数据支撑碳排放计算，待获取衢州本地数据后重新实现
        # 原始计算逻辑（硬编码线性假设，已注释）:
        # total_carbon = 0.0  # kg CO2e
        # total_nitrogen_loss = 0.0  # kg N
        # total_fertilizer_use = 0.0 # kg N
        #
        # # 1. 农户排放
        # for farmer in self.model.farmers:
        #     c, n, f = self._calculate_agent_footprint(farmer)
        #     total_carbon += c
        #     total_nitrogen_loss += n
        #     total_fertilizer_use += f
        #
        # # 2. 企业排放 (如果有)
        # if hasattr(self.model, 'agricultural_enterprises'):
        #     for ent in self.model.agricultural_enterprises:
        #         c, n, f = self._calculate_agent_footprint(ent)
        #         # 企业规模化经营通常效率更高
        #         # @theory: 规模经营减少化肥浪费 (Deng et al. 2024)
        #         total_carbon += c * 0.9  # 假设减排10%
        #         total_nitrogen_loss += n * 0.85 # 假设减少流失15%
        #         total_fertilizer_use += f
        #
        # # 计算粮食总产量（从农户和企业汇总）
        # total_grain_output = 0.0
        # for farmer in self.model.farmers:
        #     total_grain_output += getattr(farmer, 'grain_output', 0)
        # if hasattr(self.model, 'agricultural_enterprises'):
        #     for ent in self.model.agricultural_enterprises:
        #         total_grain_output += getattr(ent, 'grain_output', 0)

        # 返回零值占位，保持接口兼容
        stats = {
            'step': getattr(self.model, 'step_count', 0),
            'total_carbon_emission': 0.0,  # TODO: 待本地数据支撑后恢复
            'total_nitrogen_loss': 0.0,    # TODO: 待本地数据支撑后恢复
            'total_fertilizer_input': 0.0, # TODO: 待本地数据支撑后恢复
            'carbon_intensity': 0.0,       # TODO: 待本地数据支撑后恢复
        }
        stats.update({
            'total_carbon_emission': float("nan"),
            'total_nitrogen_loss': float("nan"),
            'total_fertilizer_input': float("nan"),
            'carbon_intensity': float("nan"),
            'data_status': 'unavailable',
        })
        self.history.append(stats)
        return stats
        
    def _calculate_agent_footprint(self, agent):
        """
        计算单个主体的环境足迹
        
        @deprecated: 暂无可靠数据支撑，返回零值。
        TODO: 待获取衢州本地数据后重新实现。
        原公式问题：
        - carbon_intensity 使用全国平均值，未考虑浙江水稻为主的种植结构
        - n_leaching_rate 使用固定比例，未考虑土壤类型和灌溉方式差异
        - 细碎化对化肥投入的影响系数(1.15)缺乏实证依据
        """
        # TODO: 暂无可靠数据支撑，以下为原始计算逻辑（硬编码线性假设）
        # carbon = 0.0
        # nitrogen = 0.0
        # fertilizer = 0.0
        #
        # # 获取作物选择
        # crop_type = 'grain' # 默认
        # if hasattr(agent, 'last_crop_choice'):
        #     choice = agent.last_crop_choice
        #     if hasattr(choice, 'value'):
        #         if choice.value.lower() == 'cash_crop':
        #             crop_type = 'cash_crop'
        #     elif isinstance(choice, str) and choice.lower() == 'cash_crop':
        #         crop_type = 'cash_crop'
        #
        # # 获取经营面积
        # area = 0.0
        # if hasattr(agent, 'get_operated_land_area'):
        #     area = agent.get_operated_land_area()
        # elif hasattr(agent, 'land_patches'):
        #     area = sum(p.area for p in agent.land_patches if p.leased_to is None)
        #
        # if area <= 0:
        #     return 0, 0, 0
        #
        # # 获取产量 (用于计算碳强度的部分)
        # yield_val = getattr(agent, 'last_yield', area * 500.0) # 默认500kg
        #
        # # 计算
        # factors = self.emission_factors[crop_type]
        #
        # # 1. 化肥投入
        # input_n = area * factors['fertilizer_input']
        #
        # # @new 2024: 细碎化对化肥投入的影响
        # if hasattr(agent, 'land_patches'):
        #     avg_size = area / max(1, len(agent.land_patches))
        #     if avg_size < 1.5: # 细碎地块
        #         input_n *= 1.15 # 增加15%投入
        #
        # fertilizer = input_n
        #
        # # 2. 氮流失
        # nitrogen = input_n * factors['n_leaching_rate']
        #
        # # 3. 碳排放
        # # 产量相关 + 耕作面积相关
        # carbon = (yield_val * factors['carbon_intensity']) + (area * factors['tillage_carbon'])
        #
        # return carbon, nitrogen, fertilizer
        
        return 0.0, 0.0, 0.0
