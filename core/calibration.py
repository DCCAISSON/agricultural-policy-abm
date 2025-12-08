# -*- coding: utf-8 -*-
"""
CalibrationManager - 基于Simulated Method of Moments的模型校准管理器

理论基础：
1. Simulated Method of Moments (SMM) - 结构化模型标准校准方法
2. 《农业经济学》校准方法论
3. 中国农业统计年鉴实证数据匹配

主要功能：
- 自动校准模型参数使其与真实数据匹配
- 计算目标统计量的距离
- 优化算法选择参数
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
import yaml

from analysis.empirical_calibration import EmpiricalMomentLoader

class CalibrationManager:
    """
    模型参数校准管理器
    
    使用Simulated Method of Moments (SMM)方法，通过最小化模型生成的统计量
    与真实数据统计量之间的距离来校准参数
    """
    
    def __init__(self, config_path: str = "config/calibrated_params.json", config: Dict[str, Any] = None):
        """
        初始化校准管理器
        
        @fix 2024-11-20: 支持传入config字典，修复runner.py调用错误
        参数:
            config_path: 配置文件路径（如果config为None）
            config: 配置字典（优先使用）
        """
        # @fix: 如果传入了config字典，使用它；否则使用config_path
        if config is not None:
            # 从config字典中提取信息
            self.config = config
            self.config_path = config.get('calibration', {}).get('calibrated_params_path', "config/calibrated_params.json")
            self.master_config = config
        else:
            # 使用config_path（字符串）
            if isinstance(config_path, dict):
                # 兼容：如果config_path是dict，当作config处理
                self.config = config_path
                self.config_path = config_path.get('calibration', {}).get('calibrated_params_path', "config/calibrated_params.json")
                self.master_config = config_path
            else:
                # 正常情况：config_path是字符串
                self.config_path = config_path
                self.config = None
                self.master_config = None
        
        self.logger = self._setup_logger()
        
        # 加载基础校准参数
        self.base_params = self._load_base_params()

        # 加载主配置以定位实证数据（如果还没有加载）
        if self.master_config is None:
            self.master_config = self._load_master_config()
        calibration_cfg = self.master_config.get("calibration", {}) if self.master_config else {}
        self.parameter_controls = calibration_cfg.get("parameter_controls", {})
        self.empirical_moments = self._load_empirical_moments()
        
        # 目标统计量（基于中国农业统计年鉴2017-2023）
        self.target_moments = {
            'enterprise_type_distribution': {
                'FAMILY_FARM_LARGE': 0.60,    # 大型家庭农场比例
                'AGRI_CORPORATION': 0.25,     # 农业公司比例
                'COOPERATIVE': 0.15           # 合作社比例
            },
            'crop_structure': {
                'grain_crop_ratio': 0.70,     # 粮食作物种植比例
                'cash_crop_ratio': 0.30       # 经济作物种植比例
            },
            'land_transfer': {
                'transfer_rate': 0.3673,      # 土地流转率（2022年数据）
                'average_transfer_price': 850,  # 平均流转价格（元/亩/年）
            },
            'income_distribution': {
                'gini_coefficient': 0.375,    # 农村收入基尼系数
                'avg_farm_income': 21691,     # 农村居民人均可支配收入（2023）
            },
            'productivity_indicators': {
                'avg_yield_per_hectare': 5845.33,  # 粮食单产（公斤/公顷）
                'labor_productivity': 183.7,       # 农业劳动生产率（元/天）
            },
            'farmer_heterogeneity': {
                # 农户异质性指标（从CLES数据提取）
                'livelihood_strategy_dist': {
                    'PURE_FARMER': 0.002,      # 纯农户比例（从数据提取）
                    'PLURIACTIVE': 0.019,      # 兼业农户比例（从数据提取）
                    'NON_FARMER': 0.058        # 非农户比例（从数据提取）
                },
                'avg_agri_income_ratio': 0.17,  # 平均农业收入占比（从数据提取）
                'farming_ability_dist': {
                    'mean': 0.998,             # 耕作能力均值（从数据提取）
                    'std': 0.114               # 耕作能力标准差（从数据提取）
                }
            }
        }
        
        # 权重矩阵（用于加权最小二乘）
        self.weight_matrix = self._construct_weight_matrix()
        
        # 校准参数的搜索空间
        self.param_bounds = self._define_parameter_bounds()

    def _setup_logger(self):
        """设置日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('CalibrationManager')

    def _load_base_params(self):
        """加载基础校准参数"""
        # @fix 2024-11-20: 如果config_path是dict，直接返回；否则从文件加载
        if isinstance(self.config_path, dict):
            return self.config_path.get('calibration', {}).get('base_params', {})
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"配置文件 {self.config_path} 未找到，使用默认参数")
            return {}

    def _load_master_config(self) -> Dict:
        """加载主配置文件，获取校准相关信息"""
        master_path = Path("config/master_config.yaml")
        if not master_path.exists():
            return {}
        with master_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_empirical_moments(self) -> Dict:
        """从实证数据中提取异质性与网络统计量"""
        calibration_cfg = self.master_config.get("calibration", {}) if self.master_config else {}
        config_path = calibration_cfg.get("empirical_moment_config")
        if not config_path:
            return {}

        try:
            loader = EmpiricalMomentLoader(Path(config_path))
            moments = loader.derive_empirical_moments()
            self.logger.info("成功加载实证矩估计，用于参数校准。")
            return moments
        except Exception as exc:
            self.logger.error(f"加载实证数据失败: {exc}")
            return {}

    def _construct_weight_matrix(self):
        """
        构建权重矩阵
        
        为不同的目标统计量分配权重，重要的统计量获得更高权重
        """
        weights = {
            'enterprise_type_distribution': 2.0,  # 企业类型分布很重要
            'crop_structure': 3.0,               # 作物结构是核心
            'land_transfer': 3.5,                # 土地流转是关键机制（提高惩罚权重）
            'income_distribution': 2.0,          # 收入分配反映政策效果
            'productivity_indicators': 1.5,      # 生产率指标
            'farmer_heterogeneity': 3.0          # 农户异质性（新增，重要指标）
        }
        return weights

    def build_calibrated_parameter_payload(self) -> Dict[str, Any]:
        """
        使用实证矩估计构建参数分布，用于写回配置文件。
        """
        if not self.empirical_moments:
            raise ValueError("尚未加载实证数据，无法生成校准参数。")

        payload: Dict[str, Any] = {}
        farmer_stats = self.empirical_moments.get("farmer", {})
        village_stats = self.empirical_moments.get("village", {})

        if farmer_stats:
            farmer_payload: Dict[str, Any] = {}

            initial_capital = farmer_stats.get("initial_capital", {})
            lognormal = initial_capital.get("lognormal")
            if lognormal:
                farmer_payload["initial_capital_dist"] = {
                    "type": "lognormal",
                    "mean": lognormal["log_mean"],
                    "std": max(lognormal["log_std"], 1e-6)
                }

            ability_summary = farmer_stats.get("farming_ability", {}).get("summary")
            if ability_summary:
                farmer_payload["farming_ability_dist"] = {
                    "type": "normal",
                    "mean": ability_summary["mean"],
                    "std": max(ability_summary["std"], 1e-6)
                }

            risk_summary = farmer_stats.get("risk_aversion", {}).get("summary")
            if risk_summary:
                farmer_payload["risk_aversion_dist"] = {
                    "type": "normal",
                    "mean": risk_summary["mean"],
                    "std": max(risk_summary["std"], 1e-6)
                }

            farmland_summary = farmer_stats.get("farmland_area", {}).get("summary")
            if farmland_summary:
                farmer_payload["farmland_area_summary"] = farmland_summary

            social_degree_summary = farmer_stats.get("social_degree", {}).get("summary")
            if social_degree_summary:
                farmer_payload["social_degree_summary"] = social_degree_summary

            if farmer_payload:
                payload["farmer"] = farmer_payload

        if village_stats:
            network_payload: Dict[str, Any] = {}
            mean_degree = village_stats.get("mean_degree")
            if mean_degree:
                network_payload["mean_degree"] = mean_degree["mean"]
                network_payload["degree_distribution"] = mean_degree

            elite_share = village_stats.get("elite_share")
            if elite_share:
                network_payload["elite_share"] = elite_share["mean"]
                network_payload["elite_distribution"] = elite_share

            clustering = village_stats.get("clustering")
            if clustering:
                network_payload["clustering"] = clustering["mean"]
                network_payload["clustering_distribution"] = clustering

            if network_payload:
                payload["social_network"] = network_payload

        return payload

    def calibrate_from_empirical_data(self, output_path: str = "config/calibrated_params_enhanced.json") -> Dict[str, Any]:
        """
        根据实证数据生成并保存校准参数。
        """
        payload = self.build_calibrated_parameter_payload()
        self.save_calibrated_parameters(payload, output_path=output_path)
        return payload

    def _define_parameter_bounds(self):
        """
        定义参数搜索边界
        
        基于理论和实证研究设定合理的参数范围
        """
        bounds = {
            # 企业异质性参数
            'family_farm_efficiency_mean': (1.2, 1.6),
            'agri_corp_efficiency_mean': (1.5, 2.0),
            'cooperative_efficiency_mean': (1.3, 1.7),
            
            # 风险偏好参数
            'family_farm_risk_aversion_mean': (1.5, 2.5),
            'agri_corp_risk_aversion_mean': (0.5, 1.5),
            'cooperative_risk_aversion_mean': (1.0, 2.0),
            
            # 市场机制参数
            'base_rental_price': (600, 1000),
            'quality_multiplier': (1.2, 1.6),
            
            # 作物相关参数
            'grain_price_volatility': (0.03, 0.08),
            'cash_crop_price_volatility': (0.15, 0.35),
            
            # 技术采纳参数
            'tech_diffusion_rate': (0.05, 0.20),
            'innovation_threshold': (0.3, 0.7),
            
            # 企业参与概率（影响土地流转率）
            'family_farm_expansion_probability': (0.3, 0.5),
            'agri_corp_expansion_probability': (0.6, 0.8),
            'cooperative_expansion_probability': (0.5, 0.7),
            
            # 劳动力分配参数（影响农业收入占比）
            'wage_elasticity': (0.1, 0.5),
            'rental_price_elasticity': (0.1, 0.4),
            'risk_aversion_weight': (0.05, 0.2),
            
            # 生产函数弹性参数（新增）
            'production_alpha': (0.3, 0.5),  # 土地弹性
            'production_beta': (0.2, 0.4),   # 劳动弹性
            'production_gamma': (0.1, 0.3),  # 资本弹性
            
            # 气候冲击参数（新增）
            'climate_mean': (-0.01, 0.01),   # 气候冲击均值
            'climate_std': (0.05, 0.15),     # 气候冲击标准差
            
            # 市场调整参数（新增）
            'wage_adjustment_elasticity': (0.01, 0.1),  # 工资调整弹性
            'price_adjustment_speed': (0.1, 0.3),       # 价格调整速度
            'rent_growth_rate': (0.0, 0.1),             # 租金增长率
            
            # 前景理论参数（新增，用于行为决策）
            'loss_aversion_lambda_mean': (2.5, 3.5),   # 损失厌恶系数
            'value_function_curvature_mean': (0.45, 0.65),  # 价值函数曲率
            'probability_weighting_mean': (0.65, 0.75),     # 概率权重参数
        }
        filtered_bounds = {}
        for param, bound in bounds.items():
            control = self.parameter_controls.get(param, {})
            if control.get('tunable', True):
                custom_bounds = control.get('bounds')
                if custom_bounds and len(custom_bounds) == 2:
                    filtered_bounds[param] = (custom_bounds[0], custom_bounds[1])
                else:
                    filtered_bounds[param] = bound
        return filtered_bounds

    def _map_params_to_config(self, params: Dict[str, float], config: Dict) -> Dict:
        """
        将校准参数映射到配置文件的相应位置
        
        参数:
        params: 校准参数字典
        config: 配置字典
        
        返回:
        updated_config: 更新后的配置字典
        """
        import copy
        updated_config = copy.deepcopy(config)
        
        # 参数映射：校准参数名 -> 配置路径
        param_mapping = {
            # 企业效率参数
            'family_farm_efficiency_mean': ['agents', 'enterprise', 'types', 'FAMILY_FARM_LARGE', 'efficiency_factor_dist', 'mean'],
            'agri_corp_efficiency_mean': ['agents', 'enterprise', 'types', 'AGRI_CORPORATION', 'efficiency_factor_dist', 'mean'],
            'cooperative_efficiency_mean': ['agents', 'enterprise', 'types', 'COOPERATIVE', 'efficiency_factor_dist', 'mean'],
            
            # 企业参与概率（影响土地流转率）
            'family_farm_expansion_probability': ['agents', 'enterprise', 'expansion_decision', 'base_participation_probability', 'FAMILY_FARM_LARGE'],
            'agri_corp_expansion_probability': ['agents', 'enterprise', 'expansion_decision', 'base_participation_probability', 'AGRI_CORPORATION'],
            'cooperative_expansion_probability': ['agents', 'enterprise', 'expansion_decision', 'base_participation_probability', 'COOPERATIVE'],
            
            # 市场参数（影响土地流转）
            'base_rental_price': ['environment', 'market', 'base_rental_price'],
            'quality_multiplier': ['environment', 'market', 'quality_multiplier'],
            
            # 劳动力分配参数（影响农业收入占比）
            'wage_elasticity': ['agents', 'farmer', 'labor_allocation', 'elasticity', 'wage_elasticity'],
            'rental_price_elasticity': ['agents', 'farmer', 'labor_allocation', 'elasticity', 'rental_price_elasticity'],
            'risk_aversion_weight': ['agents', 'farmer', 'labor_allocation', 'elasticity', 'risk_aversion_weight'],
            
            # 生产函数弹性参数（修复：正确路径为 production.production_elasticities）
            'production_alpha': ['production', 'production_elasticities', 'alpha'],
            'production_beta': ['production', 'production_elasticities', 'beta'],
            'production_gamma': ['production', 'production_elasticities', 'gamma'],
            
            # 气候冲击参数（修复：正确路径为 production.climate）
            'climate_mean': ['production', 'climate', 'mean'],
            'climate_std': ['production', 'climate', 'volatility'],
            
            # 价格参数
            'grain_price_volatility': ['environment', 'market', 'grain_price_volatility'],
            'cash_crop_price_volatility': ['environment', 'market', 'cash_crop_price_volatility'],
            
            # 前景理论参数（修复：正确路径为分布参数的mean）
            'loss_aversion_lambda_mean': ['agents', 'farmer', 'prospect_theory', 'loss_aversion_lambda_dist', 'mean'],
            'value_function_curvature_mean': ['agents', 'farmer', 'prospect_theory', 'value_function_curvature_dist', 'mean'],
            'probability_weighting_mean': ['agents', 'farmer', 'prospect_theory', 'probability_weighting_dist', 'mean'],
        }
        
        # 应用参数映射
        for param_name, param_value in params.items():
            if param_name in param_mapping:
                path = param_mapping[param_name]
                # 导航到配置路径并更新值
                current = updated_config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = param_value
                self.logger.debug(f"更新参数 {param_name} = {param_value} -> config{path}")
        
        return updated_config
    
    def calculate_simulated_moments(self, params: Dict[str, float], 
                                  model_runs: int = 50,
                                  n_workers: Optional[int] = None) -> Dict[str, float]:
        """
        计算模拟统计量
        
        参数:
        params: 当前参数组合（会更新到配置中）
        model_runs: 模型运行次数
        
        返回:
        simulated_moments: 模拟得到的统计量
        """
        from analysis.batch_simulation import run_batch_simulation, aggregate_results
        import tempfile
        import os
        
        # 加载基础配置
        config_path = "config/master_config.yaml"
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).resolve().parents[1] / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 应用参数更新到配置
        updated_config = self._map_params_to_config(params, base_config)
        
        # 转换numpy类型为Python原生类型（修复YAML序列化问题）
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        updated_config = convert_numpy_types(updated_config)
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp_file:
            yaml.dump(updated_config, tmp_file, allow_unicode=True, default_flow_style=False, sort_keys=False)
            tmp_config_path = tmp_file.name
        
        try:
            # 运行批量模拟（使用临时配置文件，支持并行）
            # @fix 2024-11-20: 传递n_workers参数，支持并行计算
            results = run_batch_simulation(
                config_path=tmp_config_path,
                n_runs=model_runs,
                n_steps=30,  # 可以配置化
                seed=None,  # 使用不同种子
                n_workers=n_workers  # 并行工作进程数
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_config_path)
            except:
                pass
        
        # 聚合结果
        aggregated = aggregate_results(results)
        
        def zero_summary():
            return {'mean': 0.0, 'std': 0.0, 'ci95': 0.0, 'n': 0}

        # 转换为目标格式
        simulated_moments = {}
        
        # 企业类型分布
        ent_dist = aggregated.get('enterprise_type_distribution', {})
        if ent_dist:
            expected_types = self.target_moments.get('enterprise_type_distribution', {}).keys()
            dist_payload = {}
            if expected_types:
                for ent_type in expected_types:
                    dist_payload[ent_type] = ent_dist.get(ent_type, zero_summary())
            else:
                dist_payload = ent_dist
            simulated_moments['enterprise_type_distribution'] = dist_payload
        
        # 作物结构
        crop_struct = aggregated.get('crop_structure', {})
        if crop_struct:
            simulated_moments['crop_structure'] = crop_struct
        
        # 土地流转
        land_transfer = aggregated.get('land_transfer', {})
        if land_transfer:
            simulated_moments['land_transfer'] = land_transfer
        
        # 收入分布
        income_dist = aggregated.get('income_distribution', {})
        if income_dist:
            simulated_moments['income_distribution'] = income_dist
        
        # 生产率指标
        productivity = aggregated.get('productivity_indicators', {})
        if productivity:
            simulated_moments['productivity_indicators'] = productivity
        
        # 农户异质性指标（新增）
        farmer_hetero = aggregated.get('farmer_heterogeneity', {})
        if farmer_hetero:
            simulated_moments['farmer_heterogeneity'] = farmer_hetero
        
        return simulated_moments

    def calculate_objective_function(self, params_vector: np.ndarray, model_runs: int = None) -> float:
        """
        计算目标函数值
        
        目标函数 = Σ w_i * (simulated_moment_i - target_moment_i)^2
        
        参数:
        params_vector: 参数向量
        model_runs: 模型运行次数（如果为None，使用默认值）
        
        返回:
        objective_value: 目标函数值（越小越好）
        """
        # 将参数向量转换为参数字典
        params = self._vector_to_params(params_vector)
        
        # 计算模拟统计量（使用较少的运行次数以加快速度）
        runs = model_runs if model_runs is not None else 20
        simulated_moments = self.calculate_simulated_moments(
            params, 
            model_runs=runs
        )
        
        # 计算距离
        total_distance = 0.0
        
        for category, weight in self.weight_matrix.items():
            if category in simulated_moments and category in self.target_moments:
                category_distance = self._calculate_category_distance(
                    simulated_moments[category],
                    self.target_moments[category]
                )
                total_distance += weight * category_distance
        
        self.logger.info(f"参数: {params}")
        self.logger.info(f"目标函数值: {total_distance}")
        
        return total_distance

    def _vector_to_params(self, params_vector: np.ndarray) -> Dict[str, float]:
        """将参数向量转换为参数字典"""
        param_names = list(self.param_bounds.keys())
        return {name: value for name, value in zip(param_names, params_vector)}

    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """将参数字典转换为参数向量"""
        param_names = list(self.param_bounds.keys())
        return np.array([params.get(name, 0.0) for name in param_names])

    def _calculate_category_distance(self, simulated: Dict[str, Any], 
                                   target: Dict[str, Any]) -> float:
        """计算某一类别统计量的距离（支持嵌套结构）"""
        distance = 0.0
        for key in target.keys():
            if key in simulated:
                distance += self._calculate_value_distance(simulated[key], target[key])
        return distance

    def _calculate_value_distance(self, simulated_value: Any, target_value: Any) -> float:
        """
        递归计算单个指标的距离，支持嵌套字典。
        """
        # 处理“summary”型字典（包含mean/std/ci95）
        if isinstance(simulated_value, dict) and isinstance(target_value, (int, float)):
            if 'mean' in simulated_value:
                mean_val = simulated_value.get('mean', 0)
                std_val = simulated_value.get('std', 0)
                ci_val = simulated_value.get('ci95', 0)

                if isinstance(mean_val, dict):
                    mean_val = mean_val.get('mean', 0)
                if isinstance(std_val, dict):
                    std_val = std_val.get('mean', 0)
                if isinstance(ci_val, dict):
                    ci_val = ci_val.get('mean', 0)

                scale = std_val if std_val and std_val > 1e-8 else (ci_val / 1.96 if ci_val else 0)
                if not scale or scale <= 1e-8:
                    scale = max(abs(target_value) * 0.05, 1.0)

                try:
                    return ((float(mean_val) - target_value) / scale) ** 2
                except (TypeError, ValueError):
                    return 0.0

        if isinstance(target_value, dict) and isinstance(simulated_value, dict):
            dist = 0.0
            for subkey in target_value.keys():
                if subkey in simulated_value:
                    dist += self._calculate_value_distance(simulated_value[subkey], target_value[subkey])
            return dist

        try:
            if target_value == 0:
                return 0.0
            relative_error = (simulated_value - target_value) / target_value
            return relative_error ** 2
        except TypeError:
            # 存在类型不匹配或非数值情况时忽略该项
            return 0.0

    def calibrate_parameters(self, initial_guess: Dict[str, float] = None,
                           method: str = 'differential_evolution',
                           max_iterations: int = 1000,
                           model_runs_per_iteration: int = 20) -> Dict[str, Any]:
        """
        执行参数校准
        
        参数:
        initial_guess: 初始参数猜测
        method: 优化方法
        max_iterations: 最大迭代次数
        
        返回:
        calibration_result: 校准结果
        """
        self.logger.info("开始参数校准...")
        
        # 设置初始猜测
        if initial_guess is None:
            initial_guess = {}
            for param, bounds in self.param_bounds.items():
                initial_guess[param] = (bounds[0] + bounds[1]) / 2
        
        # 转换为向量形式
        x0 = self._params_to_vector(initial_guess)
        
        # 设置边界
        bounds = [self.param_bounds[param] for param in self.param_bounds.keys()]
        
        # 执行优化
        try:
            # 创建一个包装函数，传递model_runs参数
            def objective_wrapper(x):
                return self.calculate_objective_function(x, model_runs=model_runs_per_iteration)
            
            # @fix 2024-11-20: 替换为差分进化算法（Differential Evolution）
            # 专家审阅意见：ABM模型具有高度非线性、非凸和随机噪声特征
            # 梯度下降类算法（L-BFGS-B）在ABM中几乎肯定会失败
            # Nelder-Mead容易陷入局部最优解
            # 差分进化算法适合处理非凸、非平滑的优化问题
            if method == 'differential_evolution' or method == 'DE':
                # 使用差分进化算法（推荐用于ABM校准）
                from scipy.optimize import differential_evolution
                result = differential_evolution(
                    objective_wrapper,
                    bounds=bounds,
                    maxiter=max_iterations,
                    popsize=15,  # 种群大小（建议为参数数量的5-10倍）
                    mutation=(0.5, 1),  # 变异系数范围
                    recombination=0.7,  # 重组概率
                    seed=self.config.get('experiment', {}).get('seed', 42) if hasattr(self, 'config') else 42,
                    workers=1,  # 并行计算（如果支持）
                    updating='immediate',  # 立即更新策略
                    atol=1e-6,  # 绝对容差
                    tol=1e-6,  # 相对容差
                    polish=True,  # 最后使用局部优化器抛光
                    disp=True
                )
            elif method == 'L-BFGS-B':
                # 保留L-BFGS-B作为选项（但不推荐用于ABM）
                self.logger.warning("L-BFGS-B不推荐用于ABM校准，建议使用differential_evolution")
                result = opt.minimize(
                    objective_wrapper,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations, 'disp': True}
                )
            else:
                # 对于Nelder-Mead等方法，添加收敛检查（不推荐）
                self.logger.warning(f"{method}不推荐用于ABM校准，建议使用differential_evolution")
                result = opt.minimize(
                    objective_wrapper,
                    x0,
                    method=method,
                    options={
                        'maxiter': max_iterations, 
                        'disp': True,
                        'xatol': 1e-6,  # 参数收敛容差
                        'fatol': 1e-6   # 目标函数收敛容差
                    }
                )
            
            # 转换结果
            calibrated_params = self._vector_to_params(result.x)
            
            # 计算最终的模拟统计量（使用更多运行次数以获得更准确的结果）
            final_simulated_moments = self.calculate_simulated_moments(
                calibrated_params,
                model_runs=50  # 最终结果使用更多运行次数
            )
            
            calibration_result = {
                'success': result.success,
                'message': result.message,
                'objective_value': result.fun,
                'calibrated_parameters': calibrated_params,
                'simulated_moments': final_simulated_moments,
                'target_moments': self.target_moments,
                'iterations': result.nit if hasattr(result, 'nit') else None
            }
            
            self.logger.info(f"校准完成。目标函数值: {result.fun}")
            return calibration_result
            
        except Exception as e:
            self.logger.error(f"校准过程中出现错误: {str(e)}")
            return {'success': False, 'error': str(e)}

    def save_calibrated_parameters(self, calibrated_params: Dict[str, Any],
                                   output_path: str = "config/calibrated_params_enhanced.json"):
        """保存校准后的参数"""
        try:
            # 合并基础参数和校准参数
            enhanced_params = self.base_params.copy()
            enhanced_params.setdefault('empirical_parameter_estimates', {})
            enhanced_params['empirical_parameter_estimates'].update(calibrated_params)
            enhanced_params['calibration_metadata'] = {
                'calibration_date': pd.Timestamp.now().isoformat(),
                'has_empirical_moments': bool(self.empirical_moments),
                'target_moments': self.target_moments,
                'calibration_method': 'Simulated Method of Moments'
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_params, f, ensure_ascii=False, indent=2)

            self.logger.info(f"校准参数已保存到: {output_path}")

        except Exception as e:
            self.logger.error(f"保存参数时出现错误: {str(e)}")

    def generate_calibration_report(self, calibration_result: Dict[str, Any],
                                  output_path: str = "results/calibration_report.json"):
        """生成校准报告"""
        try:
            report = {
                'calibration_summary': {
                    'success': calibration_result.get('success', False),
                    'objective_value': calibration_result.get('objective_value', None),
                    'iterations': calibration_result.get('iterations', None)
                },
                'parameter_comparison': {
                    'calibrated_parameters': calibration_result.get('calibrated_parameters', {}),
                    'parameter_bounds': self.param_bounds
                },
                'moment_comparison': {
                    'target_moments': calibration_result.get('target_moments', {}),
                    'simulated_moments': calibration_result.get('simulated_moments', {}),
                    'absolute_errors': self._calculate_absolute_errors(
                        calibration_result.get('simulated_moments', {}),
                        calibration_result.get('target_moments', {})
                    )
                },
                'calibration_metadata': {
                    'calibration_timestamp': pd.Timestamp.now().isoformat(),
                    'weight_matrix': self.weight_matrix,
                    'empirical_moments': self.empirical_moments
                }
            }
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"校准报告已生成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成校准报告时出现错误: {str(e)}")

    def _calculate_absolute_errors(self, simulated: Dict, target: Dict) -> Dict:
        """计算绝对误差"""
        errors = {}
        for category in target.keys():
            if category in simulated:
                errors[category] = {}
                for key in target[category].keys():
                    if key in simulated[category]:
                        error = abs(simulated[category][key] - target[category][key])
                        relative_error = error / target[category][key] * 100
                        errors[category][key] = {
                            'absolute_error': error,
                            'relative_error_percent': relative_error
                        }
        return errors


class CrossValidationManager:
    """
    交叉验证管理器
    
    用于验证校准参数的稳定性和模型的泛化能力
    """
    
    def __init__(self, calibration_manager: CalibrationManager):
        self.calibration_manager = calibration_manager
        self.logger = logging.getLogger('CrossValidationManager')

    def k_fold_validation(self, k: int = 5, data_splits: List = None) -> Dict[str, Any]:
        """
        执行k折交叉验证
        
        参数:
        k: 折数
        data_splits: 数据划分（如果有时间序列数据）
        
        返回:
        validation_results: 验证结果
        """
        validation_results = {
            'fold_results': [],
            'average_objective_value': 0.0,
            'std_objective_value': 0.0,
            'parameter_stability': {}
        }
        
        objective_values = []
        all_calibrated_params = []
        
        for fold in range(k):
            self.logger.info(f"执行第 {fold + 1}/{k} 折验证...")
            
            # 这里需要根据实际数据进行划分
            # 由于我们主要关注参数稳定性，可以通过不同的随机种子来模拟
            
            # 执行校准
            fold_result = self.calibration_manager.calibrate_parameters()
            
            if fold_result['success']:
                objective_values.append(fold_result['objective_value'])
                all_calibrated_params.append(fold_result['calibrated_parameters'])
                validation_results['fold_results'].append(fold_result)
        
        # 计算统计量
        if objective_values:
            validation_results['average_objective_value'] = np.mean(objective_values)
            validation_results['std_objective_value'] = np.std(objective_values)
            
            # 计算参数稳定性
            validation_results['parameter_stability'] = self._calculate_parameter_stability(
                all_calibrated_params
            )
        
        return validation_results

    def _calculate_parameter_stability(self, all_params: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """计算参数稳定性统计量"""
        stability = {}
        
        if not all_params:
            return stability
        
        # 获取所有参数名
        param_names = all_params[0].keys()
        
        for param_name in param_names:
            param_values = [params[param_name] for params in all_params]
            stability[param_name] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'cv': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 0,
                'min': np.min(param_values),
                'max': np.max(param_values)
            }
        
        return stability 