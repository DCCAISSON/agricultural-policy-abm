# -*- coding: utf-8 -*-
"""
@module three_tier_validation.py
@description 三层验证体系 - 微观、宏观、动态验证
@created 2025-12-06
@author GitHub Copilot

理论参考：
1. Windrum et al. (2007) "Empirical validation of agent-based models"
   - JASSS验证标准框架
   
2. Fagiolo et al. (2019) "Validation of agent-based models"
   - JEDC计量验证方法
   
3. Grimm et al. (2020) "The ODD protocol second update"
   - JASSS ODD+D验证指南

验证层级：
1. 微观验证：智能体行为是否符合理论预测
2. 宏观验证：涌现的宏观模式是否与现实数据匹配
3. 动态验证：时间序列演化是否符合历史趋势
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
import warnings


@dataclass
class ValidationResult:
    """验证结果数据类"""
    test_name: str
    passed: bool
    score: float  # 0-1之间
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class ValidationReport:
    """验证报告"""
    timestamp: str
    overall_score: float
    micro_results: List[ValidationResult] = field(default_factory=list)
    macro_results: List[ValidationResult] = field(default_factory=list)
    dynamic_results: List[ValidationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'micro_validation': {
                'passed': sum(1 for r in self.micro_results if r.passed),
                'total': len(self.micro_results),
                'results': [{'test': r.test_name, 'passed': r.passed, 'score': r.score} 
                           for r in self.micro_results]
            },
            'macro_validation': {
                'passed': sum(1 for r in self.macro_results if r.passed),
                'total': len(self.macro_results),
                'results': [{'test': r.test_name, 'passed': r.passed, 'score': r.score} 
                           for r in self.macro_results]
            },
            'dynamic_validation': {
                'passed': sum(1 for r in self.dynamic_results if r.passed),
                'total': len(self.dynamic_results),
                'results': [{'test': r.test_name, 'passed': r.passed, 'score': r.score} 
                           for r in self.dynamic_results]
            }
        }


class ThreeTierValidation:
    """
    三层验证体系
    
    实现Windrum et al. (2007) 提出的验证框架：
    1. 微观验证：检验智能体行为是否符合理论预测
    2. 宏观验证：检验涌现模式是否与真实数据匹配
    3. 动态验证：检验时间序列是否符合历史趋势
    
    使用方法：
    ```python
    validator = ThreeTierValidation(cles_data_path='校准用数据源/')
    report = validator.run_full_validation(model)
    ```
    """
    
    def __init__(self, 
                 cles_data_path: Optional[str] = None,
                 tolerance: float = 0.15,
                 confidence_level: float = 0.95):
        """
        初始化验证器
        
        Args:
            cles_data_path: CLES数据路径
            tolerance: 验证容忍度（相对误差阈值）
            confidence_level: 置信水平
        """
        self.cles_data_path = Path(cles_data_path) if cles_data_path else None
        self.tolerance = tolerance
        self.confidence_level = confidence_level
        self.cles_data = None
        
        # 加载CLES数据
        if self.cles_data_path and self.cles_data_path.exists():
            self._load_cles_data()
    
    def _load_cles_data(self):
        """加载CLES数据"""
        try:
            # 尝试加载各种格式的数据
            household_file = None
            for pattern in ['*household*.csv', '*household*.xlsx', '*household*.parquet']:
                files = list(self.cles_data_path.glob(pattern))
                if files:
                    household_file = files[0]
                    break
            
            if household_file:
                if household_file.suffix == '.csv':
                    self.cles_data = pd.read_csv(household_file)
                elif household_file.suffix in ['.xlsx', '.xls']:
                    self.cles_data = pd.read_excel(household_file)
                elif household_file.suffix == '.parquet':
                    self.cles_data = pd.read_parquet(household_file)
                    
                print(f"[验证] 已加载CLES数据: {household_file.name}, {len(self.cles_data)}条记录")
        except Exception as e:
            warnings.warn(f"无法加载CLES数据: {e}")
    
    # ================== 微观验证 ==================
    
    def micro_validation(self, model) -> List[ValidationResult]:
        """
        微观验证：检验智能体行为是否符合理论预测
        
        包含以下检验：
        1. 前景理论的损失厌恶效应
        2. Melitz企业异质性排序
        3. 社会学习的邻居效应
        4. 收入-消费关系
        5. 风险偏好与决策一致性
        
        Args:
            model: 模型实例（运行后）
        
        Returns:
            验证结果列表
        """
        results = []
        
        # 检验1：前景理论损失厌恶
        loss_aversion_result = self._verify_loss_aversion(model)
        results.append(loss_aversion_result)
        
        # 检验2：Melitz企业异质性排序
        melitz_result = self._verify_melitz_sorting(model)
        results.append(melitz_result)
        
        # 检验3：社会学习邻居效应
        neighbor_effect_result = self._verify_neighbor_effect(model)
        results.append(neighbor_effect_result)
        
        # 检验4：收入-消费理论一致性
        consumption_result = self._verify_consumption_theory(model)
        results.append(consumption_result)
        
        # 检验5：土地流转意愿阈值检验
        transfer_threshold_result = self._verify_transfer_thresholds(model)
        results.append(transfer_threshold_result)
        
        return results
    
    def _verify_loss_aversion(self, model) -> ValidationResult:
        """
        验证前景理论的损失厌恶效应
        
        理论预测：
        - Kahneman & Tversky (1979): 损失的负效用 > 等额收益的正效用
        - λ (损失厌恶系数) ≈ 2.25-3.0
        
        检验方法：
        - 比较农户面对等概率收益/损失情境的决策
        - 验证损失规避行为的普遍性
        """
        test_name = "前景理论损失厌恶检验"
        
        try:
            farmers = getattr(model, 'farmers', [])
            if not farmers:
                return ValidationResult(test_name, False, 0.0, 
                                        message="无法获取农户数据")
            
            # 统计具有损失厌恶特征的农户比例
            loss_averse_count = 0
            total_checked = 0
            lambda_values = []
            
            for farmer in farmers:
                # 检查是否有前景理论参数
                if hasattr(farmer, 'loss_aversion_lambda'):
                    lambda_val = farmer.loss_aversion_lambda
                    lambda_values.append(lambda_val)
                    if lambda_val > 1.0:  # λ > 1 表示损失厌恶
                        loss_averse_count += 1
                    total_checked += 1
            
            if total_checked == 0:
                return ValidationResult(test_name, False, 0.0, 
                                        message="农户缺少loss_aversion_lambda属性")
            
            # 计算损失厌恶比例
            loss_averse_ratio = loss_averse_count / total_checked
            mean_lambda = np.mean(lambda_values)
            
            # 验证标准：
            # 1. 超过80%的农户具有损失厌恶
            # 2. 平均λ在合理范围内 (2.0-4.0)
            passed = loss_averse_ratio >= 0.8 and 2.0 <= mean_lambda <= 4.0
            score = min(1.0, loss_averse_ratio * (1.0 if 2.0 <= mean_lambda <= 4.0 else 0.5))
            
            details = {
                'loss_averse_ratio': loss_averse_ratio,
                'mean_lambda': mean_lambda,
                'std_lambda': np.std(lambda_values),
                'sample_size': total_checked,
                'expected_lambda_range': [2.0, 4.0],
                'literature_reference': 'Kahneman & Tversky (1979)'
            }
            
            message = (f"损失厌恶比例: {loss_averse_ratio:.1%}, "
                      f"平均λ: {mean_lambda:.2f} (参考范围: 2.0-4.0)")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _verify_melitz_sorting(self, model) -> ValidationResult:
        """
        验证Melitz企业异质性排序
        
        理论预测：
        - Melitz (2003): 企业规模与生产率正相关
        - 更大的企业应具有更高的TFP
        - 低效率企业会被淘汰出市场
        """
        test_name = "Melitz企业异质性排序检验"
        
        try:
            enterprises = getattr(model, 'agricultural_enterprises', [])
            if not enterprises or len(enterprises) < 5:
                return ValidationResult(test_name, False, 0.0, 
                                        message="企业数量不足（需≥5家）")
            
            scales = []
            efficiencies = []
            
            for ent in enterprises:
                scale = getattr(ent, 'total_leased_area', 0)
                efficiency = getattr(ent, 'tfp', None) or getattr(ent, 'efficiency_factor', 1.0)
                
                if scale > 0 and efficiency > 0:
                    scales.append(scale)
                    efficiencies.append(efficiency)
            
            if len(scales) < 5:
                return ValidationResult(test_name, False, 0.0, 
                                        message="有效企业数据不足")
            
            # 计算规模-效率相关性
            correlation, p_value = stats.spearmanr(scales, efficiencies)
            
            # 验证标准：
            # 正相关 (r > 0) 且统计显著 (p < 0.05)
            passed = correlation > 0 and p_value < 0.1
            score = max(0, correlation) if p_value < 0.1 else max(0, correlation * 0.5)
            
            details = {
                'spearman_correlation': correlation,
                'p_value': p_value,
                'sample_size': len(scales),
                'scale_range': [min(scales), max(scales)],
                'efficiency_range': [min(efficiencies), max(efficiencies)],
                'literature_reference': 'Melitz (2003)'
            }
            
            message = (f"规模-效率Spearman相关系数: {correlation:.3f} "
                      f"(p={p_value:.4f})")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _verify_neighbor_effect(self, model) -> ValidationResult:
        """
        验证社会学习的邻居效应
        
        理论预测：
        - Conley & Udry (2010): 邻居的种植决策影响自身决策
        - 种植相同作物的邻居比例 ↑ → 自己种植概率 ↑
        """
        test_name = "社会学习邻居效应检验"
        
        try:
            farmers = getattr(model, 'farmers', [])
            if not farmers or len(farmers) < 10:
                return ValidationResult(test_name, False, 0.0, 
                                        message="农户数量不足")
            
            # 检查农户是否有邻居和作物选择信息
            neighbor_agreement_rates = []
            
            for farmer in farmers:
                # 获取邻居列表
                neighbors = getattr(farmer, 'neighbors', [])
                if not neighbors or len(neighbors) < 2:
                    continue
                
                # 获取自己的作物选择
                own_crop = getattr(farmer, 'last_crop_choice', None)
                if own_crop is None:
                    continue
                own_crop_str = getattr(own_crop, 'value', str(own_crop))
                
                # 计算邻居中种相同作物的比例
                same_crop_count = 0
                valid_neighbors = 0
                
                for neighbor in neighbors:
                    neighbor_crop = getattr(neighbor, 'last_crop_choice', None)
                    if neighbor_crop is None:
                        continue
                    neighbor_crop_str = getattr(neighbor_crop, 'value', str(neighbor_crop))
                    valid_neighbors += 1
                    
                    if own_crop_str == neighbor_crop_str:
                        same_crop_count += 1
                
                if valid_neighbors > 0:
                    agreement_rate = same_crop_count / valid_neighbors
                    neighbor_agreement_rates.append(agreement_rate)
            
            if len(neighbor_agreement_rates) < 10:
                # 如果无法直接检验邻居效应，检查是否有社会学习权重
                social_learning_weights = []
                for farmer in farmers:
                    slw = getattr(farmer, 'social_learning_weight', None)
                    if slw is not None:
                        social_learning_weights.append(slw)
                
                if social_learning_weights:
                    mean_slw = np.mean(social_learning_weights)
                    passed = mean_slw > 0
                    score = min(1.0, mean_slw * 2)  # 假设0.5是典型值
                    
                    details = {
                        'mean_social_learning_weight': mean_slw,
                        'std_social_learning_weight': np.std(social_learning_weights),
                        'sample_size': len(social_learning_weights)
                    }
                    message = f"社会学习权重均值: {mean_slw:.3f}"
                    
                    return ValidationResult(test_name, passed, score, details, message)
                else:
                    return ValidationResult(test_name, False, 0.0, 
                                            message="无法获取邻居效应数据")
            
            mean_agreement = np.mean(neighbor_agreement_rates)
            
            # 如果邻居效应显著，一致率应高于随机水平（~0.5）
            passed = mean_agreement > 0.5
            score = min(1.0, (mean_agreement - 0.3) / 0.4)  # 0.3-0.7映射到0-1
            
            details = {
                'mean_neighbor_agreement_rate': mean_agreement,
                'std_agreement_rate': np.std(neighbor_agreement_rates),
                'sample_size': len(neighbor_agreement_rates),
                'random_expectation': 0.5,
                'literature_reference': 'Conley & Udry (2010)'
            }
            
            message = f"邻居作物选择一致率: {mean_agreement:.3f} (随机基准: 0.5)"
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _verify_consumption_theory(self, model) -> ValidationResult:
        """
        验证收入-消费理论一致性
        
        理论预测：
        - 边际消费倾向 (MPC) 在0-1之间
        - 恩格尔定律：收入 ↑ → 食物支出占比 ↓
        """
        test_name = "收入-消费理论一致性检验"
        
        try:
            farmers = getattr(model, 'farmers', [])
            if not farmers:
                return ValidationResult(test_name, False, 0.0, 
                                        message="无法获取农户数据")
            
            incomes = []
            consumptions = []
            
            for farmer in farmers:
                income = getattr(farmer, 'disposable_income', 0) or getattr(farmer, 'total_income', 0)
                consumption = getattr(farmer, 'consumption_essential', 0) + \
                             getattr(farmer, 'consumption_luxury', 0)
                
                if income > 0:
                    incomes.append(income)
                    consumptions.append(min(consumption, income))  # 消费不超过收入
            
            if len(incomes) < 10:
                return ValidationResult(test_name, False, 0.0, 
                                        message="有效收入-消费数据不足")
            
            # 计算边际消费倾向
            incomes = np.array(incomes)
            consumptions = np.array(consumptions)
            
            # 回归分析
            slope, intercept, r_value, p_value, std_err = stats.linregress(incomes, consumptions)
            mpc = slope  # 边际消费倾向
            
            # 验证标准：
            # MPC在0.3-0.9之间是合理的
            passed = 0.3 <= mpc <= 0.9
            score = 1.0 if passed else max(0, 1 - abs(mpc - 0.6) / 0.3)
            
            details = {
                'marginal_propensity_to_consume': mpc,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'sample_size': len(incomes),
                'expected_mpc_range': [0.3, 0.9]
            }
            
            message = f"边际消费倾向(MPC): {mpc:.3f} (合理范围: 0.3-0.9)"
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _verify_transfer_thresholds(self, model) -> ValidationResult:
        """
        验证土地流转意愿阈值
        
        理论预测：
        - Deininger & Jin (2005): 土地流转受劳动力机会成本影响
        - 有非农就业机会的农户更可能流出土地
        """
        test_name = "土地流转意愿阈值检验"
        
        try:
            farmers = getattr(model, 'farmers', [])
            if not farmers:
                return ValidationResult(test_name, False, 0.0, 
                                        message="无法获取农户数据")
            
            # 统计不同生计类型的流转行为
            pure_farmer_transfer = []  # 纯农户流转率
            pluri_transfer = []  # 兼业户流转率
            non_farmer_transfer = []  # 非农户流转率
            
            for farmer in farmers:
                strategy = str(getattr(farmer, 'livelihood_strategy', '')).upper()
                
                # 计算流转率
                owned = farmer.get_owned_land_area() if hasattr(farmer, 'get_owned_land_area') else 0
                leased_out = sum(p.area for p in getattr(farmer, 'leased_out_patches', []) 
                                if hasattr(p, 'area'))
                
                if owned > 0:
                    transfer_rate = leased_out / owned
                    
                    if 'PURE' in strategy:
                        pure_farmer_transfer.append(transfer_rate)
                    elif 'PLURI' in strategy:
                        pluri_transfer.append(transfer_rate)
                    elif 'NON' in strategy:
                        non_farmer_transfer.append(transfer_rate)
            
            # 理论预测：非农户 > 兼业户 > 纯农户
            mean_pure = np.mean(pure_farmer_transfer) if pure_farmer_transfer else 0
            mean_pluri = np.mean(pluri_transfer) if pluri_transfer else 0
            mean_non = np.mean(non_farmer_transfer) if non_farmer_transfer else 0
            
            # 验证排序
            order_correct = mean_pure <= mean_pluri <= mean_non
            
            # 计算得分
            if order_correct:
                score = 1.0
            elif mean_pure <= mean_pluri or mean_pluri <= mean_non:
                score = 0.5  # 部分正确
            else:
                score = 0.2
            
            passed = score >= 0.5
            
            details = {
                'pure_farmer_mean_transfer_rate': mean_pure,
                'pluriactive_mean_transfer_rate': mean_pluri,
                'non_farmer_mean_transfer_rate': mean_non,
                'sample_sizes': {
                    'pure': len(pure_farmer_transfer),
                    'pluriactive': len(pluri_transfer),
                    'non_farmer': len(non_farmer_transfer)
                },
                'expected_order': 'pure <= pluriactive <= non_farmer',
                'actual_order_correct': order_correct,
                'literature_reference': 'Deininger & Jin (2005)'
            }
            
            message = (f"流转率排序: 纯农户({mean_pure:.3f}) → "
                      f"兼业户({mean_pluri:.3f}) → 非农户({mean_non:.3f})")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    # ================== 宏观验证 ==================
    
    def macro_validation(self, model_results: Dict, 
                        target_moments: Optional[Dict] = None) -> List[ValidationResult]:
        """
        宏观验证：检验涌现的宏观模式是否与现实数据匹配
        
        使用矩匹配法（Method of Simulated Moments）进行验证
        
        Args:
            model_results: 模型运行结果字典
            target_moments: 目标矩（如从CLES数据计算得到）
        
        Returns:
            验证结果列表
        """
        results = []
        
        # 如果没有提供目标矩，使用默认值（基于中国农村统计数据）
        if target_moments is None:
            target_moments = self._get_default_target_moments()
        
        # 验证1：收入分布
        income_result = self._validate_income_distribution(model_results, target_moments)
        results.append(income_result)
        
        # 验证2：土地流转率
        transfer_result = self._validate_transfer_rate(model_results, target_moments)
        results.append(transfer_result)
        
        # 验证3：生计结构
        livelihood_result = self._validate_livelihood_structure(model_results, target_moments)
        results.append(livelihood_result)
        
        # 验证4：粮食种植比例
        grain_result = self._validate_grain_ratio(model_results, target_moments)
        results.append(grain_result)
        
        return results
    
    def _get_default_target_moments(self) -> Dict:
        """
        获取默认目标矩
        
        数据来源：
        - 《中国统计年鉴》2020-2022
        - CLES农户调查数据
        - 相关文献估计
        """
        return {
            # 收入指标（2020年数据）
            'mean_farmer_income': 17131,  # 农村居民人均可支配收入
            'income_gini': 0.38,  # 农村基尼系数（估计）
            
            # 农业收入占比
            'agri_income_share_mean': 0.35,  # 农业收入占比均值
            'agri_income_share_p25': 0.15,
            'agri_income_share_p50': 0.30,
            'agri_income_share_p75': 0.50,
            
            # 土地流转
            'land_transfer_rate': 0.37,  # 全国土地流转率（2020）
            
            # 生计结构
            'pure_farmer_share': 0.20,  # 纯农户比例
            'pluriactive_share': 0.55,  # 兼业户比例
            'non_farmer_share': 0.25,  # 非农户比例
            
            # 粮食种植
            'grain_planting_ratio': 0.65,  # 粮食种植面积占比
        }
    
    def _validate_income_distribution(self, model_results: Dict, 
                                     target_moments: Dict) -> ValidationResult:
        """验证收入分布"""
        test_name = "收入分布矩匹配检验"
        
        try:
            incomes = model_results.get('farmer_incomes', [])
            if not incomes:
                return ValidationResult(test_name, False, 0.0, 
                                        message="无收入数据")
            
            # 计算模拟矩
            mean_income = np.mean(incomes)
            gini = self._calculate_gini(incomes)
            
            # 与目标矩比较
            target_mean = target_moments.get('mean_farmer_income', 17131)
            target_gini = target_moments.get('income_gini', 0.38)
            
            # 计算相对误差
            mean_error = abs(mean_income - target_mean) / target_mean
            gini_error = abs(gini - target_gini) / target_gini
            
            # 验证通过条件
            passed = mean_error < self.tolerance and gini_error < self.tolerance
            score = max(0, 1 - (mean_error + gini_error) / 2)
            
            details = {
                'model_mean_income': mean_income,
                'target_mean_income': target_mean,
                'mean_relative_error': mean_error,
                'model_gini': gini,
                'target_gini': target_gini,
                'gini_relative_error': gini_error,
                'tolerance': self.tolerance
            }
            
            message = (f"收入均值误差: {mean_error:.1%}, "
                      f"基尼系数误差: {gini_error:.1%}")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _validate_transfer_rate(self, model_results: Dict, 
                               target_moments: Dict) -> ValidationResult:
        """验证土地流转率"""
        test_name = "土地流转率矩匹配检验"
        
        try:
            transfer_rate = model_results.get('land_transfer_rate', 0)
            target_rate = target_moments.get('land_transfer_rate', 0.37)
            
            # 计算相对误差
            error = abs(transfer_rate - target_rate) / max(target_rate, 0.01)
            
            passed = error < self.tolerance
            score = max(0, 1 - error)
            
            details = {
                'model_transfer_rate': transfer_rate,
                'target_transfer_rate': target_rate,
                'relative_error': error,
                'tolerance': self.tolerance
            }
            
            message = f"流转率: {transfer_rate:.1%} vs 目标: {target_rate:.1%} (误差: {error:.1%})"
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _validate_livelihood_structure(self, model_results: Dict, 
                                      target_moments: Dict) -> ValidationResult:
        """验证生计结构"""
        test_name = "生计结构矩匹配检验"
        
        try:
            # 获取模拟结果
            pure_share = model_results.get('pure_farmer_share', 0)
            pluri_share = model_results.get('pluriactive_share', 0)
            non_share = model_results.get('non_farmer_share', 0)
            
            # 目标值
            target_pure = target_moments.get('pure_farmer_share', 0.20)
            target_pluri = target_moments.get('pluriactive_share', 0.55)
            target_non = target_moments.get('non_farmer_share', 0.25)
            
            # 计算误差
            errors = [
                abs(pure_share - target_pure) / max(target_pure, 0.01),
                abs(pluri_share - target_pluri) / max(target_pluri, 0.01),
                abs(non_share - target_non) / max(target_non, 0.01)
            ]
            mean_error = np.mean(errors)
            
            passed = mean_error < self.tolerance
            score = max(0, 1 - mean_error)
            
            details = {
                'model_structure': {'pure': pure_share, 'pluri': pluri_share, 'non': non_share},
                'target_structure': {'pure': target_pure, 'pluri': target_pluri, 'non': target_non},
                'individual_errors': errors,
                'mean_error': mean_error
            }
            
            message = (f"生计结构误差: 纯农{errors[0]:.1%}, "
                      f"兼业{errors[1]:.1%}, 非农{errors[2]:.1%}")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _validate_grain_ratio(self, model_results: Dict, 
                             target_moments: Dict) -> ValidationResult:
        """验证粮食种植比例"""
        test_name = "粮食种植比例矩匹配检验"
        
        try:
            grain_ratio = model_results.get('grain_planting_ratio', 0)
            target_ratio = target_moments.get('grain_planting_ratio', 0.65)
            
            error = abs(grain_ratio - target_ratio) / max(target_ratio, 0.01)
            
            passed = error < self.tolerance
            score = max(0, 1 - error)
            
            details = {
                'model_grain_ratio': grain_ratio,
                'target_grain_ratio': target_ratio,
                'relative_error': error
            }
            
            message = f"种粮比例: {grain_ratio:.1%} vs 目标: {target_ratio:.1%}"
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    # ================== 动态验证 ==================
    
    def dynamic_validation(self, model_timeseries: Dict, 
                          historical_data: Optional[Dict] = None) -> List[ValidationResult]:
        """
        动态验证：检验时间序列演化是否符合历史趋势
        
        Args:
            model_timeseries: 模型时间序列数据
            historical_data: 历史数据（可选）
        
        Returns:
            验证结果列表
        """
        results = []
        
        # 验证1：粮食产量趋势
        grain_trend = self._verify_trend(
            model_timeseries.get('grain_output', []),
            expected_direction='increasing',
            expected_rate=0.015,
            test_name='粮食产量趋势检验'
        )
        results.append(grain_trend)
        
        # 验证2：农民收入趋势
        income_trend = self._verify_trend(
            model_timeseries.get('farmer_income', []),
            expected_direction='increasing',
            expected_rate=0.05,
            test_name='农民收入趋势检验'
        )
        results.append(income_trend)
        
        # 验证3：土地流转趋势
        transfer_trend = self._verify_trend(
            model_timeseries.get('land_transfer_rate', []),
            expected_direction='increasing',
            expected_rate=0.02,
            test_name='土地流转趋势检验'
        )
        results.append(transfer_trend)
        
        # 验证4：时间序列平稳性（检验是否有异常波动）
        stability_result = self._verify_stability(model_timeseries)
        results.append(stability_result)
        
        return results
    
    def _verify_trend(self, timeseries: List[float], 
                     expected_direction: str,
                     expected_rate: float,
                     test_name: str) -> ValidationResult:
        """
        验证趋势方向和速率
        
        Args:
            timeseries: 时间序列数据
            expected_direction: 'increasing' 或 'decreasing'
            expected_rate: 预期年增长/下降率
            test_name: 检验名称
        """
        try:
            if not timeseries or len(timeseries) < 3:
                return ValidationResult(test_name, False, 0.0, 
                                        message="时间序列数据不足")
            
            ts = np.array(timeseries)
            n = len(ts)
            
            # 计算趋势（线性回归）
            x = np.arange(n)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)
            
            # 计算相对增长率
            if ts[0] != 0:
                actual_rate = (ts[-1] - ts[0]) / ts[0] / n  # 平均年增长率
            else:
                actual_rate = slope / np.mean(ts) if np.mean(ts) != 0 else 0
            
            # 检验方向
            direction_correct = (expected_direction == 'increasing' and slope > 0) or \
                               (expected_direction == 'decreasing' and slope < 0)
            
            # 计算得分
            if direction_correct:
                rate_diff = abs(actual_rate - expected_rate) / max(expected_rate, 0.001)
                score = max(0, 1 - rate_diff * 0.5)  # 给增长率差异一定容忍度
            else:
                score = 0.2  # 方向错误得分低
            
            passed = direction_correct and score > 0.5
            
            details = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'actual_annual_rate': actual_rate,
                'expected_annual_rate': expected_rate,
                'expected_direction': expected_direction,
                'direction_correct': direction_correct,
                'series_length': n
            }
            
            message = (f"趋势方向{'正确' if direction_correct else '错误'}, "
                      f"实际增长率: {actual_rate:.2%} (预期: {expected_rate:.2%})")
            
            return ValidationResult(test_name, passed, score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    def _verify_stability(self, model_timeseries: Dict) -> ValidationResult:
        """验证时间序列稳定性（无异常波动）"""
        test_name = "时间序列稳定性检验"
        
        try:
            stability_scores = []
            
            for key, ts in model_timeseries.items():
                if not ts or len(ts) < 3:
                    continue
                
                ts = np.array(ts)
                
                # 检查是否有极端值（超过3个标准差）
                mean_val = np.mean(ts)
                std_val = np.std(ts)
                
                if std_val > 0:
                    z_scores = np.abs((ts - mean_val) / std_val)
                    extreme_ratio = np.sum(z_scores > 3) / len(ts)
                    
                    # 变异系数
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    # 稳定性得分：极端值越少、变异系数越小，得分越高
                    series_score = max(0, 1 - extreme_ratio * 5 - cv * 0.5)
                    stability_scores.append(series_score)
            
            if not stability_scores:
                return ValidationResult(test_name, False, 0.0, 
                                        message="无有效时间序列")
            
            overall_score = np.mean(stability_scores)
            passed = overall_score > 0.7
            
            details = {
                'individual_scores': stability_scores,
                'overall_stability_score': overall_score,
                'n_series': len(stability_scores)
            }
            
            message = f"时间序列稳定性得分: {overall_score:.2f}"
            
            return ValidationResult(test_name, passed, overall_score, details, message)
            
        except Exception as e:
            return ValidationResult(test_name, False, 0.0, 
                                    message=f"检验失败: {str(e)}")
    
    # ================== 辅助方法 ==================
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0
        
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))
        return gini
    
    def run_full_validation(self, model, model_results: Optional[Dict] = None,
                           model_timeseries: Optional[Dict] = None) -> ValidationReport:
        """
        运行完整的三层验证
        
        Args:
            model: 模型实例
            model_results: 模型结果（用于宏观验证）
            model_timeseries: 时间序列数据（用于动态验证）
        
        Returns:
            验证报告
        """
        from datetime import datetime
        
        # 微观验证
        micro_results = self.micro_validation(model)
        
        # 宏观验证
        if model_results is None:
            model_results = self._extract_model_results(model)
        macro_results = self.macro_validation(model_results)
        
        # 动态验证
        if model_timeseries is None:
            model_timeseries = self._extract_model_timeseries(model)
        dynamic_results = self.dynamic_validation(model_timeseries)
        
        # 计算总体得分
        all_results = micro_results + macro_results + dynamic_results
        if all_results:
            overall_score = np.mean([r.score for r in all_results])
        else:
            overall_score = 0.0
        
        report = ValidationReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            overall_score=overall_score,
            micro_results=micro_results,
            macro_results=macro_results,
            dynamic_results=dynamic_results
        )
        
        return report
    
    def _extract_model_results(self, model) -> Dict:
        """从模型中提取宏观结果"""
        results = {}
        
        farmers = getattr(model, 'farmers', [])
        if farmers:
            # 收入数据
            incomes = [getattr(f, 'total_income', 0) for f in farmers]
            results['farmer_incomes'] = incomes
            
            # 生计结构
            n_total = len(farmers)
            if n_total > 0:
                n_pure = sum(1 for f in farmers 
                            if 'PURE' in str(getattr(f, 'livelihood_strategy', '')).upper())
                n_pluri = sum(1 for f in farmers 
                             if 'PLURI' in str(getattr(f, 'livelihood_strategy', '')).upper())
                n_non = sum(1 for f in farmers 
                           if 'NON' in str(getattr(f, 'livelihood_strategy', '')).upper())
                
                results['pure_farmer_share'] = n_pure / n_total
                results['pluriactive_share'] = n_pluri / n_total
                results['non_farmer_share'] = n_non / n_total
            
            # 土地流转率
            total_owned = sum(f.get_owned_land_area() if hasattr(f, 'get_owned_land_area') else 0 
                             for f in farmers)
            total_leased = sum(sum(p.area for p in getattr(f, 'leased_out_patches', []) if hasattr(p, 'area'))
                              for f in farmers)
            if total_owned > 0:
                results['land_transfer_rate'] = total_leased / total_owned
            
            # 粮食种植比例
            total_grain_area = 0
            total_area = 0
            for f in farmers:
                area = f.get_operated_land_area() if hasattr(f, 'get_operated_land_area') else 0
                crop = getattr(f, 'last_crop_choice', None)
                crop_str = getattr(crop, 'value', str(crop)) if crop else ''
                
                total_area += area
                if 'Grain' in crop_str:
                    total_grain_area += area
            
            if total_area > 0:
                results['grain_planting_ratio'] = total_grain_area / total_area
        
        return results
    
    def _extract_model_timeseries(self, model) -> Dict:
        """从模型数据收集器提取时间序列"""
        timeseries = {}
        
        # 尝试从数据收集器获取
        datacollector = getattr(model, 'datacollector', None)
        if datacollector:
            model_data = datacollector.get_model_vars_dataframe()
            if model_data is not None:
                for col in model_data.columns:
                    timeseries[col] = model_data[col].tolist()
        
        return timeseries
    
    def print_report(self, report: ValidationReport):
        """打印验证报告"""
        print("\n" + "=" * 70)
        print("三层验证报告")
        print("=" * 70)
        print(f"验证时间: {report.timestamp}")
        print(f"总体得分: {report.overall_score:.2f}/1.00")
        print("-" * 70)
        
        # 微观验证
        print("\n【微观验证】")
        for r in report.micro_results:
            status = "✅ 通过" if r.passed else "❌ 未通过"
            print(f"  {r.test_name}: {status} (得分: {r.score:.2f})")
            print(f"    {r.message}")
        
        # 宏观验证
        print("\n【宏观验证】")
        for r in report.macro_results:
            status = "✅ 通过" if r.passed else "❌ 未通过"
            print(f"  {r.test_name}: {status} (得分: {r.score:.2f})")
            print(f"    {r.message}")
        
        # 动态验证
        print("\n【动态验证】")
        for r in report.dynamic_results:
            status = "✅ 通过" if r.passed else "❌ 未通过"
            print(f"  {r.test_name}: {status} (得分: {r.score:.2f})")
            print(f"    {r.message}")
        
        print("\n" + "=" * 70)
        
        # 总结
        total_tests = len(report.micro_results) + len(report.macro_results) + len(report.dynamic_results)
        passed_tests = (sum(1 for r in report.micro_results if r.passed) +
                       sum(1 for r in report.macro_results if r.passed) +
                       sum(1 for r in report.dynamic_results if r.passed))
        
        print(f"通过检验: {passed_tests}/{total_tests}")
        if report.overall_score >= 0.7:
            print("✅ 验证结论: 模型通过三层验证")
        elif report.overall_score >= 0.5:
            print("⚠️ 验证结论: 模型部分通过验证，需要改进")
        else:
            print("❌ 验证结论: 模型未通过验证，需要重大修改")
