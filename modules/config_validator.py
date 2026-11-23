# -*- coding: utf-8 -*-
"""
@module config_validator.py
@description 配置验证器 - 强制配置注入，消除硬编码
@created 2025-12-04

核心功能：
1. 强制所有参数从config读取
2. 缺失必需参数时报错
3. 提供参数默认值和类型验证
4. 记录参数使用情况，便于追踪

设计理念：
- 硬编码是ABM模型的"技术债务"
- 所有行为参数都应该可配置
- 配置缺失应该显式报错，而非静默使用默认值
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
import logging
import warnings


# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ConfigRequirement:
    """配置需求定义"""
    path: str                           # 配置路径，如 "agents.farmer.age_dist.mean"
    required: bool = True               # 是否必需
    default: Any = None                 # 默认值（仅当required=False时使用）
    type_hint: Type = None              # 期望类型
    min_value: float = None             # 最小值（数值类型）
    max_value: float = None             # 最大值（数值类型）
    choices: List[Any] = None           # 可选值列表
    description: str = ""               # 参数描述


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class MissingConfigError(ConfigValidationError):
    """配置缺失错误"""
    pass


class InvalidConfigValueError(ConfigValidationError):
    """配置值无效错误"""
    pass


class ConfigValidator:
    """
    配置验证器
    
    使用方式：
    1. 定义配置需求
    2. 验证配置字典
    3. 获取参数值（带验证）
    """
    
    # 全局配置需求注册表
    _requirements: Dict[str, ConfigRequirement] = {}
    
    # 参数使用记录
    _usage_log: Dict[str, int] = {}
    
    def __init__(self, config: Dict[str, Any], strict_mode: bool = True):
        """
        初始化验证器
        
        Args:
            config: 配置字典
            strict_mode: 严格模式，缺失必需参数时报错
        """
        self.config = config
        self.strict_mode = strict_mode
        self._accessed_paths: List[str] = []
        self._missing_paths: List[str] = []
        self._warnings: List[str] = []
    
    @classmethod
    def register_requirement(cls, requirement: ConfigRequirement):
        """注册配置需求"""
        cls._requirements[requirement.path] = requirement
    
    @classmethod
    def register_requirements(cls, requirements: List[ConfigRequirement]):
        """批量注册配置需求"""
        for req in requirements:
            cls.register_requirement(req)
    
    def get_nested(self, path: str, default: Any = None, required: bool = False) -> Any:
        """
        获取嵌套配置值
        
        Args:
            path: 配置路径，如 "agents.farmer.age_dist.mean"
            default: 默认值
            required: 是否必需
            
        Returns:
            配置值
            
        Raises:
            MissingConfigError: 严格模式下缺失必需参数
        """
        # 记录访问
        self._accessed_paths.append(path)
        ConfigValidator._usage_log[path] = ConfigValidator._usage_log.get(path, 0) + 1
        
        # 逐级获取
        keys = path.split('.')
        value = self.config
        
        for i, key in enumerate(keys):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
            if value is None:
                break
        
        # 处理缺失情况
        if value is None:
            if required:
                self._missing_paths.append(path)
                if self.strict_mode:
                    raise MissingConfigError(
                        f"必需配置参数缺失: {path}\n"
                        f"请在config文件中添加此参数"
                    )
                else:
                    logger.warning(f"配置参数缺失: {path}，使用默认值: {default}")
            return default
        
        return value
    
    def get_validated(self, path: str, default: Any = None, 
                      type_hint: Type = None,
                      min_value: float = None,
                      max_value: float = None,
                      choices: List[Any] = None,
                      required: bool = False) -> Any:
        """
        获取并验证配置值
        
        Args:
            path: 配置路径
            default: 默认值
            type_hint: 期望类型
            min_value: 最小值
            max_value: 最大值
            choices: 可选值列表
            required: 是否必需
            
        Returns:
            验证后的配置值
            
        Raises:
            InvalidConfigValueError: 配置值验证失败
        """
        value = self.get_nested(path, default, required)
        
        if value is None:
            return default
        
        # 类型验证
        if type_hint is not None and not isinstance(value, type_hint):
            try:
                value = type_hint(value)
            except (ValueError, TypeError):
                raise InvalidConfigValueError(
                    f"配置参数类型错误: {path}\n"
                    f"期望类型: {type_hint.__name__}，实际类型: {type(value).__name__}"
                )
        
        # 范围验证
        if min_value is not None and value < min_value:
            raise InvalidConfigValueError(
                f"配置参数值过小: {path} = {value}\n"
                f"最小允许值: {min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise InvalidConfigValueError(
                f"配置参数值过大: {path} = {value}\n"
                f"最大允许值: {max_value}"
            )
        
        # 选项验证
        if choices is not None and value not in choices:
            raise InvalidConfigValueError(
                f"配置参数值不在允许范围内: {path} = {value}\n"
                f"允许值: {choices}"
            )
        
        return value
    
    def validate_all_requirements(self) -> Dict[str, Any]:
        """
        验证所有注册的配置需求
        
        Returns:
            验证结果摘要
        """
        errors = []
        validated_values = {}
        
        for path, req in self._requirements.items():
            try:
                value = self.get_validated(
                    path=req.path,
                    default=req.default,
                    type_hint=req.type_hint,
                    min_value=req.min_value,
                    max_value=req.max_value,
                    choices=req.choices,
                    required=req.required
                )
                validated_values[path] = value
            except ConfigValidationError as e:
                errors.append(str(e))
        
        if errors and self.strict_mode:
            raise ConfigValidationError(
                f"配置验证失败，共{len(errors)}个错误:\n" + 
                "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
            )
        
        return {
            'validated_values': validated_values,
            'errors': errors,
            'warnings': self._warnings,
            'missing_paths': self._missing_paths,
        }
    
    def get_usage_report(self) -> Dict[str, Any]:
        """获取参数使用报告"""
        return {
            'accessed_paths': self._accessed_paths,
            'missing_paths': self._missing_paths,
            'usage_counts': dict(ConfigValidator._usage_log),
            'total_accesses': sum(ConfigValidator._usage_log.values()),
        }


class ConfigInjector:
    """
    配置注入器 - 为Agent提供配置访问接口
    
    使用方式：
    1. 在Agent初始化时注入配置
    2. 使用injector.get()获取参数
    3. 所有硬编码参数都应通过注入器获取
    """
    
    def __init__(self, config: Dict[str, Any], agent_type: str, strict: bool = True):
        """
        初始化注入器
        
        Args:
            config: 全局配置字典
            agent_type: 主体类型（farmer, enterprise, government等）
            strict: 严格模式
        """
        self.config = config
        self.agent_type = agent_type
        self.validator = ConfigValidator(config, strict_mode=strict)
        self._cache: Dict[str, Any] = {}
    
    def get(self, param_name: str, default: Any = None, 
            required: bool = False, **validation_kwargs) -> Any:
        """
        获取主体配置参数
        
        Args:
            param_name: 参数名（相对于agent_type的路径）
            default: 默认值
            required: 是否必需
            **validation_kwargs: 额外验证参数
            
        Returns:
            参数值
        """
        # 构建完整路径
        full_path = f"agents.{self.agent_type}.{param_name}"
        
        # 检查缓存
        if full_path in self._cache:
            return self._cache[full_path]
        
        # 获取并验证
        value = self.validator.get_validated(
            path=full_path,
            default=default,
            required=required,
            **validation_kwargs
        )
        
        # 缓存结果
        self._cache[full_path] = value
        
        return value
    
    def get_global(self, path: str, default: Any = None, 
                   required: bool = False, **validation_kwargs) -> Any:
        """
        获取全局配置参数
        
        Args:
            path: 完整配置路径
            default: 默认值
            required: 是否必需
            
        Returns:
            参数值
        """
        if path in self._cache:
            return self._cache[path]
        
        value = self.validator.get_validated(
            path=path,
            default=default,
            required=required,
            **validation_kwargs
        )
        
        self._cache[path] = value
        return value


# ==================== 预定义配置需求 ====================

# 农户配置需求
FARMER_CONFIG_REQUIREMENTS = [
    ConfigRequirement(
        path="agents.farmer.age_dist.mean",
        required=True,
        type_hint=float,
        min_value=20,
        max_value=70,
        description="农户平均年龄"
    ),
    ConfigRequirement(
        path="agents.farmer.age_dist.std",
        required=True,
        type_hint=float,
        min_value=1,
        max_value=20,
        description="农户年龄标准差"
    ),
    ConfigRequirement(
        path="agents.farmer.land_transfer.threshold_in",
        required=True,
        type_hint=float,
        min_value=0,
        max_value=1,
        description="土地租入意愿阈值"
    ),
    ConfigRequirement(
        path="agents.farmer.land_transfer.threshold_out",
        required=True,
        type_hint=float,
        min_value=0,
        max_value=1,
        description="土地流出意愿阈值"
    ),
    ConfigRequirement(
        path="agents.farmer.livelihood_evolution.pure_farmer_threshold",
        required=True,
        type_hint=float,
        min_value=0.5,
        max_value=1,
        description="纯农户判定阈值"
    ),
    ConfigRequirement(
        path="agents.farmer.livelihood_evolution.non_farmer_threshold",
        required=True,
        type_hint=float,
        min_value=0,
        max_value=0.5,
        description="非农户判定阈值"
    ),
]

# 企业配置需求
ENTERPRISE_CONFIG_REQUIREMENTS = [
    ConfigRequirement(
        path="agents.enterprise.expansion_decision.min_area",
        required=True,
        type_hint=int,
        min_value=50,
        max_value=500,
        description="企业最小扩张面积"
    ),
]

# 市场配置需求
MARKET_CONFIG_REQUIREMENTS = [
    ConfigRequirement(
        path="environment.market.base_prices.grain",
        required=True,
        type_hint=float,
        min_value=0.5,
        max_value=5,
        description="粮食基准价格"
    ),
    ConfigRequirement(
        path="environment.market.base_prices.cash_crop",
        required=True,
        type_hint=float,
        min_value=1,
        max_value=10,
        description="经作基准价格"
    ),
]


def validate_config(config: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    验证完整配置
    
    Args:
        config: 配置字典
        strict: 严格模式
        
    Returns:
        验证结果
    """
    # 注册所有需求
    all_requirements = (
        FARMER_CONFIG_REQUIREMENTS + 
        ENTERPRISE_CONFIG_REQUIREMENTS + 
        MARKET_CONFIG_REQUIREMENTS
    )
    ConfigValidator.register_requirements(all_requirements)
    
    # 执行验证
    validator = ConfigValidator(config, strict_mode=strict)
    return validator.validate_all_requirements()


def create_injector_for_agent(config: Dict[str, Any], agent_type: str) -> ConfigInjector:
    """
    为Agent创建配置注入器
    
    Args:
        config: 全局配置
        agent_type: Agent类型
        
    Returns:
        ConfigInjector实例
    """
    return ConfigInjector(config, agent_type, strict=True)
