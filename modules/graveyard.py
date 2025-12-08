# -*- coding: utf-8 -*-
"""
@module graveyard.py
@description 墓地统计系统 - 避免幸存者偏差
@created 2025-12-04

核心功能：
1. 追踪所有退出主体的最终状态
2. 记录退出原因和退出时的关键指标
3. 提供统计分析接口
4. 支持事后分析和因果推断

理论依据：
- 幸存者偏差(Survivorship Bias)是ABM模型常见问题
- 只观察存活主体会导致对系统动态的误判
- 墓地系统记录"失败者"的信息，提供完整图景
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import numpy as np


class ExitReason(Enum):
    """退出原因枚举"""
    # 农户退出原因
    AGE_RETIREMENT = "age_retirement"           # 年龄退出（老年化）
    URBANIZATION = "urbanization"               # 城镇化迁移
    BANKRUPTCY = "bankruptcy"                   # 破产退出
    LAND_LOSS = "land_loss"                     # 失地农户
    GENERATIONAL_EXIT = "generational_exit"     # 代际退出（无人接班）
    VOLUNTARY_EXIT = "voluntary_exit"           # 主动退出（转行）
    HEALTH_ISSUE = "health_issue"               # 健康问题
    POLICY_FORCED = "policy_forced"             # 政策性退出
    
    # 企业退出原因
    PROFIT_NEGATIVE = "profit_negative"         # 连续亏损
    CAPITAL_DEPLETED = "capital_depleted"       # 资金链断裂
    SCALE_TOO_SMALL = "scale_too_small"         # 规模不经济
    MARKET_SQUEEZE = "market_squeeze"           # 市场挤压
    SUBSIDY_CUT = "subsidy_cut"                 # 补贴削减影响
    
    # 服务商退出原因
    EQUIPMENT_OBSOLETE = "equipment_obsolete"   # 设备老化
    COMPETITION = "competition"                 # 竞争失败
    
    # 通用
    OTHER = "other"                             # 其他原因


@dataclass
class ExitRecord:
    """退出记录"""
    # 基本信息
    agent_id: str                               # 主体ID
    agent_type: str                             # 主体类型: farmer, enterprise, service_provider
    exit_step: int                              # 退出时间步
    exit_reason: ExitReason                     # 退出原因
    
    # 退出时状态快照
    final_state: Dict[str, Any] = field(default_factory=dict)
    
    # 历史轨迹摘要
    history_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class Graveyard:
    """
    墓地统计系统
    
    核心功能：
    1. 记录退出主体
    2. 提供统计分析
    3. 支持事后分析
    """
    
    def __init__(self):
        """初始化墓地系统"""
        self.records: List[ExitRecord] = []
        self.exit_counts_by_step: Dict[int, int] = {}
        self.exit_counts_by_reason: Dict[ExitReason, int] = {}
        self.exit_counts_by_type: Dict[str, int] = {}
    
    def record_exit(self, agent, exit_reason: ExitReason, step: int, notes: str = "") -> ExitRecord:
        """
        记录主体退出
        
        Args:
            agent: 退出的主体
            exit_reason: 退出原因
            step: 退出时间步
            notes: 备注
            
        Returns:
            ExitRecord: 退出记录
        """
        # 确定主体类型
        agent_type = self._determine_agent_type(agent)
        
        # 捕获最终状态快照
        final_state = self._capture_final_state(agent, agent_type)
        
        # 生成历史轨迹摘要
        history_summary = self._generate_history_summary(agent, agent_type)
        
        # 创建退出记录
        record = ExitRecord(
            agent_id=str(agent.unique_id),
            agent_type=agent_type,
            exit_step=step,
            exit_reason=exit_reason,
            final_state=final_state,
            history_summary=history_summary,
            notes=notes
        )
        
        # 存储记录
        self.records.append(record)
        
        # 更新统计计数
        self._update_counts(record)
        
        return record
    
    def _determine_agent_type(self, agent) -> str:
        """确定主体类型"""
        class_name = agent.__class__.__name__
        if 'Farmer' in class_name:
            return 'farmer'
        elif 'Enterprise' in class_name:
            return 'enterprise'
        elif 'Provider' in class_name or 'Service' in class_name:
            return 'service_provider'
        else:
            return 'unknown'
    
    def _capture_final_state(self, agent, agent_type: str) -> Dict[str, Any]:
        """捕获主体最终状态快照"""
        state = {}
        
        if agent_type == 'farmer':
            # 农户状态
            state['age'] = getattr(agent, 'age', None)
            state['capital'] = getattr(agent, 'capital', 0)
            state['total_income'] = getattr(agent, 'total_income', 0)
            state['agricultural_income'] = getattr(agent, 'agricultural_income', 0)
            state['off_farm_income'] = getattr(agent, 'off_farm_income', 0)
            state['livelihood_strategy'] = str(getattr(agent, 'livelihood_strategy', None))
            state['land_area'] = agent.get_operated_land_area() if hasattr(agent, 'get_operated_land_area') else 0
            state['household_size'] = getattr(agent, 'household_size', 0)
            state['planned_crop'] = str(getattr(agent, 'planned_crop', None))
            
        elif agent_type == 'enterprise':
            # 企业状态
            state['capital'] = getattr(agent, 'capital', 0)
            state['enterprise_type'] = str(getattr(agent, 'enterprise_type', None))
            state['efficiency_factor'] = getattr(agent, 'efficiency_factor', 1.0)
            state['operated_area'] = agent.get_operated_land_area() if hasattr(agent, 'get_operated_land_area') else 0
            state['profit'] = getattr(agent, 'profit', 0)
            state['crop_portfolio'] = {str(k): v for k, v in getattr(agent, 'crop_portfolio', {}).items()}
            
        elif agent_type == 'service_provider':
            # 服务商状态
            state['capital'] = getattr(agent, 'capital', 0)
            state['current_capacity'] = getattr(agent, 'current_capacity', 0)
            state['service_area_used'] = getattr(agent, 'service_area_used', 0)
            state['annual_revenue'] = getattr(agent, 'annual_revenue', 0)
            state['annual_cost'] = getattr(agent, 'annual_cost', 0)
            state['cumulative_profit'] = getattr(agent, 'cumulative_profit', 0)
        
        return state
    
    def _generate_history_summary(self, agent, agent_type: str) -> Dict[str, Any]:
        """生成历史轨迹摘要"""
        summary = {}
        
        if agent_type == 'farmer':
            # 农户历史
            if hasattr(agent, 'income_memory') and agent.income_memory:
                summary['income_trend'] = agent.income_memory[-5:] if len(agent.income_memory) >= 5 else agent.income_memory
                summary['avg_income'] = np.mean(agent.income_memory) if agent.income_memory else 0
            
            if hasattr(agent, 'strategy_history'):
                summary['strategy_changes'] = len(set(agent.strategy_history))
                summary['last_strategies'] = [str(s) for s in agent.strategy_history[-3:]] if agent.strategy_history else []
            
            if hasattr(agent, 'crop_choice_history'):
                summary['crop_switches'] = len(agent.crop_choice_history)
                
        elif agent_type == 'enterprise':
            # 企业历史
            if hasattr(agent, 'profit_history') and agent.profit_history:
                summary['profit_trend'] = agent.profit_history[-5:] if len(agent.profit_history) >= 5 else agent.profit_history
                summary['avg_profit'] = np.mean(agent.profit_history) if agent.profit_history else 0
                summary['consecutive_losses'] = self._count_consecutive_losses(agent.profit_history)
        
        return summary
    
    def _count_consecutive_losses(self, profit_history: List[float]) -> int:
        """计算连续亏损次数"""
        if not profit_history:
            return 0
        count = 0
        for profit in reversed(profit_history):
            if profit < 0:
                count += 1
            else:
                break
        return count
    
    def _update_counts(self, record: ExitRecord):
        """更新统计计数"""
        # 按时间步统计
        step = record.exit_step
        self.exit_counts_by_step[step] = self.exit_counts_by_step.get(step, 0) + 1
        
        # 按原因统计
        reason = record.exit_reason
        self.exit_counts_by_reason[reason] = self.exit_counts_by_reason.get(reason, 0) + 1
        
        # 按类型统计
        agent_type = record.agent_type
        self.exit_counts_by_type[agent_type] = self.exit_counts_by_type.get(agent_type, 0) + 1
    
    # ==================== 统计分析接口 ====================
    
    def get_exit_summary(self) -> Dict[str, Any]:
        """获取退出统计摘要"""
        return {
            'total_exits': len(self.records),
            'by_type': dict(self.exit_counts_by_type),
            'by_reason': {str(k.value): v for k, v in self.exit_counts_by_reason.items()},
            'by_step': dict(self.exit_counts_by_step),
        }
    
    def get_farmer_exits(self) -> List[ExitRecord]:
        """获取所有农户退出记录"""
        return [r for r in self.records if r.agent_type == 'farmer']
    
    def get_enterprise_exits(self) -> List[ExitRecord]:
        """获取所有企业退出记录"""
        return [r for r in self.records if r.agent_type == 'enterprise']
    
    def get_exits_by_reason(self, reason: ExitReason) -> List[ExitRecord]:
        """按原因获取退出记录"""
        return [r for r in self.records if r.exit_reason == reason]
    
    def get_exits_in_step_range(self, start_step: int, end_step: int) -> List[ExitRecord]:
        """获取指定时间范围内的退出记录"""
        return [r for r in self.records if start_step <= r.exit_step <= end_step]
    
    def compute_exit_rate(self, total_population: int) -> float:
        """计算总退出率"""
        if total_population == 0:
            return 0.0
        return len(self.records) / total_population
    
    def compute_exit_rate_by_type(self, agent_type: str, initial_population: int) -> float:
        """按类型计算退出率"""
        if initial_population == 0:
            return 0.0
        exits = self.exit_counts_by_type.get(agent_type, 0)
        return exits / initial_population
    
    def get_average_exit_state(self, agent_type: str) -> Dict[str, float]:
        """获取某类型主体退出时的平均状态"""
        records = [r for r in self.records if r.agent_type == agent_type]
        if not records:
            return {}
        
        # 收集所有数值型状态
        state_sums = {}
        state_counts = {}
        
        for record in records:
            for key, value in record.final_state.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    state_sums[key] = state_sums.get(key, 0) + value
                    state_counts[key] = state_counts.get(key, 0) + 1
        
        # 计算平均值
        return {
            key: state_sums[key] / state_counts[key]
            for key in state_sums
            if state_counts[key] > 0
        }
    
    def get_exit_age_distribution(self) -> Dict[str, List[int]]:
        """获取农户退出年龄分布"""
        farmer_records = self.get_farmer_exits()
        ages = [r.final_state.get('age') for r in farmer_records if r.final_state.get('age') is not None]
        
        if not ages:
            return {'ages': [], 'mean': 0, 'median': 0, 'std': 0}
        
        return {
            'ages': ages,
            'mean': np.mean(ages),
            'median': np.median(ages),
            'std': np.std(ages),
            'under_40': sum(1 for a in ages if a < 40),
            'between_40_60': sum(1 for a in ages if 40 <= a < 60),
            'over_60': sum(1 for a in ages if a >= 60),
        }
    
    def generate_report(self) -> str:
        """生成墓地统计报告"""
        summary = self.get_exit_summary()
        
        report = []
        report.append("=" * 60)
        report.append("墓地统计报告 - 退出主体分析")
        report.append("=" * 60)
        report.append(f"\n总退出数: {summary['total_exits']}")
        
        # 按类型统计
        report.append("\n【按主体类型】")
        for agent_type, count in summary['by_type'].items():
            report.append(f"  {agent_type}: {count}")
        
        # 按原因统计
        report.append("\n【按退出原因】")
        for reason, count in sorted(summary['by_reason'].items(), key=lambda x: -x[1]):
            report.append(f"  {reason}: {count}")
        
        # 农户退出年龄分布
        age_dist = self.get_exit_age_distribution()
        if age_dist['ages']:
            report.append("\n【农户退出年龄分布】")
            report.append(f"  平均年龄: {age_dist['mean']:.1f}岁")
            report.append(f"  中位年龄: {age_dist['median']:.1f}岁")
            report.append(f"  40岁以下: {age_dist['under_40']}人")
            report.append(f"  40-60岁: {age_dist['between_40_60']}人")
            report.append(f"  60岁以上: {age_dist['over_60']}人")
        
        # 平均退出状态
        farmer_avg = self.get_average_exit_state('farmer')
        if farmer_avg:
            report.append("\n【农户平均退出状态】")
            for key, value in farmer_avg.items():
                if key in ['capital', 'total_income', 'agricultural_income', 'off_farm_income']:
                    report.append(f"  {key}: {value:,.0f}元")
                elif key == 'land_area':
                    report.append(f"  {key}: {value:.1f}亩")
                else:
                    report.append(f"  {key}: {value:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出为字典格式（用于JSON序列化）"""
        return {
            'summary': self.get_exit_summary(),
            'records': [
                {
                    'agent_id': r.agent_id,
                    'agent_type': r.agent_type,
                    'exit_step': r.exit_step,
                    'exit_reason': r.exit_reason.value,
                    'final_state': r.final_state,
                    'history_summary': r.history_summary,
                    'notes': r.notes,
                }
                for r in self.records
            ],
            'analysis': {
                'farmer_exit_age_distribution': self.get_exit_age_distribution(),
                'farmer_avg_exit_state': self.get_average_exit_state('farmer'),
                'enterprise_avg_exit_state': self.get_average_exit_state('enterprise'),
            }
        }
