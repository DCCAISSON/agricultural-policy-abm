# -*- coding: utf-8 -*-
"""
@theory
StagedActivation - 分阶段激活调度器

@critical 2025-11-30: 解决因果时序问题
专家意见：RandomActivation破坏了农业生产的严格因果链
现实逻辑：政策发布 -> 土地流转 -> 劳动力雇佣 -> 种植决策 -> 生产 -> 销售

参考：
- Grimm et al. (2020), "The ODD Protocol for Describing Agent-Based Models"
- Mesa框架的StagedActivation设计

实现方式：
每个Step分为多个微观阶段（Sub-steps），确保所有Agent完成上一阶段后，才进入下一阶段
每个阶段内部使用随机顺序消除先发优势
"""

from mesa.time import BaseScheduler
import random

class StagedActivation(BaseScheduler):
    """
    分阶段激活调度器
    
    将每个时间步划分为多个阶段：
    1. stage_policy_announce: 政策发布阶段
    2. stage_land_market: 土地市场阶段
    3. stage_labor_market: 劳动力市场阶段  
    4. stage_production_decision: 生产决策阶段
    5. stage_production_execute: 生产执行阶段
    6. stage_market_clearing: 市场出清阶段
    7. stage_settlement: 收益结算阶段
    8. stage_strategy_update: 策略更新阶段
    
    每个阶段内，Agent按随机顺序执行对应方法
    """
    
    # 定义阶段顺序和对应的Agent方法
    STAGES = [
        ('stage_policy_announce', ['government']),      # 政府发布政策
        ('stage_land_market', ['farmer', 'enterprise']), # 土地市场参与
        ('stage_labor_market', ['farmer', 'enterprise']), # 劳动力市场参与
        ('stage_production_decision', ['farmer', 'enterprise']), # 生产决策
        ('stage_production_execute', ['farmer', 'enterprise']),  # 生产执行
        ('stage_market_clearing', ['government']),      # 市场出清（政府协调）
        ('stage_settlement', ['farmer', 'enterprise']), # 收益结算
        ('stage_strategy_update', ['farmer', 'enterprise']),  # 策略更新
    ]
    
    def __init__(self, model, shuffle_between_stages=True, shuffle_within_stages=True):
        """
        初始化分阶段调度器
        
        Args:
            model: Mesa模型实例
            shuffle_between_stages: 是否在阶段间随机（默认True）
            shuffle_within_stages: 是否在阶段内随机打乱Agent顺序（默认True）
        """
        super().__init__(model)
        self.shuffle_between_stages = shuffle_between_stages
        self.shuffle_within_stages = shuffle_within_stages
        self.current_stage = None
        self.stage_log = []  # 记录阶段执行情况
        
    def step(self):
        """
        执行一个完整的时间步
        
        按顺序执行所有阶段，每个阶段内Agent随机激活
        """
        self.stage_log = []
        
        for stage_name, agent_types in self.STAGES:
            self.current_stage = stage_name
            self._execute_stage(stage_name, agent_types)
        
        # 增加步数计数
        self.steps += 1
        self.time += 1
        
    def _execute_stage(self, stage_name, agent_types):
        """
        执行单个阶段
        
        Args:
            stage_name: 阶段名称（对应Agent的方法名）
            agent_types: 该阶段需要激活的Agent类型列表
        """
        # 收集该阶段需要激活的Agent
        agents_to_activate = []
        
        for agent in self._agents.values():
            agent_type = self._get_agent_type(agent)
            if agent_type in agent_types:
                # 检查Agent是否有对应的阶段方法
                if hasattr(agent, stage_name):
                    agents_to_activate.append(agent)
        
        # 在阶段内随机打乱顺序（消除先发优势）
        if self.shuffle_within_stages:
            random.shuffle(agents_to_activate)
        
        # 执行该阶段
        executed_count = 0
        for agent in agents_to_activate:
            try:
                stage_method = getattr(agent, stage_name)
                stage_method()
                executed_count += 1
            except Exception as e:
                # 记录错误但继续执行
                print(f"[StagedActivation] {stage_name} error for {agent.unique_id}: {e}")
        
        # 记录执行情况
        self.stage_log.append({
            'stage': stage_name,
            'agents_count': len(agents_to_activate),
            'executed_count': executed_count
        })
    
    def _get_agent_type(self, agent):
        """
        获取Agent类型标识
        
        Returns:
            str: 'farmer', 'enterprise', 'government', 或 'other'
        """
        agent_class = agent.__class__.__name__
        
        if 'Farmer' in agent_class:
            return 'farmer'
        elif 'Enterprise' in agent_class or 'Cooperative' in agent_class:
            return 'enterprise'
        elif 'Government' in agent_class:
            return 'government'
        else:
            return 'other'
    
    def get_stage_log(self):
        """获取阶段执行日志"""
        return self.stage_log


class SimplifiedStagedActivation(BaseScheduler):
    """
    简化版分阶段调度器
    
    只划分三个核心阶段，降低复杂度：
    1. pre_step: 市场准备（土地、劳动力市场）
    2. step: 主决策（使用原有step方法）
    3. post_step: 后处理（策略更新、统计）
    
    适用于不想大幅修改现有Agent代码的情况
    """
    
    def __init__(self, model, shuffle=True):
        """
        初始化简化版调度器
        
        Args:
            model: Mesa模型实例
            shuffle: 是否在每个阶段内随机打乱Agent顺序
        """
        super().__init__(model)
        self.shuffle = shuffle
        
    def step(self):
        """
        执行一个完整的时间步
        """
        # 获取所有Agent并打乱顺序
        agents = list(self._agents.values())
        
        if self.shuffle:
            random.shuffle(agents)
        
        # 阶段1: pre_step（如果Agent有这个方法）
        for agent in agents:
            if hasattr(agent, 'pre_step'):
                agent.pre_step()
        
        # 阶段2: 主step（原有逻辑）
        if self.shuffle:
            random.shuffle(agents)
        for agent in agents:
            agent.step()
        
        # 阶段3: post_step（如果Agent有这个方法）
        if self.shuffle:
            random.shuffle(agents)
        for agent in agents:
            if hasattr(agent, 'post_step'):
                agent.post_step()
        
        self.steps += 1
        self.time += 1


class HybridActivation(BaseScheduler):
    """
    混合激活调度器（推荐使用）
    
    @theory 保持与现有代码兼容，同时确保因果正确性
    
    策略：
    - 使用现有的step()方法作为主执行体
    - 在model层面已经实现了阶段划分（参见model.py的step方法）
    - 此调度器负责在每个阶段内随机打乱Agent顺序
    
    这样可以在不修改Agent代码的情况下，确保：
    1. 消除先发优势（阶段内随机）
    2. 保持因果正确性（阶段间顺序由model控制）
    """
    
    def __init__(self, model, shuffle=True, seed=None):
        """
        初始化混合调度器
        
        Args:
            model: Mesa模型实例
            shuffle: 是否随机打乱顺序
            seed: 随机种子（可选，用于可复现性）
        """
        super().__init__(model)
        self.shuffle = shuffle
        if seed is not None:
            random.seed(seed)
    
    def step(self):
        """
        执行所有Agent的step方法，使用随机顺序
        """
        agents = list(self._agents.values())
        
        if self.shuffle:
            random.shuffle(agents)
        
        for agent in agents:
            agent.step()
        
        self.steps += 1
        self.time += 1
    
    def step_type(self, agent_type):
        """
        只激活特定类型的Agent
        
        @theory 支持分类型激活，用于更细粒度的控制
        
        Args:
            agent_type: Agent类型名称（'farmer', 'enterprise', 'government'）
        """
        agents = []
        
        for agent in self._agents.values():
            agent_class = agent.__class__.__name__.lower()
            if agent_type.lower() in agent_class:
                agents.append(agent)
        
        if self.shuffle:
            random.shuffle(agents)
        
        for agent in agents:
            agent.step()
