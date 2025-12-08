# -*- coding: utf-8 -*-
"""
空间可视化模块
======================================

借鉴 spatial-climate-ABM 项目的可视化方法
提供空间动态可视化功能

核心功能：
1. 土地所有权地图 - 展示不同类型农户/企业的土地分布
2. 土地流转网络图 - 可视化流转关系
3. 生计策略空间分布 - 展示农户策略的空间格局
4. 时序快照动画 - 生成动态演化GIF

参考项目：
- spatial-climate-ABM/visualization.py
- an-agent-based-model-of-farmland-transfer (NetLogo)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SpatialVisualizer:
    """
    空间可视化器
    
    提供模型的空间状态可视化功能
    """
    
    def __init__(self, output_dir: Path = None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'results' / 'spatial_maps'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色方案
        self.colors = {
            'small_farmer': '#90EE90',      # 浅绿 - 小农户
            'family_farm': '#FFD700',       # 金色 - 家庭农场
            'enterprise': '#FF6347',        # 番茄红 - 企业
            'abandoned': '#D3D3D3',         # 灰色 - 撂荒
            'grain': '#228B22',             # 森林绿 - 种粮
            'cash_crop': '#8B4513',         # 棕色 - 经作
            'pure_farmer': '#2E8B57',       # 深绿 - 纯农户
            'pluriactive': '#DAA520',       # 金杆色 - 兼业户
            'non_farmer': '#708090',        # 灰石色 - 非农户
        }
        
        # 自定义colormap
        self.land_cmap = LinearSegmentedColormap.from_list(
            'land_ownership',
            ['#90EE90', '#FFD700', '#FF6347', '#D3D3D3'],
            N=4
        )
    
    def plot_land_ownership_map(self, model, step: int = None,
                                title: str = None, save: bool = True) -> plt.Figure:
        """
        绘制土地所有权地图
        
        Args:
            model: GrainSubsidyModel实例
            step: 当前步数（用于文件名）
            title: 图表标题
            save: 是否保存图片
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if title is None:
            title = f'土地所有权分布 (Step {step if step is not None else "?"})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 获取网格尺寸
        try:
            width = model.grid.width
            height = model.grid.height
        except:
            width = 20
            height = 20
        
        # 创建所有权网格
        ownership_grid = np.zeros((height, width))
        crop_grid = np.zeros((height, width))
        
        # 遍历所有土地块
        if hasattr(model, 'land_patches'):
            for patch in model.land_patches:
                if hasattr(patch, 'pos'):
                    x, y = patch.pos
                    
                    # 确定所有者类型
                    owner = getattr(patch, 'owner', None)
                    if owner is None:
                        ownership_grid[y, x] = 3  # 撂荒
                    elif hasattr(owner, 'enterprise_area'):
                        ownership_grid[y, x] = 2  # 企业
                    elif hasattr(owner, 'farmer_scale'):
                        from agents.farmer_agent import FarmerScale
                        if getattr(owner, 'farmer_scale', None) == FarmerScale.FAMILY_FARM:
                            ownership_grid[y, x] = 1  # 家庭农场
                        else:
                            ownership_grid[y, x] = 0  # 小农户
                    else:
                        ownership_grid[y, x] = 0  # 默认小农户
                    
                    # 确定作物类型
                    crop_type = getattr(patch, 'crop_type', None)
                    if crop_type == 'grain':
                        crop_grid[y, x] = 0
                    elif crop_type == 'cash':
                        crop_grid[y, x] = 1
                    else:
                        crop_grid[y, x] = 0.5
        else:
            # 如果没有land_patches，从agents提取信息
            agents = list(model.schedule.agents)
            farmers = [a for a in agents if hasattr(a, 'total_income') and not hasattr(a, 'enterprise_area')]
            enterprises = [a for a in agents if hasattr(a, 'enterprise_area')]
            
            # 随机分配网格位置（简化可视化）
            np.random.seed(42)
            positions = np.random.choice(width * height, len(farmers) + len(enterprises), replace=False)
            
            for i, farmer in enumerate(farmers):
                pos = positions[i]
                x, y = pos % width, pos // width
                from agents.farmer_agent import FarmerScale, LivelihoodStrategy
                if getattr(farmer, 'farmer_scale', None) == FarmerScale.FAMILY_FARM:
                    ownership_grid[y, x] = 1
                else:
                    ownership_grid[y, x] = 0
            
            for i, ent in enumerate(enterprises):
                pos = positions[len(farmers) + i]
                x, y = pos % width, pos // width
                ownership_grid[y, x] = 2
        
        # 绘制所有权地图
        im = ax.imshow(ownership_grid, cmap=self.land_cmap, 
                      interpolation='nearest', origin='lower',
                      vmin=0, vmax=3)
        
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['small_farmer'], edgecolor='black', label='小农户'),
            mpatches.Patch(facecolor=self.colors['family_farm'], edgecolor='black', label='家庭农场'),
            mpatches.Patch(facecolor=self.colors['enterprise'], edgecolor='black', label='企业'),
            mpatches.Patch(facecolor=self.colors['abandoned'], edgecolor='black', label='撂荒/未分配'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        plt.tight_layout()
        
        if save:
            step_str = f'{step:03d}' if step is not None else 'final'
            filepath = self.output_dir / f'ownership_map_step_{step_str}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"所有权地图已保存: {filepath}")
        
        return fig
    
    def plot_livelihood_strategy_map(self, model, step: int = None,
                                     title: str = None, save: bool = True) -> plt.Figure:
        """
        绘制生计策略空间分布图
        
        Args:
            model: GrainSubsidyModel实例
            step: 当前步数
            title: 图表标题
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if title is None:
            title = f'生计策略空间分布 (Step {step if step is not None else "?"})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 获取所有农户
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'total_income') and 
                  not hasattr(a, 'enterprise_area') and hasattr(a, 'livelihood_strategy')]
        
        # 分类
        from agents.farmer_agent import LivelihoodStrategy
        pure_farmers = [f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.PURE_FARMER]
        pluriactive = [f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE]
        non_farmers = [f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.NON_FARMER]
        
        # 获取位置（如果没有位置信息，随机生成）
        try:
            width = model.grid.width
            height = model.grid.height
        except:
            width = 20
            height = 20
        
        np.random.seed(42)
        
        def get_positions(agent_list, n_total, offset):
            positions = []
            for i, agent in enumerate(agent_list):
                if hasattr(agent, 'pos') and agent.pos is not None:
                    positions.append(agent.pos)
                else:
                    # 随机位置
                    x = (i + offset) % width
                    y = ((i + offset) // width) % height
                    positions.append((x, y))
            return positions
        
        # 绘制各类农户
        pure_pos = get_positions(pure_farmers, len(farmers), 0)
        pluri_pos = get_positions(pluriactive, len(farmers), len(pure_farmers))
        non_pos = get_positions(non_farmers, len(farmers), len(pure_farmers) + len(pluriactive))
        
        if pure_pos:
            ax.scatter([p[0] for p in pure_pos], [p[1] for p in pure_pos],
                      c=self.colors['pure_farmer'], s=100, marker='o', 
                      label=f'纯农户 ({len(pure_farmers)})', edgecolors='black', alpha=0.8)
        
        if pluri_pos:
            ax.scatter([p[0] for p in pluri_pos], [p[1] for p in pluri_pos],
                      c=self.colors['pluriactive'], s=100, marker='s',
                      label=f'兼业户 ({len(pluriactive)})', edgecolors='black', alpha=0.8)
        
        if non_pos:
            ax.scatter([p[0] for p in non_pos], [p[1] for p in non_pos],
                      c=self.colors['non_farmer'], s=100, marker='^',
                      label=f'非农户 ({len(non_farmers)})', edgecolors='black', alpha=0.8)
        
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        total = len(farmers)
        stats_text = (f"总农户: {total}\n"
                     f"纯农户: {len(pure_farmers)} ({len(pure_farmers)/total*100:.1f}%)\n"
                     f"兼业户: {len(pluriactive)} ({len(pluriactive)/total*100:.1f}%)\n"
                     f"非农户: {len(non_farmers)} ({len(non_farmers)/total*100:.1f}%)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            step_str = f'{step:03d}' if step is not None else 'final'
            filepath = self.output_dir / f'livelihood_map_step_{step_str}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"生计策略地图已保存: {filepath}")
        
        return fig
    
    def plot_land_transfer_network(self, model, step: int = None,
                                   title: str = None, save: bool = True) -> plt.Figure:
        """
        绘制土地流转网络图
        
        展示土地流转的供需关系网络
        
        Args:
            model: GrainSubsidyModel实例
            step: 当前步数
            title: 图表标题
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if title is None:
            title = f'土地流转网络 (Step {step if step is not None else "?"})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 获取农户和企业
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'total_income') and 
                  not hasattr(a, 'enterprise_area')]
        enterprises = [a for a in agents if hasattr(a, 'enterprise_area')]
        
        # 获取网格尺寸
        try:
            width = model.grid.width
            height = model.grid.height
        except:
            width = 20
            height = 20
        
        # 为所有代理分配位置
        np.random.seed(42)
        all_agents = farmers + enterprises
        positions = {}
        for i, agent in enumerate(all_agents):
            if hasattr(agent, 'pos') and agent.pos is not None:
                positions[agent.unique_id] = agent.pos
            else:
                x = (i * 3) % width + np.random.uniform(-0.3, 0.3)
                y = (i * 3 // width) % height + np.random.uniform(-0.3, 0.3)
                positions[agent.unique_id] = (x, y)
        
        # 分类并绘制节点
        from agents.farmer_agent import LivelihoodStrategy
        
        # 供给方（流出土地的农户）
        suppliers = [f for f in farmers if getattr(f, 'leased_out_area', 0) > 0]
        # 需求方（流入土地的农户/企业）
        demanders_farmers = [f for f in farmers if getattr(f, 'leased_in_area', 0) > 0]
        demanders_ent = [e for e in enterprises if getattr(e, 'operated_area', 0) > 0]
        
        # 中立方
        neutral_farmers = [f for f in farmers if f not in suppliers and f not in demanders_farmers]
        
        # 绘制节点
        # 供给方 - 红色
        if suppliers:
            sup_x = [positions[f.unique_id][0] for f in suppliers]
            sup_y = [positions[f.unique_id][1] for f in suppliers]
            sup_sizes = [50 + getattr(f, 'leased_out_area', 0) * 5 for f in suppliers]
            ax.scatter(sup_x, sup_y, c='red', s=sup_sizes, marker='v',
                      label=f'流出方 ({len(suppliers)})', edgecolors='black', alpha=0.7, zorder=3)
        
        # 需求方农户 - 蓝色
        if demanders_farmers:
            dem_x = [positions[f.unique_id][0] for f in demanders_farmers]
            dem_y = [positions[f.unique_id][1] for f in demanders_farmers]
            dem_sizes = [50 + getattr(f, 'leased_in_area', 0) * 5 for f in demanders_farmers]
            ax.scatter(dem_x, dem_y, c='blue', s=dem_sizes, marker='^',
                      label=f'流入方-农户 ({len(demanders_farmers)})', edgecolors='black', alpha=0.7, zorder=3)
        
        # 需求方企业 - 绿色
        if demanders_ent:
            ent_x = [positions[e.unique_id][0] for e in demanders_ent]
            ent_y = [positions[e.unique_id][1] for e in demanders_ent]
            ent_sizes = [100 + getattr(e, 'operated_area', 0) * 2 for e in demanders_ent]
            ax.scatter(ent_x, ent_y, c='green', s=ent_sizes, marker='s',
                      label=f'流入方-企业 ({len(demanders_ent)})', edgecolors='black', alpha=0.7, zorder=3)
        
        # 中立农户 - 灰色（较小）
        if neutral_farmers:
            neu_x = [positions[f.unique_id][0] for f in neutral_farmers]
            neu_y = [positions[f.unique_id][1] for f in neutral_farmers]
            ax.scatter(neu_x, neu_y, c='lightgray', s=30, marker='o',
                      label=f'中立 ({len(neutral_farmers)})', edgecolors='gray', alpha=0.5, zorder=2)
        
        # 绘制流转关系箭头（如果有记录）
        if hasattr(model, 'land_clearing_house'):
            clearing_house = model.land_clearing_house
            if hasattr(clearing_house, 'transaction_history'):
                for tx in clearing_house.transaction_history[-50:]:  # 只绘制最近50笔
                    supplier_id = tx.get('supplier_id')
                    demander_id = tx.get('demander_id')
                    
                    if supplier_id in positions and demander_id in positions:
                        sx, sy = positions[supplier_id]
                        dx, dy = positions[demander_id]
                        
                        # 绘制箭头
                        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                                   arrowprops=dict(arrowstyle='->', color='purple', 
                                                  alpha=0.3, linewidth=1))
        
        ax.set_xlim(-1, width + 1)
        ax.set_ylim(-1, height + 1)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        total_leased_out = sum(getattr(f, 'leased_out_area', 0) for f in farmers)
        total_leased_in = sum(getattr(f, 'leased_in_area', 0) for f in farmers)
        total_ent_area = sum(getattr(e, 'operated_area', 0) for e in enterprises)
        
        stats_text = (f"流转统计:\n"
                     f"流出总面积: {total_leased_out:.1f} 亩\n"
                     f"农户流入: {total_leased_in:.1f} 亩\n"
                     f"企业经营: {total_ent_area:.1f} 亩")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            step_str = f'{step:03d}' if step is not None else 'final'
            filepath = self.output_dir / f'transfer_network_step_{step_str}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"流转网络图已保存: {filepath}")
        
        return fig
    
    def plot_comprehensive_dashboard(self, model, step: int = None,
                                    title: str = None, save: bool = True) -> plt.Figure:
        """
        绘制综合仪表板
        
        4个子图：
        1. 土地所有权地图
        2. 生计策略分布
        3. 土地流转网络
        4. 关键指标柱状图
        
        Args:
            model: GrainSubsidyModel实例
            step: 当前步数
            title: 图表标题
            save: 是否保存
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(18, 14))
        
        if title is None:
            title = f'空间动态综合仪表板 (Step {step if step is not None else "?"})'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 4个子图布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 土地所有权地图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ownership_subplot(ax1, model)
        ax1.set_title('(a) 土地所有权分布', fontsize=12)
        
        # 2. 生计策略分布
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_livelihood_subplot(ax2, model)
        ax2.set_title('(b) 生计策略空间分布', fontsize=12)
        
        # 3. 土地流转网络
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_transfer_subplot(ax3, model)
        ax3.set_title('(c) 土地流转网络', fontsize=12)
        
        # 4. 关键指标
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_subplot(ax4, model)
        ax4.set_title('(d) 关键指标', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            step_str = f'{step:03d}' if step is not None else 'final'
            filepath = self.output_dir / f'dashboard_step_{step_str}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"综合仪表板已保存: {filepath}")
        
        return fig
    
    def _plot_ownership_subplot(self, ax, model):
        """绘制所有权子图"""
        try:
            width = model.grid.width
            height = model.grid.height
        except:
            width, height = 20, 20
        
        # 简化版本
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'total_income') and not hasattr(a, 'enterprise_area')]
        enterprises = [a for a in agents if hasattr(a, 'enterprise_area')]
        
        ownership_grid = np.zeros((height, width))
        np.random.seed(42)
        
        for i, f in enumerate(farmers[:min(len(farmers), width*height)]):
            x, y = i % width, i // width
            from agents.farmer_agent import FarmerScale
            if getattr(f, 'farmer_scale', None) == FarmerScale.FAMILY_FARM:
                ownership_grid[y, x] = 1
            else:
                ownership_grid[y, x] = 0
        
        for i, e in enumerate(enterprises[:min(len(enterprises), 10)]):
            x, y = (len(farmers) + i) % width, (len(farmers) + i) // width
            if y < height:
                ownership_grid[y, x] = 2
        
        ax.imshow(ownership_grid, cmap=self.land_cmap, interpolation='nearest', 
                 origin='lower', vmin=0, vmax=3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 简化图例
        legend_elements = [
            mpatches.Patch(color='#90EE90', label='小农'),
            mpatches.Patch(color='#FFD700', label='家庭农场'),
            mpatches.Patch(color='#FF6347', label='企业'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_livelihood_subplot(self, ax, model):
        """绘制生计策略子图"""
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'livelihood_strategy')]
        
        from agents.farmer_agent import LivelihoodStrategy
        
        strategy_counts = {
            'PURE_FARMER': len([f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.PURE_FARMER]),
            'PLURIACTIVE': len([f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE]),
            'NON_FARMER': len([f for f in farmers if f.livelihood_strategy == LivelihoodStrategy.NON_FARMER]),
        }
        
        colors = [self.colors['pure_farmer'], self.colors['pluriactive'], self.colors['non_farmer']]
        labels = ['纯农户', '兼业户', '非农户']
        values = list(strategy_counts.values())
        
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    
    def _plot_transfer_subplot(self, ax, model):
        """绘制流转子图"""
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'leased_out_area')]
        
        leased_out = sum(getattr(f, 'leased_out_area', 0) for f in farmers)
        leased_in = sum(getattr(f, 'leased_in_area', 0) for f in farmers)
        self_operated = sum(getattr(f, 'own_operated_area', getattr(f, 'contracted_area', 0)) for f in farmers)
        
        categories = ['流出', '流入', '自营']
        values = [leased_out, leased_in, self_operated]
        colors = ['red', 'blue', 'gray']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('面积 (亩)')
        ax.set_xlabel('土地状态')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_metrics_subplot(self, ax, model):
        """绘制关键指标子图"""
        agents = list(model.schedule.agents)
        farmers = [a for a in agents if hasattr(a, 'total_income') and not hasattr(a, 'enterprise_area')]
        
        # 提取指标
        metrics = {
            '总农户': len(farmers),
            '平均收入\n(万元)': np.mean([f.total_income for f in farmers if hasattr(f, 'total_income')]) / 10000 if farmers else 0,
            '粮食产量\n(万kg)': sum(getattr(f, 'grain_output', 0) for f in farmers) / 10000,
            '流转率\n(%)': sum(getattr(f, 'leased_out_area', 0) for f in farmers) / \
                         max(sum(getattr(f, 'contracted_area', 1) for f in farmers), 1) * 100,
        }
        
        x = np.arange(len(metrics))
        values = list(metrics.values())
        labels = list(metrics.keys())
        
        bars = ax.bar(x, values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('数值')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    def create_animation(self, model, n_steps: int = 30,
                        interval: int = 500, save_gif: bool = True) -> Optional[str]:
        """
        创建空间动态演化动画
        
        Args:
            model: GrainSubsidyModel实例
            n_steps: 模拟步数
            interval: 帧间隔(ms)
            save_gif: 是否保存为GIF
        
        Returns:
            保存路径（如果保存）
        """
        print(f"生成动画: {n_steps}步...")
        
        # 收集每步的图像
        frames = []
        
        for step in range(n_steps):
            # 运行一步
            model.step()
            
            # 绘制当前状态
            fig = self.plot_comprehensive_dashboard(model, step=step, save=False)
            
            # 保存为临时文件
            temp_path = self.output_dir / f'_temp_frame_{step:03d}.png'
            fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frames.append(temp_path)
            
            if (step + 1) % 10 == 0:
                print(f"  完成 {step + 1}/{n_steps} 帧")
        
        if save_gif:
            try:
                from PIL import Image
                
                # 读取所有帧
                images = [Image.open(fp) for fp in frames]
                
                # 保存GIF
                gif_path = self.output_dir / 'spatial_animation.gif'
                images[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=images[1:],
                    duration=interval,
                    loop=0
                )
                print(f"动画已保存: {gif_path}")
                
                # 清理临时文件
                for fp in frames:
                    fp.unlink()
                
                return str(gif_path)
                
            except ImportError:
                print("请安装Pillow以生成GIF: pip install Pillow")
                return None
        
        return None


# ============== 便捷函数 ==============

def visualize_model_state(model, output_dir: Path = None):
    """
    一键可视化模型当前状态
    
    Args:
        model: GrainSubsidyModel实例
        output_dir: 输出目录
    """
    viz = SpatialVisualizer(output_dir)
    
    step = getattr(model, 'step_count', getattr(model, 'schedule', None))
    if hasattr(step, 'steps'):
        step = step.steps
    elif not isinstance(step, int):
        step = None
    
    print("生成空间可视化...")
    viz.plot_comprehensive_dashboard(model, step=step)
    
    print("完成！")


# ============== 主函数 ==============

def main():
    """测试空间可视化"""
    print("空间可视化模块测试")
    print("=" * 50)
    
    # 尝试导入模型
    try:
        from core.model import GrainSubsidyModel
        
        # 加载配置
        import yaml
        config_path = PROJECT_ROOT / 'config' / 'master_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        print("初始化模型...")
        model = GrainSubsidyModel(config)
        
        # 运行几步
        print("运行10步模拟...")
        for _ in range(10):
            model.step()
        
        # 可视化
        print("生成可视化...")
        visualize_model_state(model)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
