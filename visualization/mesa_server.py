# -*- coding: utf-8 -*-
"""
Mesa浏览器可视化服务器
基于Mesa框架创建交互式浏览器界面
"""

import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization import ModularServer, Slider, NumberInput
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder
from core.model import MainModel
import yaml


class MesaModelWrapper:
    """
    Mesa模型包装类
    将ModelBuilder和MainModel包装成Mesa兼容的模型
    """
    def __init__(self, n_farmers=300, n_steps=50, seed=42):
        """初始化模型"""
        # 创建ModelBuilder
        self.builder = ModelBuilder(n_farmers=n_farmers, n_steps=n_steps, seed=seed)
        # 构建模型
        self.model = self.builder.build()
        # Mesa需要的属性
        self.schedule = self.model.schedule
        self.grid = self.model.grid
        self.farmers = self.model.farmers
        self.datacollector = self.model.datacollector
    
    def step(self):
        """执行一步模拟"""
        self.model.step()


class LivelihoodChart(ChartModule):
    """生计类型演化图表"""
    def __init__(self):
        super().__init__(
            [
                {"Label": "Pure_Farmer", "Color": "#3498db"},
                {"Label": "Pluriactive", "Color": "#e67e22"},
                {"Label": "Non_Farmer", "Color": "#2ecc71"},
            ],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class IncomeChart(ChartModule):
    """收入分布图表"""
    def __init__(self):
        super().__init__(
            [
                {"Label": "平均收入", "Color": "#9b59b6"},
                {"Label": "中位数收入", "Color": "#34495e"},
            ],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class GiniChart(ChartModule):
    """基尼系数图表"""
    def __init__(self):
        super().__init__(
            [{"Label": "基尼系数", "Color": "#e74c3c"}],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class PriceChart(ChartModule):
    """市场价格图表"""
    def __init__(self):
        super().__init__(
            [
                {"Label": "粮食价格", "Color": "#f39c12"},
                {"Label": "经济作物价格", "Color": "#16a085"},
                {"Label": "日工资", "Color": "#c0392b"},
            ],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class CropFarmerLivelihoodChart(ChartModule):
    """作物类型农户生计类型演化图表"""
    def __init__(self):
        super().__init__(
            [
                {"Label": "粮食作物-纯农户", "Color": "#3498db"},
                {"Label": "粮食作物-兼业农户", "Color": "#e67e22"},
                {"Label": "粮食作物-非农户", "Color": "#2ecc71"},
                {"Label": "经济作物-纯农户", "Color": "#9b59b6"},
                {"Label": "经济作物-兼业农户", "Color": "#f39c12"},
                {"Label": "经济作物-非农户", "Color": "#1abc9c"},
            ],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class EnterpriseTypeChart(ChartModule):
    """企业类型演化图表"""
    def __init__(self):
        super().__init__(
            [
                {"Label": "大型家庭农场", "Color": "#3498db"},
                {"Label": "农业公司", "Color": "#e67e22"},
                {"Label": "合作社", "Color": "#2ecc71"},
            ],
            canvas_height=200,
            data_collector_name="datacollector"
        )


class ModelInfo(TextElement):
    """模型信息文本显示"""
    def render(self, model):
        pure_farmers = sum(1 for f in model.farmers 
                          if f.livelihood_strategy.value == "Pure_Farmer")
        pluriactive = sum(1 for f in model.farmers 
                         if f.livelihood_strategy.value == "Pluriactive")
        non_farmers = sum(1 for f in model.farmers 
                         if f.livelihood_strategy.value == "Non_Farmer")
        
        avg_income = sum(f.total_income for f in model.farmers) / len(model.farmers) if model.farmers else 0
        
        return (f"步数: {model.schedule.steps} | "
                f"纯农户: {pure_farmers} | "
                f"兼业: {pluriactive} | "
                f"非农户: {non_farmers} | "
                f"平均收入: {avg_income:.0f}元")


def farmer_portrayal(agent):
    """农户可视化表现"""
    # @fix: 只显示农户，企业和政府不显示
    if not hasattr(agent, 'livelihood_strategy'):
        return None  # 不是农户，不显示
    
    # 根据生计策略设置颜色
    strategy = agent.livelihood_strategy.value
    if strategy == "Pure_Farmer":
        color = "#3498db"  # 蓝色
    elif strategy == "Pluriactive":
        color = "#e67e22"  # 橙色
    else:  # Non_Farmer
        color = "#2ecc71"  # 绿色
    
    # 根据收入设置大小（收入越高，圆圈越大）
    income = getattr(agent, 'total_income', 0)
    # @fix: 调整大小计算，使圆圈更明显
    if income > 0:
        size = max(2, min(8, income / 30000))  # 收入0-240000映射到2-8像素
    else:
        size = 2  # 零收入也显示，但很小
    
    portrayal = {
        "Shape": "circle",
        "Color": color,
        "Filled": "true",
        "Layer": 0,
        "r": size,
    }
    
    return portrayal


def create_server(config_path=None):
    """创建Mesa可视化服务器"""
    
    # 加载配置
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "master_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建网格（用于显示农户位置）
    # @fix: 使用配置中的网格大小，确保与实际模型匹配
    grid_width = config.get('scale', {}).get('grid_width', 50)
    grid_height = config.get('scale', {}).get('grid_height', 50)
    grid = CanvasGrid(
        farmer_portrayal,
        grid_width,
        grid_height,
        800,
        400
    )
    
    # 创建图表
    livelihood_chart = LivelihoodChart()
    income_chart = IncomeChart()
    gini_chart = GiniChart()
    price_chart = PriceChart()
    crop_farmer_livelihood_chart = CropFarmerLivelihoodChart()  # 新增：作物类型农户生计类型演化
    enterprise_type_chart = EnterpriseTypeChart()  # 新增：企业类型演化
    
    # 模型信息
    model_info = ModelInfo()
    
    # 用户可调参数（Mesa 2.3.2版本使用Slider和NumberInput）
    model_params = {
        "n_farmers": Slider(
            "农户数量",
            value=config.get('scale', {}).get('n_farmers', 300),
            min_value=50,
            max_value=500,
            step=50
        ),
        "n_steps": Slider(
            "模拟步数",
            value=50,
            min_value=10,
            max_value=200,
            step=10
        ),
        "seed": NumberInput(
            "随机种子",
            value=42
        ),
    }
    
    # 创建服务器
    # 使用包装类，使模型与Mesa兼容
    server = ModularServer(
        MesaModelWrapper,
        [grid, livelihood_chart, income_chart, gini_chart, price_chart, 
         crop_farmer_livelihood_chart, enterprise_type_chart, model_info],  # 添加新图表
        "粮食补贴政策多智能体模型",
        model_params
    )
    
    return server


def run_server(port=8521, config_path=None):
    """运行可视化服务器"""
    server = create_server(config_path)
    print(f"启动Mesa可视化服务器...")
    print(f"访问地址: http://127.0.0.1:{port}")
    print(f"按 Ctrl+C 停止服务器")
    server.port = port
    server.launch()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="启动Mesa浏览器可视化")
    parser.add_argument("--port", type=int, default=8521, help="服务器端口")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    run_server(port=args.port, config_path=args.config)

