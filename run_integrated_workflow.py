"""\
集成化模型运行脚本
==================
- 单次运行：复用原 `run_simulation_demo.py` 的多图可视化与摘要
- 补贴率情景：批量调整 `policy.overrides.grain_subsidies_rate`，输出对比图与摘要
- Mesa 可视化：一键启动浏览器服务器
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from core.model_builder import ModelBuilder

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class SimulationHistory:
    data: Dict[str, List[float]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def final_step(self) -> int:
        return int(self.data['step'][-1])


def load_config(config_path: Path) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_override(config: Dict, key: str, value) -> Dict:
    cfg = deepcopy(config)
    cfg.setdefault('policy', {})
    cfg['policy'].setdefault('overrides', {})
    cfg['policy']['overrides'][key] = value
    return cfg


def run_model(config: Dict, steps: int, verbose: bool = True) -> Tuple[SimulationHistory, object]:
    cfg = deepcopy(config)
    cfg.setdefault('experiment', {})
    cfg['experiment']['steps'] = steps

    builder = ModelBuilder(cfg)
    model = builder.build()

    history = {
        'step': [],
        'grain_price': [],
        'cash_crop_price': [],
        'pure_farmer_count': [],
        'pluriactive_count': [],
        'non_farmer_count': [],
        'total_grain_supply': [],
        'total_cash_supply': [],
        'avg_farmer_income': [],
        'enterprise_profit_rate': [],
        'land_gini': [],
        'total_subsidy': []
    }

    for step in range(steps):
        model.step()

        history['step'].append(step + 1)
        history['grain_price'].append(model.environment.get_market_price('crop_price'))
        history['cash_crop_price'].append(model.environment.get_market_price('cash_crop'))

        pure_count = 0
        pluri_count = 0
        non_count = 0
        for farmer in model.farmers:
            strategy_name = str(getattr(farmer, 'livelihood_strategy', 'Unknown'))
            if 'PURE' in strategy_name.upper():
                pure_count += 1
            elif 'PLURI' in strategy_name.upper():
                pluri_count += 1
            else:
                non_count += 1
        history['pure_farmer_count'].append(pure_count)
        history['pluriactive_count'].append(pluri_count)
        history['non_farmer_count'].append(non_count)

        grain_supply = sum(getattr(farmer, 'last_yield', 0) for farmer in model.farmers
                           if getattr(getattr(farmer, 'last_crop_choice', None), 'value', None) == 'Grain')
        cash_supply = sum(getattr(farmer, 'last_yield', 0) for farmer in model.farmers
                          if getattr(getattr(farmer, 'last_crop_choice', None), 'value', None) == 'Cash_Crop')
        for ent in model.agricultural_enterprises:
            if hasattr(ent, 'last_yields') and isinstance(ent.last_yields, dict):
                grain_supply += ent.last_yields.get('grain', 0)
                cash_supply += ent.last_yields.get('cash_crop', 0)
        history['total_grain_supply'].append(grain_supply / 1000)
        history['total_cash_supply'].append(cash_supply / 1000)

        avg_income = float(np.mean([getattr(f, 'total_income', 0) for f in model.farmers]))
        history['avg_farmer_income'].append(avg_income)

        profitable = sum(1 for e in model.agricultural_enterprises
                         if getattr(e, 'profit', 0) > 0)
        active = sum(1 for e in model.agricultural_enterprises
                     if getattr(e, 'total_revenue', 0) > 0)
        profit_rate = profitable / active if active else 0
        history['enterprise_profit_rate'].append(profit_rate * 100)

        gini = 0
        if hasattr(model, 'get_gini_coefficient_land'):
            try:
                gini = model.get_gini_coefficient_land()
            except Exception:
                gini = 0
        history['land_gini'].append(gini)

        subsidy = sum(getattr(f, 'subsidy_income', 0) for f in model.farmers)
        subsidy += sum(getattr(e, 'subsidy_revenue', 0) for e in model.agricultural_enterprises)
        history['total_subsidy'].append(subsidy / 10000)

        if verbose and ((step + 1) % 5 == 0 or step == 0):
            print(f"Step {step+1:02d}: 粮价={history['grain_price'][-1]:.2f}, 经作价={history['cash_crop_price'][-1]:.2f}, "
                  f"纯农户={pure_count}, 兼业={pluri_count}, 非农={non_count}")

    return SimulationHistory(history), model


def generate_dashboard(history: SimulationHistory, save_path: Path) -> None:
    data = history.data
    steps = data['step']

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('中国粮食补贴政策多智能体模型 - 模拟结果', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(steps, data['grain_price'], 'b-o', label='粮食价格', markersize=4)
    ax1.plot(steps, data['cash_crop_price'], 'g-s', label='经济作物价格', markersize=4)
    ax1.axhline(y=1.2, color='r', linestyle='--', alpha=0.5, label='最低收购价')
    ax1.set_title('市场价格动态')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格 (元/斤)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.stackplot(steps,
                  data['pure_farmer_count'],
                  data['pluriactive_count'],
                  data['non_farmer_count'],
                  labels=['纯农户(含家庭农场)', '兼业户', '非农户'],
                  colors=['#2ecc71', '#f39c12', '#e74c3c'],
                  alpha=0.85)
    ax2.set_title('生计策略演化\n(纯农户=农业收入>70%)')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('农户数量')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    ax3.bar(steps, data['total_grain_supply'], color='#3498db', alpha=0.85)
    if max(data['total_grain_supply']) > 0:
        z = np.polyfit(steps, data['total_grain_supply'], 1)
        ax3.plot(steps, np.poly1d(z)(steps), 'r--', alpha=0.7, label='趋势')
        ax3.legend(fontsize=8)
    ax3.set_title('粮食供给变化')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('供给量 (吨)')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[0, 3]
    ax4.bar(steps, data['total_cash_supply'], color='#27ae60', alpha=0.85)
    if max(data['total_cash_supply']) > 0:
        z = np.polyfit(steps, data['total_cash_supply'], 1)
        ax4.plot(steps, np.poly1d(z)(steps), 'r--', alpha=0.7, label='趋势')
        ax4.legend(fontsize=8)
    ax4.set_title('经济作物供给变化')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('供给量 (吨)')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 0]
    ax5.fill_between(steps, data['avg_farmer_income'], alpha=0.3, color='#9b59b6')
    ax5.plot(steps, data['avg_farmer_income'], 'o-', color='#9b59b6', markersize=4)
    ax5.set_title('农户平均收入')
    ax5.set_xlabel('时间步')
    ax5.set_ylabel('平均收入 (元)')
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 1]
    ax6_twin = ax6.twinx()
    line1, = ax6.plot(steps, data['enterprise_profit_rate'], 'b-o', label='企业盈利率', markersize=4)
    line2, = ax6_twin.plot(steps, data['land_gini'], 'r-s', label='土地基尼系数', markersize=4)
    ax6.set_xlabel('时间步')
    ax6.set_ylabel('企业盈利率 (%)', color='b')
    ax6_twin.set_ylabel('土地基尼系数', color='r')
    ax6.set_title('企业盈利率 & 土地集中度')
    ax6.legend([line1, line2], ['企业盈利率', '土地基尼系数'], fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)

    ax7 = axes[1, 2]
    ax7.bar(steps, data['total_subsidy'], color='#e67e22', alpha=0.85)
    ax7.axhline(y=200, color='r', linestyle='--', alpha=0.7, label='预算上限(200万)')
    ax7.set_title('年度补贴发放')
    ax7.set_xlabel('时间步')
    ax7.set_ylabel('补贴总额 (万元)')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)

    ax8 = axes[1, 3]
    categories = ['纯农户(含家庭农场)', '兼业户', '非农户']
    initial = [data['pure_farmer_count'][0], data['pluriactive_count'][0], data['non_farmer_count'][0]]
    final = [data['pure_farmer_count'][-1], data['pluriactive_count'][-1], data['non_farmer_count'][-1]]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax8.bar(x - width/2, initial, width, label='初始', color='#3498db', alpha=0.85)
    bars2 = ax8.bar(x + width/2, final, width, label='最终', color='#e74c3c', alpha=0.85)
    ax8.set_title('生计策略变化对比')
    ax8.set_ylabel('农户数量')
    ax8.set_xticks(x)
    ax8.set_xticklabels(categories, fontsize=9)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    for idx, (ini, fin) in enumerate(zip(initial, final)):
        change = fin - ini
        ax8.annotate(f"{change:+d}", xy=(x[idx], max(ini, fin) + 3),
                     ha='center', va='bottom', fontweight='bold', color='green' if change >= 0 else 'red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary(history: SimulationHistory) -> str:
    data = history.data
    summary = {
        'grain_price': (data['grain_price'][0], data['grain_price'][-1]),
        'cash_crop_price': (data['cash_crop_price'][0], data['cash_crop_price'][-1]),
        'pure_farmer': (data['pure_farmer_count'][0], data['pure_farmer_count'][-1]),
        'pluriactive': (data['pluriactive_count'][0], data['pluriactive_count'][-1]),
        'non_farmer': (data['non_farmer_count'][0], data['non_farmer_count'][-1]),
        'avg_income': (data['avg_farmer_income'][0], data['avg_farmer_income'][-1]),
        'profit_rate_final': data['enterprise_profit_rate'][-1],
        'land_gini_final': data['land_gini'][-1],
        'total_subsidy': sum(data['total_subsidy'])
    }
    lines = ["=" * 70, "模拟结果摘要", "=" * 70]
    lines.append(f"粮价: {summary['grain_price'][0]:.2f} → {summary['grain_price'][1]:.2f} 元/斤")
    lines.append(f"经作价: {summary['cash_crop_price'][0]:.2f} → {summary['cash_crop_price'][1]:.2f} 元/斤")
    lines.append(f"纯农户: {summary['pure_farmer'][0]} → {summary['pure_farmer'][1]}")
    lines.append(f"兼业户: {summary['pluriactive'][0]} → {summary['pluriactive'][1]}")
    lines.append(f"非农户: {summary['non_farmer'][0]} → {summary['non_farmer'][1]}")
    lines.append(f"平均农户收入: {summary['avg_income'][0]:.0f} → {summary['avg_income'][1]:.0f} 元")
    lines.append(f"企业盈利率(最终): {summary['profit_rate_final']:.2f}%")
    lines.append(f"土地基尼系数(最终): {summary['land_gini_final']:.3f}")
    lines.append(f"累计补贴: {summary['total_subsidy']:.1f} 万元")
    text = "\n".join(lines)
    print(text)
    return text


def run_single_mode(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    history, _ = run_model(config, steps=args.steps, verbose=not args.quiet)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'simulation_data.csv'
    history.to_frame().to_csv(csv_path, index=False, encoding='utf-8-sig')

    dashboard_path = output_dir / 'dashboard.png'
    generate_dashboard(history, dashboard_path)

    summary_text = print_summary(history)
    summary_path = output_dir / 'summary_report.txt'
    summary_path.write_text(summary_text, encoding='utf-8')

    summary_json = {
        'timestamp': timestamp,
        'steps': args.steps,
        'outputs': {
            'dashboard': str(dashboard_path),
            'data': str(csv_path),
            'summary': str(summary_path)
        }
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[OK] 单次运行完成，结果位于: {output_dir}")


def run_policy_sweep_mode(args: argparse.Namespace) -> None:
    if not args.subsidy_list:
        raise ValueError("policy-sweep 模式需要 --subsidy-list 参数，例如 0.8,1.0,1.2")

    config_base = load_config(Path(args.config))
    subsidies = [float(x.strip()) for x in args.subsidy_list.split(',') if x.strip()]

    records = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"policy_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for value in subsidies:
        cfg = ensure_override(config_base, 'grain_subsidies_rate', value)
        history, _ = run_model(cfg, steps=args.steps, verbose=False)
        df = history.to_frame()
        final = df.iloc[-1]
        records.append({
            'subsidy_multiplier': value,
            'grain_price': final['grain_price'],
            'cash_crop_price': final['cash_crop_price'],
            'pure_farmer': final['pure_farmer_count'],
            'avg_income': final['avg_farmer_income'],
            'enterprise_profit_rate': final['enterprise_profit_rate'],
            'grain_supply_ton': final['total_grain_supply'],
            'cash_supply_ton': final['total_cash_supply'],
            'land_gini': final['land_gini'],
            'total_subsidy_per_year': final['total_subsidy']
        })

    df = pd.DataFrame(records).sort_values('subsidy_multiplier')
    sweep_csv = output_dir / 'policy_sweep_summary.csv'
    df.to_csv(sweep_csv, index=False, encoding='utf-8-sig')

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(df['subsidy_multiplier'], df['grain_supply_ton'], 'o-', color='#1b5e20', label='粮食供给(吨)')
    ax1.set_xlabel('补贴倍率（相对于基准350元/亩）')
    ax1.set_ylabel('粮食供给 (吨)', color='#1b5e20')
    ax1.grid(True, alpha=0.3)
    ax2.plot(df['subsidy_multiplier'], df['avg_income'], 's--', color='#0d47a1', label='农户平均收入')
    ax2.set_ylabel('平均农户收入 (元)', color='#0d47a1')
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout()
    sweep_plot = output_dir / 'policy_sweep.png'
    plt.savefig(sweep_plot, dpi=200)
    plt.close(fig)

    print(f"\n[OK] 补贴率情景评估完成，结果位于: {output_dir}")


def run_mesa_mode(args: argparse.Namespace) -> None:
    from visualization.mesa_server import run_server

    print("=" * 60)
    print("Mesa 浏览器可视化")
    print("=" * 60)
    print(f"访问地址: http://127.0.0.1:{args.port}")
    run_server(port=args.port, config_path=args.config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='集成模型运行与可视化脚本')
    parser.add_argument('--mode', choices=['single', 'policy-sweep', 'mesa'], default='single')
    parser.add_argument('--config', default='config/master_config.yaml', help='配置文件路径')
    parser.add_argument('--steps', type=int, default=20, help='模拟步数')
    parser.add_argument('--output-dir', default='results', help='输出目录')
    parser.add_argument('--tag', default='run', help='单次运行结果子目录标签')
    parser.add_argument('--subsidy-list', help='补贴倍率列表，例如 0.8,1.0,1.2，用于policy-sweep')
    parser.add_argument('--port', type=int, default=8521, help='Mesa可视化端口')
    parser.add_argument('--quiet', action='store_true', help='单次运行时不输出逐步日志')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == 'single':
        run_single_mode(args)
    elif args.mode == 'policy-sweep':
        run_policy_sweep_mode(args)
    else:
        run_mesa_mode(args)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
