"""
生成包含所有更新内容的Word文档

Usage:
    python analysis/generate_docx.py --output docs/模型实现完整文档.docx
"""

import argparse
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


def add_title(doc, text, level=1):
    """添加标题"""
    heading = doc.add_heading(text, level=level)
    heading.style.font.name = '宋体'
    heading.style.font.size = Pt(16 if level == 1 else 14 if level == 2 else 12)
    return heading


def add_paragraph(doc, text, bold=False, indent=False):
    """添加段落"""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.5)
    run = p.add_run(text)
    run.font.name = '宋体'
    run.font.size = Pt(12)
    if bold:
        run.bold = True
    return p


def add_code_block(doc, code, language='python'):
    """添加代码块"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    run = p.add_run(code)
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 0, 0)
    p.style = 'List Paragraph'


def create_document():
    """创建完整的Word文档"""
    doc = Document()
    
    # 设置中文字体
    doc.styles['Normal'].font.name = '宋体'
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    
    # 标题页
    title = doc.add_heading('粮食补贴政策多智能体模型', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('实现进度与技术文档')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.runs[0]
    run.font.size = Pt(14)
    
    date_p = doc.add_paragraph(f'生成时间：{datetime.now().strftime("%Y年%m月%d日")}')
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_page_break()
    
    # 目录（手动添加，Word会自动生成）
    add_title(doc, '目录', 1)
    doc.add_paragraph('1. 项目概述')
    doc.add_paragraph('2. 阶段一：数据预处理 + 补贴机制扩展')
    doc.add_paragraph('3. 阶段二：市场模块统一')
    doc.add_paragraph('4. 阶段三：目标矩校准')
    doc.add_paragraph('5. 关键文件说明')
    doc.add_paragraph('6. 测试与验证')
    doc.add_paragraph('7. 待办事项与已知问题')
    
    doc.add_page_break()
    
    # 1. 项目概述
    add_title(doc, '1. 项目概述', 1)
    
    add_paragraph(doc, '本项目旨在构建一个基于多智能体建模（ABM）的粮食补贴政策仿真系统，通过模拟农户、企业和政府等主体的决策行为，评估不同补贴政策对农业生产、土地流转和收入分配的影响。', indent=False)
    
    add_paragraph(doc, '模型采用多层次博弈理论框架，实现了：', indent=False)
    add_paragraph(doc, '• 异质性主体的行为建模（前景理论、贝叶斯学习、社会网络）', indent=True)
    add_paragraph(doc, '• 内生化的市场出清机制（农产品、劳动力、土地市场）', indent=True)
    add_paragraph(doc, '• 多样化的补贴政策情景（现金、实物、分段补贴）', indent=True)
    add_paragraph(doc, '• 基于实证数据的参数校准（CLES数据集）', indent=True)
    
    doc.add_page_break()
    
    # 2. 阶段一：数据预处理 + 补贴机制扩展
    add_title(doc, '2. 阶段一：数据预处理 + 补贴机制扩展', 1)
    
    add_title(doc, '2.1 CLES数据预处理', 2)
    
    add_paragraph(doc, '完成时间：2024年', bold=True)
    
    add_paragraph(doc, '实现内容：', bold=True)
    add_paragraph(doc, '创建了 analysis/cles_preprocessing.py 脚本，支持处理4个CLES数据集：', indent=False)
    add_paragraph(doc, '• CLES_2020_2021_household_panel_mapping_label.csv（农户面板数据）', indent=True)
    add_paragraph(doc, '• CLES_2020_2021_village_panel_label_mapping.csv（村庄面板数据）', indent=True)
    add_paragraph(doc, '• CLES_2022_household_mapping_label.csv（2022年农户数据）', indent=True)
    add_paragraph(doc, '• CLES_2022_village_label_mapping.csv（2022年村庄数据）', indent=True)
    
    add_paragraph(doc, '技术方法：', bold=True)
    add_paragraph(doc, '• 使用 gb18030 编码读取CSV文件，正确处理中文变量名', indent=True)
    add_paragraph(doc, '• 第一行为变量名，第二行为中文释义（作为元数据保存）', indent=True)
    add_paragraph(doc, '• 自动类型转换（数值列）', indent=True)
    add_paragraph(doc, '• 输出为 Parquet 格式（如果可用）或 CSV 作为备选', indent=True)
    
    add_paragraph(doc, '输出文件：', bold=True)
    add_paragraph(doc, '• data/cles/*.parquet: 清洗后的数据文件', indent=True)
    add_paragraph(doc, '• config/cles_variable_metadata.json: 变量元数据字典', indent=True)
    
    add_paragraph(doc, '测试结果：', bold=True)
    add_paragraph(doc, '成功处理所有4个数据集：', indent=False)
    add_paragraph(doc, '• household_2020_2021: 915个变量', indent=True)
    add_paragraph(doc, '• household_2022: 918个变量', indent=True)
    add_paragraph(doc, '• village_2020_2021: 993个变量', indent=True)
    add_paragraph(doc, '• village_2022: 749个变量', indent=True)
    
    add_title(doc, '2.2 补贴机制扩展', 2)
    
    add_paragraph(doc, '完成时间：2024年', bold=True)
    
    add_paragraph(doc, '实现内容：', bold=True)
    add_paragraph(doc, '扩展了 config/policy_params.json 支持多种补贴场景，更新了核心代码以支持时滞、实物补贴等功能。', indent=False)
    
    add_paragraph(doc, '补贴场景：', bold=True)
    
    add_paragraph(doc, '场景1：baseline_cash（基准面积现金补贴）', bold=True)
    add_paragraph(doc, '• 按经营者面积发放现金补贴', indent=True)
    add_paragraph(doc, '• 基准补贴 100 元/亩，补贴率 15%', indent=True)
    add_paragraph(doc, '• 无时滞', indent=True)
    
    add_paragraph(doc, '场景2：owner_tiered_cash（所有者分段现金补贴）', bold=True)
    add_paragraph(doc, '• 依据自有土地面积分段给予补贴', indent=True)
    add_paragraph(doc, '• 分段点: [50, 150] 亩', indent=True)
    add_paragraph(doc, '• 分段率: [0.12, 0.18, 0.22]', indent=True)
    add_paragraph(doc, '• 时滞: 1期', indent=True)
    
    add_paragraph(doc, '场景3：mixed_in_kind_support（现金+实物综合补贴）', bold=True)
    add_paragraph(doc, '• 50%现金 + 50%实物（农资抵扣、技术培训）', indent=True)
    add_paragraph(doc, '• 农资抵扣: 35元/亩', indent=True)
    add_paragraph(doc, '• 技术培训: 12元/亩', indent=True)
    
    add_paragraph(doc, '关键方法：', bold=True)
    add_paragraph(doc, '• Environment.calculate_subsidy_for_actor(): 计算补贴金额（支持现金、实物、混合）', indent=True)
    add_paragraph(doc, '• Environment._get_tiered_rate(): 分段补贴率计算', indent=True)
    add_paragraph(doc, '• Environment._compute_in_kind_value(): 实物补贴价值计算', indent=True)
    add_paragraph(doc, '• GovernmentAgent.distribute_subsidies(): 补贴发放（支持时滞）', indent=True)
    
    add_paragraph(doc, '关键假设：', bold=True)
    add_paragraph(doc, '• 实物补贴的价值等于其市场价值', indent=True)
    add_paragraph(doc, '• 时滞通过 payment_lag 参数控制，延迟发放', indent=True)
    add_paragraph(doc, '• 补贴目标可以是 owner（土地所有者）或 operator（实际耕作者）', indent=True)
    
    doc.add_page_break()
    
    # 3. 阶段二：市场模块统一
    add_title(doc, '3. 阶段二：市场模块统一', 1)
    
    add_paragraph(doc, '完成时间：2024年', bold=True)
    
    add_paragraph(doc, '实现内容：', bold=True)
    add_paragraph(doc, '确认并统一了市场出清机制，确保所有市场在每个时间步统一更新，保持年度循环一致性。', indent=False)
    
    add_paragraph(doc, '市场机制架构：', bold=True)
    add_paragraph(doc, '• core/environment.py: 实现农产品和劳动力市场出清', indent=True)
    add_paragraph(doc, '  - 农产品市场：基于供需平衡的价格形成，价格粘性控制调整速度', indent=True)
    add_paragraph(doc, '  - 劳动力市场：供需缺口调整工资率，考虑农业和非农部门需求', indent=True)
    add_paragraph(doc, '• modules/land_clearing_house.py: 实现土地租赁市场', indent=True)
    add_paragraph(doc, '  - 中央清算所机制，多轮竞价博弈', indent=True)
    add_paragraph(doc, '  - 村精英优先权机制', indent=True)
    
    add_paragraph(doc, '决策说明：', bold=True)
    add_paragraph(doc, '• modules/market.py 中的 MarketMechanism 类未被使用，保留为参考', indent=True)
    add_paragraph(doc, '• 土地市场由 LandClearingHouse 处理，农产品和劳动力市场由 Environment 处理', indent=True)
    add_paragraph(doc, '• 保持年度循环一致性：所有市场在每个时间步统一更新', indent=True)
    
    doc.add_page_break()
    
    # 4. 阶段三：目标矩校准
    add_title(doc, '4. 阶段三：目标矩校准', 1)
    
    add_title(doc, '4.1 批量模拟框架', 2)
    
    add_paragraph(doc, '完成时间：2024年', bold=True)
    
    add_paragraph(doc, '实现内容：', bold=True)
    add_paragraph(doc, '创建了 analysis/batch_simulation.py 批量运行脚本，实现了模型统计量收集和结果聚合功能。', indent=False)
    
    add_paragraph(doc, '收集的统计量：', bold=True)
    add_paragraph(doc, '1. 企业类型分布', indent=True)
    add_paragraph(doc, '2. 作物结构（粮食/经济作物比例）', indent=True)
    add_paragraph(doc, '3. 土地流转（流转率、平均流转价格）', indent=True)
    add_paragraph(doc, '4. 收入分布（平均收入、基尼系数）', indent=True)
    add_paragraph(doc, '5. 生产率指标（单产）', indent=True)
    add_paragraph(doc, '6. 补贴统计', indent=True)
    add_paragraph(doc, '7. 市场价格', indent=True)
    
    add_paragraph(doc, '使用方法：', bold=True)
    add_code_block(doc, 'python analysis/batch_simulation.py --runs 50 --steps 30 --output results/batch_output.json')
    
    add_title(doc, '4.2 校准管理器集成', 2)
    
    add_paragraph(doc, '完成时间：2024年', bold=True)
    
    add_paragraph(doc, '实现内容：', bold=True)
    add_paragraph(doc, '更新了 core/calibration.py 中的 calculate_simulated_moments() 方法，集成了批量模拟框架，实现了目标函数计算（基于SMM方法）。', indent=False)
    
    add_paragraph(doc, '目标矩（基于中国农业统计年鉴2017-2023）：', bold=True)
    add_paragraph(doc, '• 企业类型分布: FAMILY_FARM_LARGE 60%, AGRI_CORPORATION 25%, COOPERATIVE 15%', indent=True)
    add_paragraph(doc, '• 作物结构: 粮食 70%, 经济作物 30%', indent=True)
    add_paragraph(doc, '• 土地流转率: 36.73%', indent=True)
    add_paragraph(doc, '• 平均流转价格: 850元/亩/年', indent=True)
    add_paragraph(doc, '• 基尼系数: 0.375', indent=True)
    add_paragraph(doc, '• 平均收入: 21691元', indent=True)
    add_paragraph(doc, '• 粮食单产: 5845.33公斤/公顷', indent=True)
    
    add_paragraph(doc, '校准方法：', bold=True)
    add_paragraph(doc, '• Simulated Method of Moments (SMM)', indent=True)
    add_paragraph(doc, '• 加权最小二乘目标函数', indent=True)
    add_paragraph(doc, '• 支持多种优化算法（Nelder-Mead, L-BFGS-B等）', indent=True)
    
    doc.add_page_break()
    
    # 5. 关键文件说明
    add_title(doc, '5. 关键文件说明', 1)
    
    add_paragraph(doc, '核心实现文件：', bold=True)
    add_paragraph(doc, '• analysis/cles_preprocessing.py - 数据预处理脚本', indent=True)
    add_paragraph(doc, '• analysis/batch_simulation.py - 批量模拟框架', indent=True)
    add_paragraph(doc, '• config/policy_params.json - 补贴场景配置', indent=True)
    add_paragraph(doc, '• core/environment.py - 补贴计算逻辑和市场出清', indent=True)
    add_paragraph(doc, '• agents/government_agent.py - 补贴发放机制', indent=True)
    add_paragraph(doc, '• core/calibration.py - 校准管理器（已集成批量模拟）', indent=True)
    
    add_paragraph(doc, '文档文件：', bold=True)
    add_paragraph(doc, '• docs/model_system_overview.md - 模型系统概述', indent=True)
    add_paragraph(doc, '• docs/implementation_progress.md - 实现进度文档', indent=True)
    
    add_paragraph(doc, '数据文件：', bold=True)
    add_paragraph(doc, '• data/cles/*.parquet - 清洗后的CLES数据', indent=True)
    add_paragraph(doc, '• config/cles_variable_metadata.json - 变量元数据', indent=True)
    
    doc.add_page_break()
    
    # 6. 测试与验证
    add_title(doc, '6. 测试与验证', 1)
    
    add_paragraph(doc, '已完成测试：', bold=True)
    add_paragraph(doc, '✓ 数据预处理脚本测试 - 成功处理所有4个CLES数据集', indent=True)
    add_paragraph(doc, '✓ 代码语法检查 - 所有文件通过linter检查', indent=True)
    
    add_paragraph(doc, '待完成测试：', bold=True)
    add_paragraph(doc, '□ 完整模型运行测试', indent=True)
    add_paragraph(doc, '□ 批量模拟测试', indent=True)
    add_paragraph(doc, '□ 校准流程测试', indent=True)
    add_paragraph(doc, '□ 与历史数据对比验证', indent=True)
    add_paragraph(doc, '□ 敏感性分析', indent=True)
    add_paragraph(doc, '□ 政策情景分析', indent=True)
    
    doc.add_page_break()
    
    # 7. 待办事项与已知问题
    add_title(doc, '7. 待办事项与已知问题', 1)
    
    add_title(doc, '7.1 高优先级待办', 2)
    add_paragraph(doc, '1. 运行批量模拟，验证模型稳定性', indent=True)
    add_paragraph(doc, '2. 实现参数映射，将校准参数更新到模型配置', indent=True)
    add_paragraph(doc, '3. 运行完整校准流程，生成校准报告', indent=True)
    add_paragraph(doc, '4. 验证校准后的模型能否重现目标矩', indent=True)
    
    add_title(doc, '7.2 中优先级待办', 2)
    add_paragraph(doc, '1. 添加敏感性分析功能', indent=True)
    add_paragraph(doc, '2. 实现交叉验证', indent=True)
    add_paragraph(doc, '3. 完善文档和示例', indent=True)
    
    add_title(doc, '7.3 低优先级待办', 2)
    add_paragraph(doc, '1. 优化模型运行速度', indent=True)
    add_paragraph(doc, '2. 添加可视化功能', indent=True)
    add_paragraph(doc, '3. 实现参数空间探索', indent=True)
    
    add_title(doc, '7.4 已知问题', 2)
    add_paragraph(doc, '1. 补贴时滞实现：当前实现简单，可能需要更复杂的时滞机制', indent=True)
    add_paragraph(doc, '2. 实物补贴：实物补贴的价值计算可能需要更细致的建模', indent=True)
    add_paragraph(doc, '3. 参数映射：校准参数到模型配置的映射需要完善', indent=True)
    add_paragraph(doc, '4. 模型运行时间：批量运行可能需要较长时间，考虑并行化', indent=True)
    
    doc.add_page_break()
    
    # 附录：代码示例
    add_title(doc, '附录A：关键代码示例', 1)
    
    add_title(doc, 'A.1 批量模拟使用示例', 2)
    add_code_block(doc, '''# 运行50次模拟，每次30步
python analysis/batch_simulation.py \\
    --runs 50 \\
    --steps 30 \\
    --output results/batch_output.json \\
    --verbose''')
    
    add_title(doc, 'A.2 数据预处理使用示例', 2)
    add_code_block(doc, '''# 处理所有数据集
python analysis/cles_preprocessing.py

# 处理指定数据集
python analysis/cles_preprocessing.py \\
    --datasets household_2020_2021 village_2022 \\
    --verbose''')
    
    add_title(doc, 'A.3 补贴场景配置示例', 2)
    add_code_block(doc, '''# 在 master_config.yaml 中设置
policy:
  government:
    scenario_id: "baseline_cash"  # 或 "owner_tiered_cash", "mixed_in_kind_support"''')
    
    return doc


def main():
    parser = argparse.ArgumentParser(description="生成包含所有更新内容的Word文档")
    parser.add_argument('--output', type=str, default='docs/模型实现完整文档.docx',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    print("正在生成Word文档...")
    doc = create_document()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    
    print(f"文档已生成: {output_path}")


if __name__ == '__main__':
    main()


