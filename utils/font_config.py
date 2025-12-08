# -*- coding: utf-8 -*-
"""
字体配置模块
解决matplotlib中文字体显示问题，特别针对Windows系统优化
@refactor: 加强字体配置，解决字体错乱问题
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import warnings
import platform
import sys
import numpy as np
from matplotlib import rcParams

def clear_matplotlib_cache():
    """清理matplotlib字体缓存"""
    try:
        cache_dir = matplotlib.get_cachedir()
        cache_files = [
            os.path.join(cache_dir, 'fontlist-v330.json'),
            os.path.join(cache_dir, 'fontList.cache'),
            os.path.join(cache_dir, 'fontList.py3k.cache')
        ]
        
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"✓ 已清理缓存文件: {os.path.basename(cache_file)}")
    except Exception as e:
        print(f"清理缓存时出现警告: {e}")

def setup_chinese_fonts():
    """
    配置matplotlib以支持中文显示。
    优先尝试常用中文字体，如果失败则扫描系统字体。
    """
    print("--- 中文字体配置 ---")
    
    # 优先尝试的字体列表
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial Unicode MS']
    
    # 查找并设置字体
    for font_name in preferred_fonts:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            rcParams['font.sans-serif'] = [font_name]
            print(f"[OK] 找到并设置优先字体: {font_name}")
            break
    else:
        # 如果优先字体都找不到，则扫描系统字体
        print("[INFO] 未找到优先中文字体，正在扫描系统字体...")
        
        # 查找系统中所有可用的中文字体
        available_chinese_fonts = []
        for font in fm.fontManager.ttflist:
            # 简单通过字体名称判断（可能不完全准确，但作为备选方案可行）
            if any(char in font.name for char in ['Sim', 'Kai', 'Fang', 'Hei', 'Song', 'Yuan']):
                available_chinese_fonts.append(font.name)
        
        if available_chinese_fonts:
            found_font = available_chinese_fonts[0]
            rcParams['font.sans-serif'] = [found_font]
            print(f"[OK] 找到并设置备选字体: {found_font}")
        else:
            print("[ERROR] 未找到任何可用的中文字体。图表中的中文可能无法正常显示。")
            # 使用无衬线字体作为最后的备选
            rcParams['font.sans-serif'] = ['sans-serif']

    # 解决负号显示问题
    rcParams['axes.unicode_minus'] = False
    print(f"[OK] 当前字体优先级: {rcParams['font.sans-serif']}")
    print("[OK] 负号处理: 已禁用")
    
    # 重新加载字体管理器
    try:
        # 使用已设置的字体而不是'sans-serif'，避免解析错误
        current_font = rcParams['font.sans-serif'][0] if rcParams['font.sans-serif'] else 'SimHei'
        fm.findfont(fm.FontProperties(family=current_font))
    except Exception as e:
        # 忽略字体加载警告，不影响实际使用
        pass

def _verify_font_setting():
    """验证字体设置是否生效"""
    try:
        current_font = plt.rcParams['font.sans-serif'][0]
        print(f"✓ 当前字体优先级: {plt.rcParams['font.sans-serif'][:3]}")
        print(f"✓ 负号处理: {'已启用' if plt.rcParams['axes.unicode_minus'] else '已禁用'}")
        
        # 测试实际使用的字体
        fig, ax = plt.subplots(figsize=(1, 1))
        text = ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        actual_font = text.get_fontfamily()
        print(f"✓ 实际使用字体: {actual_font}")
        plt.close(fig)
        
    except Exception as e:
        warnings.warn(f"字体验证时出现警告: {e}")

def get_available_fonts():
    """
    获取系统中可用的中文字体列表
    用于调试字体问题
    """
    from matplotlib.font_manager import findSystemFonts, FontProperties
    
    font_files = findSystemFonts()
    chinese_fonts_found = []
    
    # 常见中文字体
    common_chinese_fonts = [
        'Microsoft YaHei', 'SimSun', 'SimHei', 'KaiTi', 
        'FangSong', 'LiSu', 'YouYuan', 'NSimSun',
        'PingFang SC', 'Heiti SC', 'WenQuanYi Micro Hei'
    ]
    
    for font_name in common_chinese_fonts:
        try:
            fp = FontProperties(family=font_name)
            if fp.get_name() == font_name:
                chinese_fonts_found.append(font_name)
        except:
            continue
    
    return chinese_fonts_found

def test_chinese_display():
    """
    测试中文显示效果
    生成一个简单的测试图表
    """
    import numpy as np
    
    # 确保字体已设置
    setup_chinese_fonts()
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = -0.5 * np.sin(x)  # 测试负号
    
    # 创建更完整的测试图表
    plt.figure(figsize=(10, 8))
    
    # 第一个子图 - 基本线条
    plt.subplot(2, 1, 1)
    plt.plot(x, y1, label='正弦曲线 (Sin)')
    plt.plot(x, y2, label='余弦曲线 (Cos)')
    plt.plot(x, y3, label='负半正弦 (-0.5·Sin)', linestyle='--')
    plt.title('中文字体测试 - Font Test')
    plt.xlabel('时间 (Time)')
    plt.ylabel('数值 (Value)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加注释
    plt.annotate('峰值', xy=(1.6, 1), xytext=(3, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 第二个子图 - 柱状图
    plt.subplot(2, 1, 2)
    categories = ['第一类别', '第二类别', '第三类别', '第四类别']
    values = [5, 7, 3, 4]
    plt.bar(categories, values)
    plt.title('中文类别标签测试')
    plt.xlabel('类别')
    plt.ylabel('数值')
    
    plt.tight_layout()
    
    # 保存测试图片
    test_file = 'font_test.png'
    plt.savefig(test_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 字体测试图表已保存为: {test_file}")
    return test_file

def fix_fonts_in_all_images(directory):
    """
    修复目录中所有图像的字体问题
    
    @refactor: 添加批量修复功能
    """
    import glob
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    # 确保字体设置正确
    setup_chinese_fonts()
    
    # 查找所有PNG和JPG文件
    image_files = glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)
    image_files.extend(glob.glob(os.path.join(directory, "**", "*.jpg"), recursive=True))
    
    if not image_files:
        print(f"没有在 {directory} 中找到图像文件")
        return
    
    print(f"正在处理 {len(image_files)} 个图像文件...")
    
    # 对每个文件尝试重新生成
    # 注意：这只能处理通过matplotlib生成的图像，而且只有在原始数据可获取时才有效
    # 这里只是一个示例，实际项目中需要更复杂的实现
    
    for image_file in image_files[:3]:  # 仅处理前几个作为示例
        try:
            print(f"处理: {os.path.basename(image_file)}")
            # 实际实现中，需要重新加载数据并重绘图形
            # 这里只是示例，实际无法恢复原始数据
        except Exception as e:
            print(f"处理 {image_file} 时出错: {e}")
    
    print("提示: 要真正修复字体问题，建议重新运行生成图表的脚本")

if __name__ == "__main__":
    print("=== 字体配置测试 ===")
    clear_matplotlib_cache()
    setup_chinese_fonts()
    print(f"可用中文字体: {get_available_fonts()}")
    test_chinese_display() 

    # 用于测试的函数
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        
        setup_chinese_fonts()
        
        plt.figure()
        plt.title("你好，世界！")
        plt.xlabel("X轴标签")
        plt.ylabel("Y轴标签")
        plt.text(0.5, 0.5, "测试文本", ha='center')
        
        # 保存测试图片
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'font_test.png')
        plt.savefig(test_image_path)
        print(f"测试图片已保存到: {test_image_path}")

        # 显示字体路径
        try:
            font_path = fm.findfont(fm.FontProperties(family=rcParams['font.sans-serif']))
            print(f"实际使用字体路径: {font_path}")
        except Exception as e:
            print(f"无法定位字体路径: {e}") 