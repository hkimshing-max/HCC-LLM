# --*-- conding:utf-8 --*--
# @Time : 2026/5/6 17:28
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : model_test_results_present_with_dsv4.py
# @Software : PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

# ==========================================
# 全局配置与路径
# ==========================================
# 已修改为直接读取真实的 Excel 文件 (含 DeepSeek-V4-flash 的新结果)
EXCEL_PATH = os.environ.get('HCC_METRICS_WORKBOOK', 'data/model_test_results.xlsx')
OUTPUT_DIR = os.environ.get('HCC_OUTPUT_DIR', 'outputs/model_test_results_present_with_dsv4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置顶级学术图表的主题风格 (更干净的白底灰线)
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.2,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--"
})
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42  # 保证导出PDF时文字可编辑

# ==========================================
# 学术高级配色 (Nature Publishing Group 风格)
# ==========================================
# 暖色系：用于强调 Multi-agents
COLOR_MA_1 = '#E64B35'  # 珊瑚红
COLOR_MA_2 = '#F39B7F'  # 莫兰迪橙
COLOR_MA_3 = '#A78BC4'  # 紫罗兰 (DeepSeek-V4-flash)

# 冷色/中性系：用于 CoT 和 基线
COLOR_COT_1 = '#4DBBD5'  # 学术青
COLOR_COT_2 = '#3C5488'  # 深海蓝
COLOR_BASE  = '#999999'  # 银灰色

# ==========================================
# 权衡图专用全新配色
# ==========================================
COLOR_SENS = '#9ED17B'
COLOR_SPEC = '#9DC7DD'

def get_color_and_style(model_name, method_name):
    """根据模型和方法动态分配学术配色和线型"""
    if 'Multi' in method_name:
        if 'DeepSeek-V4' in model_name: return COLOR_MA_3, '-', 3.0, 11
        elif 'DeepSeek' in model_name: return COLOR_MA_1, '-', 3.5, 10
        else: return COLOR_MA_2, '-', 2.5, 9
    else:
        if 'DeepSeek-V3.2' in model_name: return COLOR_COT_1, '--', 2.0, 8
        elif 'Qwen' in model_name and '7B' not in model_name: return COLOR_COT_2, '--', 2.0, 7
        else: return COLOR_BASE, ':', 2.0, 5 # 7B 蒸馏小模型等

# ==========================================
# 图表 A：ROC-AUC 时序衰减折线图 (展示所有模型)
# ==========================================
def plot_roc_temporal_decay(df):
    output_pdf = os.path.join(OUTPUT_DIR, "Fig2A_ROC_AUC_Temporal_Decay.pdf")

    # 五个随访时间窗，按结束月份升序 (同为 0 起点时短窗在前)，保证累积窗 0-24 / 0-60 也画出
    time_windows = ['0-6 months', '0-24 months', '6-24 months', '24-60 months', '0-60 months']
    # 仅绘制包含这些时间窗的纯 AUC 数据
    df_plot = df[df['Time window'].isin(time_windows)].copy()

    plt.figure(figsize=(10, 6.5))
    ax = plt.gca()

    # 动态获取所有模型组合
    configs = df_plot[['model', 'Method']].drop_duplicates().values

    for model, method in configs:
        subset = df_plot[(df_plot['model'] == model) & (df_plot['Method'] == method)]
        if subset.empty: continue

        color, ls, lw, z = get_color_and_style(model, method)
        # 将 MA 修改为 Multi-agent
        label = f"{model} ({'Multi-agent' if 'Multi' in method else 'CoT'})"

        y_values = []
        for tw in time_windows: # 连线全部五个随访窗
            val = subset[subset['Time window'] == tw]['roc_auc'].values
            y_values.append(val[0] if len(val) > 0 else np.nan)

        # 绘制折线
        ax.plot(time_windows, y_values, marker='o', markersize=9 if 'Multi' in method else 6,
                color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

        # 为深海红色(DS-V3.2 MA)凸显最高点
        if 'DeepSeek-V3.2' in model and 'Multi' in method and not np.isnan(y_values[0]):
            ax.annotate(f"{y_values[0]:.3f}",
                        xy=(0, y_values[0]), xytext=(0, 15),
                        textcoords="offset points", ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color=color)

    sns.despine(ax=ax, top=True, right=True)
    # ax.set_title('Dynamic Predictive Performance (ROC-AUC) Over Time', fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('')
    ax.set_ylabel('ROC-AUC Score', fontsize=18)
    ax.set_ylim(0.45, 0.72)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='#EEEEEE', shadow=True)

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300)
    plt.show()
    plt.close()
    print(f"[Panel A] 成功生成: {output_pdf}")


# ==========================================
# 图表 B：多维度诊断效能雷达图 (按时间窗循环)
# ==========================================
def plot_radar_charts(df):
    target_windows = ['0-6 months', '0-24 months', '24-60 months','0-60 months']
    # 添加 ppv precision
    metrics = ['roc_auc', 'accuracy', 'sensitivity', 'npv', 'specificity', 'ppv precision']
    display_metrics = ['ROC-AUC', 'Accuracy', 'Sensitivity', 'NPV', 'Specificity', 'PPV']
    num_vars = len(metrics)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    configs = [
        ('DeepSeek-V3.2', 'Multi agents'),
        ('DeepSeek-V4-flash', 'Multi agents'),
        ('Qwen3-8B', 'Multi agents'),
        ('DeepSeek-V3.2', 'chain of thought'),
        ('Qwen3-8B', 'chain of thought'),
        ('DeepSeek-R1-Distill-Qwen-7B', 'chain of thought')
    ]

    # ✅ 外圈块状颜色配置
    bg_colors = ['#AFCBE3', '#BFD8B8', '#F3D999', '#F4B183', '#E6B8AF', '#C5D9F1']

    for idx, window in enumerate(target_windows):
        safe_window_name = window.replace(' ', '_').replace('-', '_')
        output_pdf = os.path.join(OUTPUT_DIR, f"Fig2B_Diagnostic_Radar_{safe_window_name}.pdf")

        # 将图片稍微放大一点点以容纳外部结构
        fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # ==== 核心重构区：外部色块与文字完美嵌入 ====
        ring_inner = 1.0   # 色环的内半径
        ring_width = 0.13   # 色环的绝对厚度

        # 画外部底色色块
        ax.bar(
            angles[:-1],
            [ring_width]*num_vars,
            width=2*pi/num_vars,
            bottom=ring_inner,
            color=bg_colors,
            edgecolor='white',
            linewidth=2.5,
            alpha=0.7,
            zorder=0                        # 👈 放到底层
        )

        # 放弃使用会跑偏的 xticks，将其清空
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])

        # 使用几何绝对坐标将文本 "钉" 在彩色外环的正中心
        for angle, label in zip(angles[:-1], display_metrics):
            ax.text(angle, ring_inner + ring_width / 2, label,
                    size=16, fontweight='bold', color='#333333',
                    ha='center', va='center',
                    zorder=5)

        ax.set_rlabel_position(0)
        # 只保留 0.2 ~ 1.0 的内网格线
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#999999", size=16)

        # ✅ 将整个极坐标的显示范围放大到色环之外！防止被系统自动裁切
        plt.ylim(0, ring_inner + ring_width + 0.05)

        # ✅ 分离网格线控制：只让同心圆(yaxis)全局生效
        ax.yaxis.grid(True, color='#E5E5E5', linestyle='-', linewidth=1.5)
        # 关闭系统默认的放射状网格线（防止画出 1.0 边界刺破外部色块）
        ax.xaxis.grid(False)

        # ✅ 手动绘制精确控制在 0 到 1.0 范围内的放射线！
        for angle in angles[:-1]:
            ax.plot([angle, angle], [0, 1.0], color='#E5E5E5', linestyle='-', linewidth=1.5, zorder=1)

        # 隐藏系统最外侧的极坐标圆圈轮廓
        ax.spines['polar'].set_visible(False)

        # 手动在 r=1.0 的位置画一个硬朗的深色边框，作为雷达数据区的物理分界线
        theta_circle = np.linspace(0, 2*pi, 100)
        ax.plot(theta_circle, [1.0]*100, color='#333333', linewidth=1.5, zorder=1)
        # ===========================================

        has_data = False
        for model, method in configs:
            subset = df[(df['model'] == model) & (df['Method'] == method) & (df['Time window'] == window)]
            if subset.empty: continue

            has_data = True
            color, ls, lw, _ = get_color_and_style(model, method)
            label = f"{model} ({'Multi-agent' if 'Multi' in method else 'CoT'})"

            values = subset[metrics].values.flatten().tolist()
            values += values[:1]

            # 加入数据点 Marker (圆点) 并统一增加多边形填充
            ax.plot(angles, values, linewidth=lw, linestyle=ls, color=color, label=label,
                    marker='o', markersize=7, markerfacecolor=color, zorder=3)

            fill_alpha = 0.15 if ('DeepSeek-V3.2' in model and 'Multi' in method) else 0.05
            ax.fill(angles, values, color=color, alpha=fill_alpha, zorder=2)

        if not has_data:
            plt.close()
            continue

        # 因为极坐标系放大了，标题相应往上提一点避免重叠
        # plt.title(f'Diagnostic Profiling ({window})', size=16, color='#222222', fontweight='bold', y=1.10)
        # ax.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, -0.12),
        #     ncol=3,
        #     frameon=False,
        #     fontsize=10
        # )

        plt.tight_layout()
        plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"[Panel B] 成功生成: {output_pdf}")

# ==========================================
# 图表 C：C-index 全局生存排序效能对比条形图 (独立5柱排序)
# ==========================================
def plot_c_index_comparison(df):
    output_pdf = os.path.join(OUTPUT_DIR, "Fig2C_C_Index_Comparison.pdf")

    # 筛选包含 C index 的数据行
    df_c = df[df['C index'].notna()].copy()
    if df_c.empty:
        df_c = df[df['Time window'] == 'c_index'].copy()
        if df_c.empty: return

    # 构建统一的配置标签，并精简7B模型的名字
    def make_label(row):
        # 将 MA 修改为 Multi-agent
        method_short = 'Multi-agent' if 'Multi' in row['Method'] else 'CoT'
        model_name = '7B Baseline' if '7B' in row['model'] else row['model']
        return f"{model_name}\n({method_short})"

    df_c['Config_Label'] = df_c.apply(make_label, axis=1)

    # 按照 C-index 值从小到大排序
    df_c = df_c.sort_values(by='C index', ascending=True).reset_index(drop=True)

    # 动态匹配已排序数据的颜色
    bar_colors = []
    for _, row in df_c.iterrows():
        color, _, _, _ = get_color_and_style(row['model'], row['Method'])
        bar_colors.append(color)

    plt.figure(figsize=(10, 6.5))

    # 绘制6根独立的柱子，利用匹配好的调色板
    ax = sns.barplot(data=df_c, x='Config_Label', y='C index',
                     palette=bar_colors, edgecolor='black', linewidth=1.2)

    # 添加具体数值标签
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

    sns.despine()
    # ax.set_title('Global Concordance Index (C-index) Performance Ranking', fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('Model Configuration', fontsize=16)
    ax.set_ylabel('C-index Score', fontsize=16,)
    ax.set_ylim(0.3, 0.65) # 调整刻度，完美包容0.339到0.617
    ax.tick_params(axis='both', labelsize=16)
    # 只倾斜横轴标签 45 度，避免 6 个配置名挤在一起 (纵轴保持水平)
    ax.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300)
    plt.show()
    plt.close()
    print(f"[Panel C] 成功生成: {output_pdf}")

# ==========================================
# 图表 D：假阳性偏倚 (敏感性 vs 特异性) 权衡图 (按时间窗)
# ==========================================
def plot_sens_spec_tradeoff(df):
    target_windows = ['0-6 months', '0-24 months', '24-60 months','0-60 months']

    for window in target_windows:
        safe_window_name = window.replace(' ', '_').replace('-', '_')
        output_pdf = os.path.join(OUTPUT_DIR, f"Fig2D_Sens_Spec_Tradeoff_{safe_window_name}.pdf")

        df_trade = df[df['Time window'] == window].copy()
        if df_trade.empty: continue

        # 将 MA 修改为 Multi-agent
        df_trade['Config'] = df_trade['model'] + " \n(" + df_trade['Method'].apply(lambda x: 'Multi-agent' if 'Multi' in x else 'CoT') + ")"

        # 将 Sensitivity 和 Specificity 融合成长表
        df_melt = df_trade.melt(id_vars='Config', value_vars=['sensitivity', 'specificity'],
                                var_name='Metric', value_name='Score')
        df_melt['Metric'] = df_melt['Metric'].str.capitalize()

        plt.figure(figsize=(10, 6.5))
        ax = sns.barplot(data=df_melt, y='Config', x='Score', hue='Metric',
                         palette={'Sensitivity': COLOR_SENS, 'Specificity': COLOR_SPEC}, # 使用新配色
                         edgecolor='black', linewidth=1.2)

        for p in ax.patches:
            width = p.get_width()
            if not np.isnan(width) and width > 0.001:
                ax.annotate(f"{width:.3f}",
                            xy=(width, p.get_y() + p.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=16)

        sns.despine(top=True, right=True)
        # ax.set_title(f'Trade-off: Sensitivity vs. Specificity (Window: {window})', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Score (0.0 to 1.0)', fontsize=16)
        ax.set_ylabel('')
        ax.set_xlim(0, 1.15) # 留出空间给数值标签
        ax.tick_params(axis='both', labelsize=16)

        # 将图例移至图表右侧外部
        plt.legend(title='Clinical Metric', loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
        plt.tight_layout()
        # 添加 bbox_inches='tight' 防止图例被裁切
        plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"[Panel D] 成功生成: {output_pdf}")

# ==========================================
# 主执行入口
# ==========================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"找不到数据文件: {EXCEL_PATH}")
        print("请确保你的 Excel 文件存放在当前运行目录下。")
    else:
        try:
            df_results = pd.read_excel(EXCEL_PATH)

            # 新 Excel 的 'Time window' 列只在每个时间窗首行有值，其余为 NaN，
            # 需要前向填充，否则按时间窗过滤会丢掉大部分行。
            df_results['Time window'] = df_results['Time window'].ffill()

            # 执行图表生成流水线
            # 已经为您取消了注释，这样运行就可以一次性生成所有图片了
            plot_roc_temporal_decay(df_results)
            plot_radar_charts(df_results)
            plot_c_index_comparison(df_results)
            plot_sens_spec_tradeoff(df_results)

            print("\n全部四类图表渲染完成！图表保存在 %s 目录下。" % OUTPUT_DIR)
        except Exception as e:
            print(f"读取 Excel 文件或处理时失败，错误详情: {e}")


if __name__ in ("__main__", "__mp_main__"):
    main()
