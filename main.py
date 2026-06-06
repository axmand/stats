import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("国赛/成绩-国赛.xls", sheet_name="ALL", header=None)

def extreme_group_discrimination(score_rates, total_scores, ratio=0.27):
    """
    使用极端分组法计算区分度。
    参数:
        score_rates: pd.Series，每个学生的得分率（题目/大题得分 / 满分）
        total_scores: pd.Series，每个学生的总分（索引需与 score_rates 对齐）
        ratio: 高分组和低分组所占比例（默认0.27）
    返回:
        区分度 D = 高分组平均得分率 - 低分组平均得分率
    """
    # 合并有效数据
    valid = score_rates.notna() & total_scores.notna()
    if valid.sum() == 0:
        return np.nan
    total_valid = total_scores[valid]
    rates_valid = score_rates[valid]

    # 按总分降序排序
    sorted_idx = total_valid.sort_values(ascending=False).index
    n = len(sorted_idx)
    high_n = int(np.ceil(n * ratio))
    low_n = int(np.ceil(n * ratio))

    high_idx = sorted_idx[:high_n]
    low_idx = sorted_idx[-low_n:]

    high_rates = rates_valid.loc[high_idx].dropna()
    low_rates = rates_valid.loc[low_idx].dropna()

    if len(high_rates) == 0 or len(low_rates) == 0:
        return np.nan
    return high_rates.mean() - low_rates.mean()

# ==================== 以下函数均改用 extreme_group_discrimination ====================

def load_total_scores(col_label="总分"):
    """
    加载总分列数据
    """
    header = df.iloc[3].astype(str).str.strip()
    col_idx = header[header == col_label].index[0]
    scores = pd.to_numeric(df.iloc[4:, col_idx], errors='coerce').dropna()
    return scores

def stats_base(scorespd, excellent_line=90, pass_line=60):
    """计算总分统计指标（无区分度，不需修改）"""
    scores = scorespd.values
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    min_val = np.min(scores)
    max_val = np.max(scores)
    median = np.median(scores)
    full_range = max_val - min_val
    skew = stats.skew(scores)
    kurt = stats.kurtosis(scores)
    excellent_count = int(np.sum(scores >= excellent_line))
    excellent_rate = excellent_count / n * 100
    pass_count = int(np.sum(scores >= pass_line))
    pass_rate = pass_count / n * 100
    fail_count = int(np.sum(scores < pass_line))
    fail_rate = fail_count / n * 100
    shapiro_stat, shapiro_p = stats.shapiro(scores)
    is_normal = shapiro_p > 0.05
    r = {
        "样本量": n,
        "均值": round(mean, 2),
        "中位数": round(median, 2),
        "标准差": round(std, 2),
        "最小值": round(min_val, 2),
        "最大值": round(max_val, 2),
        "全距": round(full_range, 2),
        "偏度": round(skew, 3),
        "峰度(超额)": round(kurt, 3),
        "优秀线": excellent_line,
        "及格线": pass_line,
        "优秀人数": excellent_count,
        "优秀率": round(excellent_rate, 2),
        "及格人数": pass_count,
        "及格率": round(pass_rate, 2),
        "不及格人数": fail_count,
        "不及格率": round(fail_rate, 2),
        "Shapiro-Wilk W": round(shapiro_stat, 4),
        "Shapiro-Wilk p": round(shapiro_p, 4),
        "是否正态": is_normal
    }
    with pd.ExcelWriter('结果/基本统计.xlsx') as writer:
        pd.DataFrame([r]).to_excel(writer, index=False)
    return r

def stats_overall_section(perfect):
    """
    输出试卷整体、客观题(1-40)、主观题(41-52)的难度与区分度
    区分度改为极端分组法
    """
    type_row = df.iloc[2, 2:-1].astype(str).str.strip()
    qnum_row = df.iloc[3, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')

    # 每题满分
    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        question_marks.append(perfect[t] if t in perfect else scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    total_full = question_marks.sum()
    qnum_int = qnum_row.astype(int)

    def section_stats(start, end):
        mask = (qnum_int >= start) & (qnum_int <= end)
        cols = qnum_int[mask].index
        if mask.sum() == 0:
            return None
        full = question_marks[cols].sum()
        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / full
        avg_rate = score_rates.mean()
        # 极端分组法区分度
        disc = extreme_group_discrimination(score_rates, total_scores)
        return {
            '题目数': mask.sum(),
            '满分': full,
            '难度(平均得分率)': round(avg_rate, 3),
            '区分度': round(disc, 3) if not np.isnan(disc) else np.nan
        }

    # 客观题、主观题、整体
    obj = section_stats(1, 40)
    subj = section_stats(41, 52)
    overall = section_stats(1, len(qnum_int))

    rows = []
    # 整体（区分度用题目平均区分度代替）
    per_q = stats_per_question(perfect)   # 此时 per_question 已改为极端分组法
    avg_discrimination = per_q['区分度'].mean()
    rows.append({
        '部分': '整体',
        '题目数': overall['题目数'],
        '满分': overall['满分'],
        '难度(平均得分率)': overall['难度(平均得分率)'],
        '题目平均区分度': round(avg_discrimination, 3)
    })
    rows.append({
        '部分': '客观题(1-40)',
        '题目数': obj['题目数'],
        '满分': obj['满分'],
        '难度(平均得分率)': obj['难度(平均得分率)'],
        '区分度': obj['区分度']
    })
    rows.append({
        '部分': '主观题(41-52)',
        '题目数': subj['题目数'],
        '满分': subj['满分'],
        '难度(平均得分率)': subj['难度(平均得分率)'],
        '区分度': subj['区分度']
    })

    with pd.ExcelWriter('结果/试卷整体分析报告.xlsx') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='试卷整体分析', index=False)
    return pd.DataFrame(rows)

def stats_content_validity(perfect):
    """内容效度分析（无区分度，不需修改）"""
    outline_row = df.iloc[0, 2:-1].astype(str).str.strip()
    type_row    = df.iloc[2, 2:-1].astype(str).str.strip()
    scores_mat  = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')

    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        question_marks.append(perfect[t] if t in perfect else scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    total_questions = len(outline_row)
    unique_outlines = pd.unique(outline_row)
    results = []
    for outline in unique_outlines:
        cols = outline_row[outline_row == outline].index
        q_count = len(cols)
        proportion = q_count / total_questions
        outline_full = question_marks[cols].sum()
        stu_scores = scores_mat[cols].sum(axis=1)
        avg_rate = (stu_scores / outline_full).mean()
        results.append({
            '大纲类别': outline,
            '题目数': q_count,
            '占比': round(proportion, 3),
            '满分': outline_full,
            '平均得分率': round(avg_rate, 3)
        })
    with pd.ExcelWriter('结果/内容效度分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='内容效度分析', index=False)
    return pd.DataFrame(results)

def stats_construct_validity(perfect):
    """结构效度（无区分度计算，不需修改）"""
    level_row = df.iloc[1, 2:-1].astype(str).str.strip()
    type_row  = df.iloc[2, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')
    
    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        question_marks.append(perfect[t] if t in perfect else scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    levels = pd.unique(level_row)
    level_scores = {}
    for lvl in levels:
        cols = level_row[level_row == lvl].index
        level_full = question_marks[cols].sum()
        stu_score = scores_mat[cols].sum(axis=1)
        level_scores[lvl] = stu_score / level_full

    df_levels = pd.DataFrame(level_scores)
    df_levels['总分'] = total_scores.values
    corr_matrix = df_levels.corr()
    dims = list(level_scores.keys())
    dim_total_corr = corr_matrix.loc[dims, '总分']

    with pd.ExcelWriter('结果/结构效度分析报告.xlsx') as writer:
        corr_matrix.to_excel(writer, sheet_name='结构效度相关矩阵', index=False)
        dim_total_corr.to_excel(writer, sheet_name='结构效度', startrow=corr_matrix.shape[0] + 2)
    return corr_matrix, dim_total_corr

def stats_topic(perfect):
    """
    按题型分析，区分度改为极端分组法
    """
    types_row = df.iloc[2, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')
    unique_types = pd.unique(types_row)
   
    results = []
    for t in unique_types:
        cols = types_row[types_row == t].index
        q_count = len(cols)

        if t in perfect:
            type_full = q_count * perfect[t]
        else:
            stu_type_scores = scores_mat[cols].sum(axis=1)
            type_full = stu_type_scores.max()

        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / type_full
        avg_rate = score_rates.mean()
        disc = extreme_group_discrimination(score_rates, total_scores)

        results.append({
            '题型': t,
            '题目数': q_count,
            '满分': type_full,
            '平均得分': round(stu_scores.mean(), 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(disc, 3) if not np.isnan(disc) else np.nan
        })

    with pd.ExcelWriter('结果/题型分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='题型分析', index=False)
    return pd.DataFrame(results)

def stats_outline(perfect, total_scores):
    """
    分析大题纲要，区分度改为极端分组法
    """
    outline_row = df.iloc[0, 2:-1].astype(str).str.strip()
    type_row    = df.iloc[2, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        if t in perfect:
            question_marks.append(perfect[t])
        else:
            question_marks.append(scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)
    unique_outlines = pd.unique(outline_row)
    results = []
    for outline in unique_outlines:
        cols = outline_row[outline_row == outline].index
        q_count = len(cols)
        outline_full = question_marks[cols].sum()
        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / outline_full
        avg_rate = score_rates.mean()
        disc = extreme_group_discrimination(score_rates, total_scores)
        results.append({
            '大纲类别': outline,
            '题目数': q_count,
            '满分': outline_full,
            '平均得分': round(stu_scores.mean(), 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(disc, 3) if not np.isnan(disc) else np.nan
        })

    with pd.ExcelWriter('结果/大纲分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='大纲分析', index=False)
    return pd.DataFrame(results)

def stats_cognitive(perfect):
    """
    按认知层次分析，区分度改为极端分组法
    """
    level_row = df.iloc[1, 2:-1].astype(str).str.strip()
    type_row  = df.iloc[2, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')

    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        if t in perfect:
            question_marks.append(perfect[t])
        else:
            question_marks.append(scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    unique_levels = pd.unique(level_row)
    results = []
    for lvl in unique_levels:
        cols = level_row[level_row == lvl].index
        q_count = len(cols)
        level_full = question_marks[cols].sum()
        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / level_full
        avg_rate = score_rates.mean()
        disc = extreme_group_discrimination(score_rates, total_scores)

        results.append({
            '认知层次': lvl,
            '题目数': q_count,
            '满分': level_full,
            '平均得分': round(stu_scores.mean(), 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(disc, 3) if not np.isnan(disc) else np.nan
        })
    
    with pd.ExcelWriter('结果/认知层次分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='认知层次分析', index=False)
    return pd.DataFrame(results)

def stats_per_question(perfect):
    """
    逐题分析，区分度改为极端分组法
    """
    type_row = df.iloc[2, 2:-1].astype(str).str.strip()
    qnum_row = df.iloc[3, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')

    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        if t in perfect:
            question_marks.append(perfect[t])
        else:
            question_marks.append(scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    results = []
    for col in scores_mat.columns:
        q_num = qnum_row[col]         
        q_type = type_row[col]
        full = question_marks[col]
        stu_answers = scores_mat[col]  
        score_rates = stu_answers / full
        avg_rate = score_rates.mean()
        disc = extreme_group_discrimination(score_rates, total_scores)

        results.append({
            '题号': q_num,
            '题型': q_type,
            '满分': full,
            '平均得分': round(stu_answers.mean(), 3),
            "难度(平均得分率)":round(avg_rate, 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(disc, 3) if not np.isnan(disc) else np.nan
        })

    with pd.ExcelWriter('结果/逐题分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='逐题分析', index=False)
    return pd.DataFrame(results)

def stats_ctt(perfect, num_groups=5):
    """多组CTT分析（无区分度计算，保持不变）"""
    type_row = df.iloc[2, 2:-1].astype(str).str.strip()
    qnum_row = df.iloc[3, 2:-1].astype(str).str.strip()
    scores_mat = df.iloc[4:, 2:-1].apply(pd.to_numeric, errors='coerce')
    total_scores = pd.to_numeric(df.iloc[4:, -1], errors='coerce')

    question_marks = []
    for col in scores_mat.columns:
        t = type_row[col]
        question_marks.append(perfect[t] if t in perfect else scores_mat[col].max())
    question_marks = pd.Series(question_marks, index=scores_mat.columns)

    total_full = question_marks.sum()
    total_rate = total_scores / total_full

    try:
        groups = pd.qcut(total_rate, q=num_groups, labels=range(1, num_groups+1), duplicates='drop')
    except ValueError:
        groups = pd.cut(total_rate, bins=num_groups, labels=range(1, num_groups+1))

    group_avg_rate = total_rate.groupby(groups).mean()

    results = []
    for col in scores_mat.columns:
        q_num = qnum_row[col]
        q_type = type_row[col]
        full = question_marks[col]
        scores = scores_mat[col]
        for grp in sorted(groups.unique()):
            mask = groups == grp
            avg_item_score = scores[mask].mean()
            item_rate = avg_item_score / full
            results.append({
                '题号': q_num,
                '题型': q_type,
                '满分': full,
                '组别': grp,
                '组平均总分得分率': round(group_avg_rate[grp], 3),
                '题目得分率': round(item_rate, 3)
            })

    with pd.ExcelWriter('结果/逐题ctt分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='逐题ctt分析', index=False)
    return pd.DataFrame(results)
def plot_total_distribution(scores, base_stats):
    """
    绘制总分分布直方图，并叠加正态拟合曲线
    图中标注基本统计指标和正态性检验结果
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mu, sigma = base_stats['均值'], base_stats['标准差']
    ax.hist(scores, bins='auto', density=True, alpha=0.7, rwidth=0.8, color='#5B9BD5', edgecolor='white', label='分数分布')
    x = np.linspace(min(scores), max(scores), 300)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, 'r-', lw=2, label=f'正态拟合 (μ={mu:.1f}, σ={sigma:.1f})')
    text = (f"样本量={base_stats['样本量']}  均值={mu:.1f}  中位数={base_stats['中位数']}\n"
            f"标准差={sigma:.1f}  全距={base_stats['全距']}\n"
            f"优秀率(≥{base_stats['优秀线']})={base_stats['优秀率']}%  "
            f"及格率(≥{base_stats['及格线']})={base_stats['及格率']}%\n"
            f"Shapiro-Wilk p={base_stats['Shapiro-Wilk p']}  "
            f"{'正态' if base_stats['是否正态'] else '非正态'}")
    ax.text(0.98, 0.95, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=9)
    ax.set_title('总分分布直方图与正态拟合曲线')
    ax.set_xlabel('总分')
    ax.set_ylabel('概率密度')
    ax.legend()
    plt.tight_layout()
    plt.savefig('结果/总分分布.png', dpi=300)

def plot_grouped_bar(df, title, x_col, value_cols, save_name, order=None, bar_width=0.2, bar_gap=0.05, line_col="掌握程度(平均得分率)"):
    """
    绘制分组柱状图，可选添加右轴折线图。
    参数：
        df: DataFrame
        title: 图表标题
        x_col: x轴分类列名（如'题型'）
        value_cols: 需要绘制柱状图的列名列表（如['满分']）
        save_name: 保存文件名
        order: x轴分类的顺序
        bar_width: 每个柱子的宽度
        bar_gap: 柱子之间的间隙
        line_col: 需要绘制折线的列名（如'掌握程度(平均得分率)'），None 时不添加折线
    """
    if order is not None:
        # 按指定的顺序排序，只保留 order 中存在的类别
        valid_order = [o for o in order if o in df[x_col].values]
        df = df.set_index(x_col).loc[valid_order].reset_index()
    else:
        df = df.sort_values(by=value_cols[0], ascending=False)
    x = np.arange(len(df))
    n = len(value_cols)
    total_width = n * bar_width + (n - 1) * bar_gap
    start_offset = -total_width / 2 + bar_width / 2
    positions = []
    for i in range(n):
        offset = start_offset + i * (bar_width + bar_gap)
        positions.append(x + offset)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#5470c6', '#91cc75', '#fac858']
    for i, col in enumerate(value_cols):
        # 确保数据是一维数值数组
        y_data = df[col].values.ravel() if hasattr(df[col], 'values') else df[col]
        bars = ax.bar(positions[i], y_data, bar_width, label=col, color=colors[i % len(colors)])
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=45, ha='right')
    ax.set_ylabel('值', color='black')
    ax.set_title(title)
    ax.legend(loc='upper right')
    if line_col is not None and line_col in df.columns:
        ax2 = ax.twinx()
        line_data = df[line_col].values.ravel()
        ax2.plot(x, line_data, 'o-', color='#ee6666', linewidth=2, markersize=6, label=line_col)
        ax2.set_ylabel(line_col, color='#ee6666')
        ax2.tick_params(axis='y', labelcolor='#ee6666')
        # 添加折线标签
        for i, val in enumerate(line_data):
            ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', color='#ee6666', fontsize=8)
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.set_ylim(0, 1.05)  # 掌握程度范围 0~1
    max_bar = max(df[value_cols].max().max(), 1.0)
    ax.set_ylim(0, max_bar * 1.05)
    plt.tight_layout()
    plt.savefig("结果/" + save_name, dpi=300)
    plt.close()

def plot_per_question_scatter(per_question_stats):
    """
    绘制逐题难度-区分度散点图
    - 横轴：难度（平均得分率，0~1）
    - 纵轴：区分度
    - 点颜色按题型区分，标注题号
    """
    plt.figure(figsize=(10, 7))
    types = per_question_stats['题型'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    color_map = dict(zip(types, colors))
    for t in types:
        sub = per_question_stats[per_question_stats['题型'] == t]
        plt.scatter(sub['难度(平均得分率)'], sub['区分度'], c=[color_map[t]], label=t, s=60, edgecolors='white', zorder=3)
        for _, row in sub.iterrows():
            plt.annotate(row['题号'], (row['难度(平均得分率)'], row['区分度']), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='区分度 0.2（需修改）')
    plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='区分度 0.3（良好）')
    plt.axvline(x=0.3, color='gray', linestyle=':', alpha=0.4)
    plt.axvline(x=0.7, color='gray', linestyle=':', alpha=0.4)
    plt.fill_betweenx([-0.1, 1.2], 0.3, 0.7, color='green', alpha=0.05)
    plt.xlabel('难度 (平均得分率)')
    plt.ylabel('区分度 (Pearson r)')
    plt.title('试题难度-区分度分布图')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("结果/难度区分度散点图.png", dpi=300, bbox_inches='tight')

def plot_per_question_difficulty_pie(per_question_stats, thresholds=(0.2, 0.4, 0.6, 0.8), labels=('过难 (<0.2)', '较难 (0.2~0.4)', '适中 (0.4~0.6)', '较易 (0.6~0.8)', '过易 (>0.8)')):
    """
    根据每题难度（平均得分率）绘制饼图
    thresholds: (低分界, 高分界)，默认 0.3、0.7
    labels: 对应三个类别的名称
    """
    diff = per_question_stats['难度(平均得分率)']
    bins = [-1, thresholds[0], thresholds[1], thresholds[2], thresholds[3], 2]
    counts = [
        ((diff < thresholds[0]).sum()),
        ((diff >= thresholds[0]) & (diff < thresholds[1])).sum(),
        ((diff >= thresholds[1]) & (diff < thresholds[2])).sum(),
        ((diff >= thresholds[2]) & (diff < thresholds[3])).sum(),
        ((diff >= thresholds[3])).sum()
    ]
    
    colors = ['#ee6666', '#fac858', '#91cc75', '#5470c6', '#73c0de']

    filtered = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]

    if not filtered:
        return
    
    counts_f, labels_f, colors_f = zip(*filtered)
    explode = (0.02,) * len(counts_f)
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        counts_f, labels=None, autopct='%1.1f%%',
        startangle=140, colors=colors_f, explode=explode,
        pctdistance=0.6
    )
    for i, (label, count) in enumerate(zip(labels_f, counts_f)):
        texts[i].set_text(f'{label}\n({count}道)')
    
    plt.title('试卷难度分布（五级分类）', fontsize=14)
    plt.tight_layout()
    plt.savefig('结果/试卷难度分布饼图.png', dpi=300, bbox_inches='tight')

def plot_per_question(per_question_stats):
    """
    绘制逐题难度和区分度
    """
    fig, ax1 = plt.subplots(figsize=(16, 6))
    x = range(len(per_question_stats))
    ax1.plot(x, per_question_stats['掌握程度(平均得分率)'], 'o-', color='#5470c6', label='难度(得分率)')
    ax1.set_xlabel('题目序号')
    ax1.set_ylabel('难度 (得分率)', color='#5470c6')
    ax1.tick_params(axis='y', labelcolor='#5470c6')
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(x, per_question_stats['区分度'], 's--', color='#ee6666', label='区分度')
    ax2.set_ylabel('区分度', color='#ee6666')
    ax2.tick_params(axis='y', labelcolor='#ee6666')
    ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='区分度=0.2')
    plt.title('逐题难度与区分度分布')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig('结果/逐题分析.png', dpi=300)

def plot_per_question_difficulty_bar(per_question_stats, bin_width=0.1):
    """
    按固定间隔统计难度分布并绘制柱状图
    bin_width: 区间宽度，默认 0.1
    """
    difficulties = per_question_stats['难度(平均得分率)']
    min_val = np.floor(difficulties.min() * 10) / 10
    max_val = np.ceil(difficulties.max() * 10) / 10
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if bins[-1] < difficulties.max():
        bins = np.append(bins, bins[-1] + bin_width)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    binned = pd.cut(difficulties, bins=bins, right=False, include_lowest=True)
    counts = binned.value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(counts)), counts.values, color='#5470c6', edgecolor='white', alpha=0.85)
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')
    plt.xlabel('难度区间（得分率）')
    plt.ylabel('题目数量')
    plt.title('试卷难度分布（间隔0.1）')
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(count), ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("结果/难度分布柱状图_0.1间隔统计.png", dpi=300, bbox_inches='tight')

def plot_ctt_per_question(ctt_stats, bInterp = True):
    """
    绘制多组 CTT 曲线（组平均得分率 → 题目得分率）
    """
    if(bInterp):
        for qnum, qdata in ctt_stats.groupby('题号'):
            x_orig = qdata['组平均总分得分率'].values
            y_orig = qdata['题目得分率'].values
            if len(x_orig) >= 4:
                x_smooth = np.linspace(x_orig.min(), x_orig.max(), 100)
                spl = make_interp_spline(x_orig, y_orig, bc_type='natural')
                y_smooth = spl(x_smooth)
            else:
                x_smooth = x_orig
                y_smooth = y_orig
            plt.figure(figsize=(6, 4))
            plt.plot(x_smooth, y_smooth, '-', color='#5470c6', linewidth=2, alpha=0.7, label='拟合曲线')
            plt.scatter(x_orig, y_orig, color='#ee6666', s=40, zorder=5, label='实测点')
            plt.plot(x_orig, y_orig, 'o--', color='gray', linewidth=1, alpha=0.5)
            plt.title(f'题号 {qnum}  CTT 曲线（平滑拟合）')
            plt.xlabel('组平均总分得分率')
            plt.ylabel('题目得分率')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"结果/ctt_{qnum}.png", dpi=200)
            plt.close()
    else:
        types = ctt_stats['题型'].unique()
        cols = min(3, len(types))
        rows = (len(types) - 1) // cols + 1
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        for qnum, qdata in ctt_stats.groupby('题号'):
            plt.figure(figsize=(6, 4))
            x = qdata['组平均总分得分率']
            y = qdata['题目得分率']
            plt.plot(x, y, 'o-', color='#5470c6', linewidth=2, markersize=6)
            plt.title(f'题号 {qnum}  CTT 曲线')
            plt.xlabel('组平均总分得分率')
            plt.ylabel('题目得分率')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"结果/ctt_{qnum}.png", dpi=200)
            plt.close()

def plot_content_validity(content_df):
    """
    内容效度双轴图：大纲类别题目占比（柱状图）+ 平均得分率（折线）
    大纲类别按占比由低到高排序（可改为按平均得分率）
    """
    content_df = content_df.sort_values(by='占比', ascending=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(content_df))
    categories = content_df['大纲类别']
    proportion = content_df['占比']
    avg_rate = content_df['平均得分率']
    bars = ax1.bar(x, proportion, color='#5470c6', alpha=0.7, label='题目占比')
    ax1.set_ylabel('题目占比', color='#5470c6')
    ax1.tick_params(axis='y', labelcolor='#5470c6')
    ax1.set_ylim(0, max(proportion) * 1.2)

    for bar, prop in zip(bars, proportion):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prop:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax2 = ax1.twinx()
    ax2.plot(x, avg_rate, 'o-', color='#ee6666', linewidth=2, markersize=8, label='平均得分率')
    ax2.set_ylabel('平均得分率', color='#ee6666')
    ax2.tick_params(axis='y', labelcolor='#ee6666')
    ax2.set_ylim(0, 1.05)
    
    for i, rate in enumerate(avg_rate):
        ax2.text(i, rate + 0.02, f'{rate:.2f}', ha='center', va='bottom', color='#ee6666', fontsize=9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30)
    ax1.set_title('试卷内容效度：大纲类别覆盖与得分率', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.savefig("结果/内容效度.png", dpi=300, bbox_inches='tight')

def plot_construct_validity(corr_matrix, dim_total_corr):
    """
    结构效度可视化：
    1. 认知层次相关矩阵热力图
    2. 各维度与总分相关系数柱状图
    corr_matrix: construct_validity 返回的完整相关矩阵 DataFrame
    dim_total_corr: construct_validity 返回的维度-总分相关系数 Series
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True, linewidths=0.2, fmt='.2f', vmin=-1, vmax=1, cbar_kws={'shrink':0.8})
    plt.title('结构效度：认知层次与总分相关矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'结果/结构效度_相关矩阵.png', dpi=300, bbox_inches='tight')
    
    dims = [d for d in dim_total_corr.index if d != '总分']
    values = dim_total_corr[dims]
    
    plt.figure(figsize=(8, 5))
    colors = ['#91cc75' if v >= 0.3 else '#fac858' if v >= 0.2 else '#ee6666' for v in values]
    bars = plt.bar(dims, values, color=colors, edgecolor='white', width=0.4)
    plt.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='良好 (0.3)')
    plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='可接受 (0.2)')
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel('与总分的相关系数')
    plt.title('结构效度：各认知层次与总分相关', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'结果/结构效度_维度总分相关.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    scores = load_total_scores()
    base_stats = stats_base(scores)
    
    perfect = {
        "A1": 1, 
        "A2": 1, 
        "A3": 1,
        "A4": 1,
        "名词解释": 3,
        "简答题": 5,
        "案例分析题": 10
    }

    topic_stats = stats_topic(perfect)
    outline_stats = stats_outline(perfect, scores)
    cognitive_stats = stats_cognitive(perfect)
    per_question_stats = stats_per_question(perfect)
    ctt_27_stats = stats_ctt(perfect)
    construct_validity_stats = stats_construct_validity(perfect)
    content_validity_stats = stats_content_validity(perfect)
    overall_section_stats = stats_overall_section(perfect)

    plot_total_distribution(scores, base_stats)
    plot_grouped_bar(topic_stats, '题型与掌握度', '题型', ['满分','平均得分'], '题型分析.png', [ "A1", "A2", "A3", "A4","名词解释", "简答题", "案例分析题"])
    plot_grouped_bar(outline_stats, '大纲类别与掌握度', '大纲类别', ['满分', '平均得分'], '大纲分析.png')
    plot_grouped_bar(cognitive_stats, '认知层次与掌握度', '认知层次', ['满分', '平均得分'], '认知层次分析.png')
    plot_per_question(per_question_stats)
    plot_per_question_scatter(per_question_stats)
    plot_per_question_difficulty_pie(per_question_stats)
    plot_per_question_difficulty_bar(per_question_stats)
    plot_ctt_per_question(ctt_27_stats)
    plot_content_validity(content_validity_stats)
    plot_construct_validity(*construct_validity_stats)
