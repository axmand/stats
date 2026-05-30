import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


"""
读取国赛成绩表，分析总分数据的统计特征，并计算优秀率、及格率等指标
"""
df = pd.read_excel("国赛/成绩-国赛.xls", sheet_name="ALL", header=None)

def load_total_scores(col_label="总分"):
    """
    加载总分列数据
    """
    header = df.iloc[3].astype(str).str.strip()
    col_idx = header[header == col_label].index[0]
    scores = pd.to_numeric(df.iloc[4:, col_idx], errors='coerce').dropna()
    return scores.values

def stats_base(scores, excellent_line=90, pass_line=60):
    """
    计算总分统计指标，返回所有结果字典
    优秀、及格线自定义
    """
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)          # 样本标准差
    min_val = np.min(scores)
    max_val = np.max(scores)
    median = np.median(scores)
    full_range = max_val - min_val        # 全距
    skew = stats.skew(scores)
    kurt = stats.kurtosis(scores)         # 超额峰度
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

def stats_topic(perfect):
    """
    按题型计算难度、掌握程度、区分度
    需要给定题型字典, 提供 [题型: 题型满分] 的格式
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
            """
            警告：缺省写法，如果给定的题型字典不包含题中实际题型, 默认使用最高得分作为题型满分。
            """
            stu_type_scores = scores_mat[cols].sum(axis=1)
            type_full = stu_type_scores.max()

        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / type_full
        avg_rate = score_rates.mean() 
        valid = stu_scores.notna() & total_scores.notna()

        if valid.sum() > 1:
            corr = pearsonr(stu_scores[valid], total_scores[valid])[0]
        else:
            corr = np.nan

        results.append({
            '题型': t,
            '题目数': q_count,
            '满分': type_full,
            '难度(平均得分率)': round(avg_rate, 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(corr, 3)
        })

    with pd.ExcelWriter('结果/题型分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='题型分析', index=False)

    return pd.DataFrame(results)

def stats_outline(perfect):
    """
    分析大题纲要，计算每个大题的满分、平均得分率等指标
    需要给定题型字典, 提供 [题型: 题型满分] 的格式
    """
    outline_row = df.iloc[0, 2:-1].astype(str).str.strip()
    type_row    = df.iloc[2, 2:-1].astype(str).str.strip()
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
    unique_outlines = pd.unique(outline_row)
    results = []
    for outline in unique_outlines:
        cols = outline_row[outline_row == outline].index
        q_count = len(cols)
        outline_full = question_marks[cols].sum()
        stu_scores = scores_mat[cols].sum(axis=1)
        score_rates = stu_scores / outline_full
        avg_rate = score_rates.mean()
        valid = stu_scores.notna() & total_scores.notna()

        if valid.sum() > 1:
            corr, _ = pearsonr(stu_scores[valid], total_scores[valid])
        else:
            corr = np.nan

        results.append({
            '大纲类别': outline,
            '题目数': q_count,
            '满分': outline_full,
            '难度(平均得分率)': round(avg_rate, 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(corr, 3)
        })

        with pd.ExcelWriter('结果/大纲分析报告.xlsx') as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name='大纲分析', index=False)

    return pd.DataFrame(results)

def stats_cognitive(perfect):
    """
    按认知层次分类统计难度、掌握程度、区分度
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
        valid = stu_scores.notna() & total_scores.notna()
        corr = pearsonr(stu_scores[valid], total_scores[valid])[0] if valid.sum() > 1 else np.nan
        results.append({
            '认知层次': lvl,
            '题目数': q_count,
            '满分': level_full,
            '难度(平均得分率)': round(avg_rate, 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(corr, 3)
        })
    
    with pd.ExcelWriter('结果/认知层次分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='认知层次分析', index=False)

    return pd.DataFrame(results)

def stats_per_question(perfect):
    """
    逐题计算难度、掌握程度、区分度
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
        valid = stu_answers.notna() & total_scores.notna()

        if valid.sum() > 1:
            corr, _ = pearsonr(stu_answers[valid], total_scores[valid])
        else:
            corr = np.nan

        results.append({
            '题号': q_num,
            '题型': q_type,
            '满分': full,
            '难度(平均得分率)': round(avg_rate, 3),
            '掌握程度(平均得分率)': round(avg_rate, 3),
            '区分度': round(corr, 3)
        })

    with pd.ExcelWriter('结果/逐题分析报告.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='逐题分析', index=False)

    return pd.DataFrame(results)

def stats_ctt(perfect, num_groups=5):
    """
    多组 CTT（经典测量理论）分析
    将学生按总分得分率等分为 num_groups 组，输出每组每道题的得分率，
    并计算基于最高组和最低组的区分度（D值）和难度。
    返回:
        pd.DataFrame 长格式，包含字段：题号、题型、满分、组别、组平均得分率、题目得分率
    """
    # 提取数据
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
    total_rate = total_scores / total_full   # 试卷总分得分率

    # 按总分得分率等分为 num_groups 组（组号从 1 开始）
    try:
        groups = pd.qcut(total_rate, q=num_groups, labels=range(1, num_groups+1), duplicates='drop')
    except ValueError:
        # 如果无法等分，降级为等宽分组
        groups = pd.cut(total_rate, bins=num_groups, labels=range(1, num_groups+1))

    # 每组的平均总分得分率
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
    ax.hist(scores, bins='auto', density=True, alpha=0.7, color='#5B9BD5', edgecolor='white', label='分数分布')
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

def plot_grouped_bar(df, title, x_col, value_cols, save_name):
    """
    通用分组条形图：难度、区分度
    """
    df = df.sort_values(by=value_cols[0], ascending=False)
    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar([i - width/2 for i in x], df[value_cols[0]], width, label=value_cols[0], color='#5470c6')
    bars2 = ax.bar([i + width/2 for i in x], df[value_cols[1]], width, label=value_cols[1], color='#91cc75')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=45, ha='right')
    ax.set_ylabel('值')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("结果/" + save_name, dpi=300)

def plot_per_question_scatter(per_question_stats):
    """
    绘制逐题难度-区分度散点图
    - 横轴：难度（平均得分率，0~1）
    - 纵轴：区分度（皮尔逊相关系数）
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

def plot_per_question_difficulty_pie(per_question_stats, thresholds=(0.3, 0.7), labels=('难题 (<0.3)', '中等题 (0.3~0.7)', '容易题 (>0.7)')):
    """
    根据每题难度（平均得分率）绘制饼图
    thresholds: (低分界, 高分界)，默认 0.3、0.7
    labels: 对应三个类别的名称
    """
    diff = per_question_stats['难度(平均得分率)']
    hard = (diff < thresholds[0]).sum()
    medium = ((diff >= thresholds[0]) & (diff <= thresholds[1])).sum()
    easy = (diff > thresholds[1]).sum()
    counts = [hard, medium, easy]
    colors = ['#ee6666', '#fac858', '#91cc75']
    explode = (0.02, 0.02, 0.02)
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        counts, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode,
        pctdistance=0.6, labeldistance=1.1
    )

    for i, (label, count) in enumerate(zip(labels, counts)):
        texts[i].set_text(f'{label}\n({count}道)')

    plt.title('试卷难度分布（按题目难度分类）', fontsize=14)
    plt.tight_layout()
    plt.savefig("结果/试卷难度分布饼图", dpi=300, bbox_inches='tight')

def plot_per_question(per_question_stats):
    """
    """
    fig, ax1 = plt.subplots(figsize=(16, 6))
    x = range(len(per_question_stats))
    ax1.plot(x, per_question_stats['难度(平均得分率)'], 'o-', color='#5470c6', label='难度(得分率)')
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

def plot_ctt_per_question(ctt_stats):
    """
    绘制多组 CTT 曲线（组平均得分率 → 题目得分率）
    """
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

if(__name__ == "__main__"):
    scores = load_total_scores()
    base_stats = stats_base(scores)
    """ 
    题型类型与对应的分值，通用
    """
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
    outline_stats = stats_outline(perfect)
    cognitive_stats = stats_cognitive(perfect)
    per_question_stats = stats_per_question(perfect)
    ctt_27_stats = stats_ctt(perfect)
    plot_total_distribution(scores, base_stats)
    plot_grouped_bar(topic_stats, '题型难度与区分度', '题型',['难度(平均得分率)', '区分度'], '题型分析.png')
    plot_grouped_bar(outline_stats, '大纲类别难度与区分度', '大纲类别',['难度(平均得分率)', '区分度'], '大纲分析.png')
    plot_grouped_bar(cognitive_stats, '认知层次难度与区分度', '认知层次', ['难度(平均得分率)', '区分度'], '认知层次分析.png')
    plot_per_question(per_question_stats)
    plot_per_question_scatter(per_question_stats)
    plot_per_question_difficulty_pie(per_question_stats)
    plot_ctt_per_question(ctt_27_stats)
