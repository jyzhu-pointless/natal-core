"""
NATAL Quick Demo App
Run with: streamlit run demo_app.py
"""

import importlib
import sys
from typing import Dict, Any

# List of natal modules to reload to prevent isinstance errors in Streamlit's
# hot-reloading environment. This is a workaround for development and should
# not be needed in a production environment (e.g., FastAPI backend).
modules_to_reload = [
    "natal.type_def",
    "natal.helpers",
    "natal.numba_utils",
    "natal.genetic_structures",
    "natal.genetic_entities",
    "natal.genetic_patterns",
    "natal.index_registry",
    "natal.population_state",
    "natal.algorithms",
    "natal.simulation_kernels",
    "natal.population_config",
    "natal.modifiers",
    "natal.gamete_allele_conversion",
    "natal.zygote_allele_conversion",
    "natal.genetic_presets",
    "natal.population_builder",
    "natal.base_population",
    "natal.age_structured_population",
    "natal.discrete_generation_population",
    "natal.visualization",
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

# Explicitly clear the global caches from the reloaded modules.
# This prevents "duplicate key" errors when re-creating structures
# in Streamlit's hot-reloading environment.
from natal.genetic_structures import GeneticStructure
from natal.genetic_entities import GeneticEntity

GeneticStructure.clear_all_caches()
GeneticEntity.clear_all_caches()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from natal.genetic_structures import Species
from natal.index_registry import decompress_hg_glab
from natal.population_builder import AgeStructuredPopulationBuilder
from natal.genetic_presets import HomingDrive
from natal.visualization import render_cell_svg

# --- 页面基础设置 ---
st.set_page_config(page_title="NATAL Population Demo", layout="wide")

st.title("🧬 NATAL 基因驱动模拟演示")
st.markdown("这是一个使用 `natal` 核心库构建的快速演示，用于展示基因驱动在种群中的传播过程。")

# --- 侧边栏：模拟参数配置 ---
st.sidebar.header("🕹️ 模拟参数 (Simulation Parameters)")

with st.sidebar.expander("基础设置 (Basic)", expanded=True):
    n_steps = st.slider("模拟步数 (Steps)", 10, 200, 100, help="模拟的总时间步长。")
    carrying_capacity = st.number_input("环境承载量 (Carrying Capacity)", 100, 10000, 1000, help="环境能支持的最大幼虫数量。")

with st.sidebar.expander("基因驱动设置 (Gene Drive)", expanded=True):
    drive_eff = st.slider("驱动转化率 (Homing Rate)", 0.0, 1.0, 0.95, help="在杂合子中，野生型等位基因成功转化为驱动等位基因的概率。")
    resistance_rate = st.slider("抗性产生率 (Resistance Rate)", 0.0, 0.5, 0.01, help="在驱动转化失败时，野生型等位基因突变为抗性等位基因的概率。")
    emb_res_rate = st.slider("胚胎抗性率 (Embryo Resistance)", 0.0, 1.0, 0.0, help="受精后在合子中发生抗性突变的概率（通常由母本沉积的 Cas9 引起）。这会导致受精结果不再是 One-hot。")

with st.sidebar.expander("适应度代价 (Fitness Costs)", expanded=True):
    # Viability
    st.markdown("**生存适应度 (Viability)**")
    via_cost = st.slider("生存代价 (Cost)", 0.0, 1.0, 0.1, key="v_cost", help="降低携带者的生存率 (1-cost)")
    via_mode = st.selectbox("计算模式 (Mode)", ["multiplicative", "dominant", "recessive"], index=0, key="v_mode")
    via_sex = st.selectbox("作用性别 (Sex)", ["Both", "Female", "Male"], index=0, key="v_sex")

    st.divider()

    # Fecundity
    st.markdown("**繁殖适应度 (Fecundity)**")
    fec_cost = st.slider("繁殖代价 (Cost)", 0.0, 1.0, 0.0, key="f_cost", help="降低携带者的繁殖力 (1-cost)")
    fec_mode = st.selectbox("计算模式 (Mode)", ["multiplicative", "dominant", "recessive"], index=0, key="f_mode")
    fec_sex = st.selectbox("作用性别 (Sex)", ["Both", "Female", "Male"], index=0, key="f_sex")

with st.sidebar.expander("补偿系统 (Rescue System)", expanded=True):
    use_rescue = st.checkbox("启用适应度补偿 (Enable Rescue)", value=False, help="使用一个独立的 Rescue 基因来完全补偿驱动带来的适应度代价。")

with st.sidebar.expander("初始状态 (Initial State)", expanded=True):
    init_wt_val = st.number_input("初始野生型雌性 (Initial WT Females)", 0, 2000, 500, help="模拟开始时种群中每个成年年龄段的野生型（WT/WT）雌性数量。")
    init_drive_val = st.number_input("初始释放驱动雄性 (Release Drive Males)", 0, 2000, 50, help="在模拟开始时释放的每个成年年龄段的纯合子驱动（Drive/Drive）雄性数量。")

# --- 1. 定义物种和遗传预设 ---
# 注意：在开发模式下启用模块重载时，不要使用 @st.cache_resource 缓存物种对象。
# 否则缓存中的旧类实例会导致 "must be a Chromosome instance" 类型检查错误。
def get_species():
    # 定义一个简单的单位点（single-locus）遗传架构
    return Species.from_dict("Mosquito", { # 如果启用 Rescue，则为双位点
        "chr1": {"Loc1": ["WT", "Drive", "Resistance"]},
        "chr2": {"Loc2": ["Rescue", "WT_rescue"]}
    })

species = get_species()

def get_scaling_config(cost, sex_selection):
    val = 1.0 - cost
    if sex_selection == "Female":
        return {"female": val}
    elif sex_selection == "Male":
        return {"male": val}
    return val

class RescueDrive(HomingDrive):
    """
    一个继承自 HomingDrive 的特殊预设，它重写了 fitness_patch 方法
    来实现一个多基因座的 Rescue 系统。
    """
    def __init__(self, rescue_allele: str, **kwargs):
        self._str_rescue_allele = self._resolve_allele_name(rescue_allele)
        super().__init__(**kwargs)

    def fitness_patch(self) -> Dict[str, Any]:
        """
        重写适应度补丁，使用基因型模式而不是等位基因拷贝计数。
        """
        # 定义一个包含所有会产生代价的等位基因的集合
        costly_alleles = [self._str_drive_allele]
        if self._str_resistance_allele:
            costly_alleles.append(self._str_resistance_allele)
        
        # 模式1: 匹配携带代价等位基因，且 *不* 携带 Rescue 等位基因的个体
        # Drive 部分: {Drive,Resistance}::* (至少有一条染色体上有 Drive 或 Resistance)
        drive_locus_pattern = f"{{{','.join(costly_alleles)}}}::*"
        
        # Rescue 部分: !Rescue::!Rescue (两条染色体都必须 *不是* Rescue，即 Rescue 缺失)
        rescue_locus_pattern = f"!{self._str_rescue_allele}::!{self._str_rescue_allele}"
        drive_without_rescue_pattern = f"{drive_locus_pattern}; {rescue_locus_pattern}"

        # 将适应度代价应用到这个特定的基因型模式上
        patch = {
            'viability': {
                drive_without_rescue_pattern: self.viability_scaling,
            },
            'fecundity': {
                drive_without_rescue_pattern: self.fecundity_scaling,
            }
        }
        return patch


# --- 根据 UI 选择创建不同的 Preset ---
if use_rescue:
    st.sidebar.info("Rescue 系统已启用。适应度代价仅在无 Rescue 基因时生效。")
    drive_preset = RescueDrive(
        name="RescueDrive",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="Resistance",
        rescue_allele="Rescue", # 新增
        drive_conversion_rate=drive_eff,
        embryo_resistance_formation_rate=emb_res_rate,
        late_germline_resistance_formation_rate=resistance_rate,
        viability_scaling=get_scaling_config(via_cost, via_sex), # 这些值现在被用于模式匹配
        fecundity_scaling=get_scaling_config(fec_cost, fec_sex),
    )
else:
    drive_preset = HomingDrive(
        name="HomingDrive",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="Resistance",
        drive_conversion_rate=drive_eff,
        embryo_resistance_formation_rate=emb_res_rate,
        late_germline_resistance_formation_rate=resistance_rate,
        viability_scaling=get_scaling_config(via_cost, via_sex),
        viability_mode=via_mode,
        fecundity_scaling=get_scaling_config(fec_cost, fec_sex),
        fecundity_mode=fec_mode
    )

# 定义年龄结构参数
n_ages = 2
new_adult_age = 1

# 构造初始状态数组 (年龄 0 为 0，成年年龄段为输入值)
init_wt = [0.0] * new_adult_age + [float(init_wt_val)] * (n_ages - new_adult_age)
init_drive = [0.0] * new_adult_age + [float(init_drive_val)] * (n_ages - new_adult_age)

# --- 2. 使用 Builder 构建种群 ---
# AgeStructuredPopulationBuilder 提供了一个链式 API 来流畅地配置种群
builder = (
    AgeStructuredPopulationBuilder(species)
    .setup(name="DemoPop", stochastic=True) # 使用随机模型以模拟真实世界的波动
    .age_structure(n_ages=n_ages, new_adult_age=new_adult_age) # 简化的年龄结构：0=幼虫, 1-3=成虫
    .initial_state({
        "female": {"WT|WT; WT_rescue|WT_rescue": init_wt},
        "male": {"Drive|WT; WT_rescue|WT_rescue": init_drive}
    })
    .survival(female_age_based_survival_rates=[1,0], male_age_based_survival_rates=[1,0])
    .competition(old_juvenile_carrying_capacity=carrying_capacity, juvenile_growth_mode="concave") # 设置密度依赖的幼虫竞争
    .reproduction(eggs_per_female=50)
    .presets(drive_preset) # 应用上面定义的基因驱动预设
)

pop = builder.build()

# --- Visualization Helpers ---
# --- 3. “语义化”配置查看器 ---
# 这个部分将 natal 内部的数值化配置（Numpy 张量）转换为人类可读的表格
with st.expander("📊 查看种群配置 (Config Visualization)"):
    config = pop.export_config()
    registry = pop.registry
    genotypes = registry.index_to_genotype
    
    # --- 新增：基因型可视化图示 ---
    st.markdown("### 🧬 基因型图示 (Genotype Legend)")
    st.caption("图示说明：圆圈代表细胞，内部条状物代表染色体。蓝色=WT, 红色=Drive, 黄色=Resistance, 绿色=Rescue。")
    
    # 使用 Grid 布局展示所有基因型
    n_cols = 4
    cols = st.columns(n_cols)
    target_age_viz = max(0, pop.new_adult_age - 1)
    
    for i, g_obj in enumerate(genotypes):
        g_idx = i
        with cols[i % n_cols]:
            # 渲染 SVG
            svg_html = render_cell_svg(g_obj, species, size=80)
            st.markdown(svg_html, unsafe_allow_html=True)
            st.markdown(f"**{str(g_obj)}**")
            
            # --- 显示 Fitness (仅当非 1.0 时) ---
            v_f = config.viability_fitness[0, target_age_viz, g_idx]
            v_m = config.viability_fitness[1, target_age_viz, g_idx]
            f_f = config.fecundity_fitness[0, g_idx]
            f_m = config.fecundity_fitness[1, g_idx]
            
            if v_f != 1.0 or v_m != 1.0:
                st.markdown(f"❤️ Via: `{v_f:.2g}`(F) / `{v_m:.2g}`(M)")
            if f_f != 1.0 or f_m != 1.0:
                st.markdown(f"🥚 Fec: `{f_f:.2g}`(F) / `{f_m:.2g}`(M)")

            # --- 显示配子分布 (Popover) ---
            with st.popover("🧬 查看配子 (Gametes)"):
                for sex_id, sex_name in enumerate(["Female", "Male"]):
                    st.markdown(f"**{sex_name} Produces:**")
                    probs = config.genotype_to_gametes_map[sex_id, g_idx]
                    valid_indices = np.nonzero(probs > 1e-6)[0]
                    for flat_idx in valid_indices:
                        p = probs[flat_idx]
                        hg_idx, glab_idx = decompress_hg_glab(flat_idx, config.n_glabs)
                        hg_name = str(registry.index_to_haplo[hg_idx])
                        glab_suffix = f" [{registry.index_to_glab[glab_idx]}]" if config.n_glabs > 1 else ""
                        st.text(f"- {hg_name}{glab_suffix}: {p:.1%}")

    st.markdown("### 适应度配置 (Viability Fitness)")
    st.caption("展示成年个体的生存适应度 (Female/Male)。仅显示非默认值或与驱动相关的基因型。")
    
    viability_data = []
    target_age = max(0, pop.new_adult_age - 1)
    
    # 遍历所有基因型，提取其适应度值
    for g_idx, g_obj in enumerate(genotypes):
        f_val = config.viability_fitness[0, target_age, g_idx] # 雌性
        m_val = config.viability_fitness[1, target_age, g_idx] # 雄性
        
        # 简单的过滤逻辑：只显示被修改过或与驱动相关的基因型
        if f_val != 1.0 or m_val != 1.0 or "Drive" in str(g_obj):
            viability_data.append({
                "Genotype": str(g_obj),
                "Female Viability": f_val,
                "Male Viability": m_val
            })
    
    if viability_data:
        st.dataframe(pd.DataFrame(viability_data), use_container_width=True)
    else:
        st.info("所有基因型适应度均为 1.0 (默认值)")

    st.markdown("### 繁殖适应度配置 (Fecundity Fitness)")
    st.caption("展示个体的繁殖能力 (Female/Male)。仅显示非默认值或与驱动相关的基因型。")
    
    fecundity_data = []
    
    for g_idx, g_obj in enumerate(genotypes):
        f_val = config.fecundity_fitness[0, g_idx]
        m_val = config.fecundity_fitness[1, g_idx]
        
        if f_val != 1.0 or m_val != 1.0 or "Drive" in str(g_obj):
            fecundity_data.append({
                "Genotype": str(g_obj),
                "Female Fecundity": f_val,
                "Male Fecundity": m_val
            })
            
    if fecundity_data:
        st.dataframe(pd.DataFrame(fecundity_data), use_container_width=True)
    else:
        st.info("所有基因型繁殖适应度均为 1.0 (默认值)")

    st.divider()
    st.markdown("### 遗传规则 (Genetic Rules)")
    st.caption("可视化种群的遗传传递规则，包含减数分裂（生成配子）和受精（生成合子）。")

    # 为两个标签页准备通用的轴标签
    g2g = config.genotype_to_gametes_map
    n_glabs = config.n_glabs
    n_hg = config.n_haploid_genotypes
    
    row_labels = [str(g) for g in genotypes]
    col_labels = []
    for hg_idx in range(n_hg):
        hg_obj = registry.index_to_haplo[hg_idx]
        for glab_idx in range(n_glabs):
            label = str(hg_obj)
            if n_glabs > 1:
                label += f" [{registry.index_to_glab[glab_idx]}]"
            col_labels.append(label)

    tab_meiosis, tab_fert = st.tabs(["🧬 减数分裂 (Meiosis)", "💕 受精 (Fertilization)"])

    with tab_meiosis:
        st.markdown("**二倍体基因型 → 单倍体配子 (Genotype to Gametes)**")
        st.caption("热力图展示了不同基因型产生各类配子的概率。颜色越亮代表概率越高。")

        # 绘制雌雄两性热力图
        col_f, col_m = st.columns(2)
        
        for sex_idx in range(config.n_sexes):
            sex_label = "Female" if sex_idx == 0 else "Male"
            matrix = g2g[sex_idx]  # Shape: (n_genotypes, n_gametes)
            
            fig = px.imshow(matrix,
                            labels=dict(x="配子 (Gamete)", y="亲本基因型 (Parent)", color="概率 (Prob)"),
                            x=col_labels,
                            y=row_labels,
                            color_continuous_scale="Viridis",
                            aspect="auto",
                            title=f"{sex_label} Meiosis")
            fig.update_layout(xaxis_tickangle=-45)
            
            with (col_f if sex_idx == 0 else col_m):
                st.plotly_chart(fig, use_container_width=True)

    with tab_fert:
        st.markdown("**配子结合 → 二倍体合子 (Gametes to Zygote)**")
        st.caption("热力图展示了不同配子（母本 vs 父本）结合产生的后代基因型。")

        g2z = config.gametes_to_zygote_map
        n_hg_glabs = n_hg * n_glabs
        
        # 如果矩阵太大，热力图会无法阅读且性能低下，回退到表格
        if n_hg_glabs > 40:
            st.warning(f"配子组合过多 ({n_hg_glabs}x{n_hg_glabs})，热力图渲染会非常缓慢。将以表格形式展示。")
            # 使用 np.nonzero 提取非零项 (稀疏矩阵)
            indices = np.nonzero(g2z > 1e-9)
            
            fert_rows = []
            # 限制显示数量以防卡顿
            max_rows = 2000
            count = 0
            
            for i in range(len(indices[0])):
                if count >= max_rows:
                    st.warning(f"条目过多，仅显示前 {max_rows} 条。")
                    break
                    
                mat_flat, pat_flat, zyg_idx = indices[0][i], indices[1][i], indices[2][i]
                prob = float(g2z[mat_flat, pat_flat, zyg_idx])
                
                m_hg_idx, m_glab_idx = decompress_hg_glab(mat_flat, n_glabs)
                p_hg_idx, p_glab_idx = decompress_hg_glab(pat_flat, n_glabs)
                
                m_str = str(registry.index_to_haplo[m_hg_idx]) + (f" [{registry.index_to_glab[m_glab_idx]}]" if n_glabs > 1 else "")
                p_str = str(registry.index_to_haplo[p_hg_idx]) + (f" [{registry.index_to_glab[p_glab_idx]}]" if n_glabs > 1 else "")
                
                fert_rows.append({
                    "Maternal Gamete": m_str, "Paternal Gamete": p_str,
                    "Zygote Genotype": str(genotypes[zyg_idx]), "Probability": prob
                })
                count += 1
            st.dataframe(pd.DataFrame(fert_rows), use_container_width=True)
        else:
            # 绘制热力图
            # 1. 找到每个配子对概率最大的合子索引作为颜色映射的基础
            zygote_indices = np.argmax(g2z, axis=2)
            # 2. 标记有效（非零）的配子对
            valid_pairs_mask = np.any(g2z > 1e-9, axis=2)
            # 3. 检查是否为混合结果（非 One-hot）：即该配子对产生的合子中，非零概率的个数 > 1
            is_mixed = np.sum(g2z > 1e-9, axis=2) > 1
            
            display_matrix = np.full((n_hg_glabs, n_hg_glabs), -1, dtype=int)
            display_matrix[valid_pairs_mask] = zygote_indices[valid_pairs_mask]

            text_annotations = np.full(display_matrix.shape, "", dtype=object)
            
            # 构造显示文本
            for r in range(n_hg_glabs):
                for c in range(n_hg_glabs):
                    if not valid_pairs_mask[r, c]:
                        continue
                        
                    if is_mixed[r, c]:
                        # 如果是混合结果，显示 Mixed 和主要概率
                        primary_idx = zygote_indices[r, c]
                        primary_prob = g2z[r, c, primary_idx]
                        text_annotations[r, c] = f"Mixed<br>({primary_prob:.2f})"
                    else:
                        # 如果是确定性结果 (One-hot)，直接显示基因型
                        text_annotations[r, c] = str(genotypes[display_matrix[r, c]])
            
            # 提示信息
            if np.any(is_mixed):
                st.info("检测到非确定性受精结果（Mixed）。这通常是由于胚胎抗性（Embryo Resistance）导致的，即同一对配子可能产生多种不同的合子。")

            fig = px.imshow(display_matrix,
                            labels=dict(x="父本配子 (Paternal)", y="母本配子 (Maternal)", color="合子索引 (Zygote Idx)"),
                            x=col_labels, y=col_labels,
                            color_continuous_scale="Plasma", aspect="auto", title="Fertilization Matrix")
            fig.update_traces(text=text_annotations, texttemplate="%{text}")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# --- 4. 运行模拟 ---
if st.button("▶️ 开始模拟 (Run Simulation)", type="primary"):
    
    history_data = []
    
    # 创建进度条和状态文本
    progress_bar = st.progress(0, text="准备开始...")
    status_text = st.empty()
    
    for t in range(n_steps + 1):
        # 1. 收集当前状态的统计数据
        stats = {
            "Tick": pop.tick,
            "Total Population": pop.get_total_count(),
            "Females": pop.get_female_count(),
            "Males": pop.get_male_count()
        }
        
        # 2. 计算等位基因频率
        freqs = pop.compute_allele_frequencies()
        for allele, freq in freqs.items():
            stats[f"Allele: {allele}"] = freq
            
        history_data.append(stats)
        
        # 3. 执行一步演化 (如果不是最后一步)
        if t < n_steps:
            pop.run_tick()
            progress_bar.progress((t + 1) / n_steps, text=f"正在模拟第 {t+1}/{n_steps} 步...")
    
    status_text.success("模拟完成！")
    
    # 将历史数据转换为 DataFrame 以便绘图
    df = pd.DataFrame(history_data)
    
    # --- 5. 结果可视化 ---
    st.subheader("📈 模拟结果")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 种群动态 (Population Dynamics)")
        fig_pop = px.line(df, x="Tick", y=["Total Population", "Females", "Males"], 
                          title="种群数量随时间变化")
        st.plotly_chart(fig_pop, use_container_width=True)
        
    with col2:
        st.markdown("#### 等位基因频率 (Allele Frequencies)")
        allele_cols = [c for c in df.columns if c.startswith("Allele: ")]
        fig_freq = px.line(df, x="Tick", y=allele_cols,
                           title="等位基因频率随时间变化",
                           range_y=[-0.05, 1.05]) # 稍微放宽Y轴以便看清0和1
        st.plotly_chart(fig_freq, use_container_width=True)

    # 提供数据下载功能
    st.download_button(
        label="📥 下载模拟数据 (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='simulation_results.csv',
        mime='text/csv',
    )
