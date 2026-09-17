# 遗传预设（Genetic Presets）

`Genetic Presets`（遗传预设）是 NATAL 框架中用于定义可重用遗传修饰的机制，支持基因驱动、突变系统和其他遗传修饰的快速配置。

## 概述

**遗传预设**（Genetic Presets）提供了一种标准化的方式来定义遗传修饰规则，包括：

- 修改配子生成规则（如基因驱动的过度分离）
- 改变合子发育过程（如胚胎抗性形成）
- 调整适应度参数（如驱动等位基因的成本）

## 应用预设

```python
import natal as nt

pop = (nt.DiscreteGenerationPopulation.setup(species, name="TestPop")
       .presets(preset1, preset2)  # 可以应用多个预设
       .build())
```

## 内置预设

### CytoplasmicPreset 与 Wolbachia：母系标签遗传

`CytoplasmicPreset` 的配子标记和合子标签遗传都使用转换规则。仅限关键字的参数
`default_glab` 和 `default_slab` 显式指定允许转换的来源配子标签和体细胞标签，
两者默认都为名字 `default`。编译遗传规则时，这些名字必须存在于物种对应的标签列表中。

对于带有映射中体细胞标签的雌性，只有仍带默认配子标签的配子会获得对应的母系标记。
受精时，该标记将仍带默认体细胞标签的后代转到映射中的标签，保持基因型不变。
已有的其他标签不会被覆盖。

这些规则接着处理前面 modifier 留下的分布。因此，前面 modifier 对基因型的修改会保留，
已经获得非默认体细胞标签的后代不会再被重新标记；modifier 的执行顺序会影响结果。
`Wolbachia` 使用这一机制实现感染状态的遗传。它的 `default_glab` 指定来源配子标签
（默认 `default`），已有的 `normal_slab` 也用于指定来源体细胞标签（默认 `normal`）。
使用自定义名字时需显式传入；例如，物种的未感染体细胞标签叫 `default` 时，
需传入 `normal_slab="default"`。

这些 preset 参数不会重排标签，也不会改变物种基础遗传表仍向各列表第一项分配概率的行为。
如果指定的来源标签不是第一项，规则只处理已经带有该标签的分支，例如前面的 modifier
已经将它们标记为该标签的情况。

#### 可选的不相容代价

`Wolbachia` 假设单菌株、完全母系传播，以及感染母本提供完全救援。
`incompatibility_cost` 默认 `0`，保留原有母系遗传行为且不需要额外标签。
正代价会把未感染母本与感染父本产生的子代标为 `incompatibility_slab`
（默认 `incompatible`），再将这些个体自身所选的 fitness 乘以
`1 - incompatibility_cost`。代价必须为有限值且在 `[0, 1]` 内。

| `incompatibility_effect` | 对标记个体的作用 |
|---|---|
| `"zygote_viability"`（默认） | 幼体竞争之前的胚胎存活 |
| `"viability"` | 最后一个幼体年龄的普通存活适合度（`new_adult_age - 1`；离散世代中为年龄 0） |
| `"fecundity"` | 该个体自身繁殖时的繁殖输出，不减少它的出生数量或存活率 |

标量代价作用于两性。选择 `fecundity` 时，一个 CI 亲本贡献一个乘数，两个 CI 亲本
贡献其平方，沿用框架现有的双亲生育力规则；它不减少最初那次不相容交配的产卵量。
`viability_scaling` 和 `fecundity_scaling` 仍独立作用于感染个体，且必须为有限非负值。

使用正代价时，需额外声明体细胞标签 `incompatible` 和配子标签 `wolbachia_ci`，
或用 `incompatibility_slab` 和 `paternal_glab` 指定自定义名字。
后者表示感染雄虫的配子具有 CI 诱导效应，不代表父系传播感染。
来源配子与体细胞标签仍须对应 `default_glab` 和 `normal_slab`，其他 modifier
已赋予的标签不会被覆盖。存活个体保留来源标记，但不会遗传该标记：
CI 母本仍未感染、不能救援，与感染父本交配可再次产生 CI 子代；相容交配产生正常标签子代。
仅选择 `*@normal` 的显式 fitness 配置不包含独立的 `*@incompatible` 群体；
若两组还需承担相同的其他代价，应同时指定两组或使用更宽的选择器。

将代价改为零会关闭 CI 标记规则及相应 fitness patch，但不会重标现有个体。
运行时重配置沿用固定布局规则，不能引入构建时已被压缩删除的子代类型。
可运行示例见 [Modifier 机制 5.2 节](3_modifiers.md#52-细胞质不兼容)。

### HomingDrive - 同源重组基因驱动

`HomingDrive` 实现 CRISPR/Cas9 类型的同源重组基因驱动：

```python
from natal.frontend.presets import HomingDrive

# 创建基本的基因驱动
drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,  # 95%的转换效率
    late_germline_resistance_formation_rate=0.03  # 3%形成抗性
)

# 应用到种群
population.apply_preset(drive)
```

#### 高级配置

```python
import natal as nt
from natal.frontend.presets import HomingDrive

species = nt.Species.from_dict(
    name="DepositionExample",
    structure={"chr1": {"drive": ["WT", "Drive", "Resistance", "FunctionalResistance"]}},
    gamete_labels=["default", "Cas9_deposited"],
)

# 性别特异性参数
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    functional_resistance_allele="FunctionalResistance",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # 性别差异
    late_germline_resistance_formation_rate=(0.02, 0.04),  # 元组形式 (female, male)
    embryo_resistance_formation_rate=0.01,
    cas9_deposition_glab="Cas9_deposited",
    functional_resistance_ratio=0.2,  # 20%的抗性等位基因是功能性的

    # 适应度成本
    viability_scaling=0.9,      # 10%生存力成本
    fecundity_scaling=0.95,     # 5%繁殖力成本
    sexual_selection_scaling=0.85  # 15%性选择劣势
)
population = (
    nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
    .initial_state({"female": {"WT|Drive": 100}, "male": {"WT|WT": 100}})
    .presets(drive)
    .build()
)
population.run(1)
```

胚胎抗性仅由亲本 Cas9 沉积触发。必须在物种的 `gamete_labels` 中注册
`cas9_deposition_glab` 指定的标签；未配置标签时，即使胚胎继承了 drive 或 Cas9，
也不发生胚胎编辑。携带者母本会给所有输出配子加标签，因此未继承 drive 的胚胎
也可以被编辑。对于 split drive，发生沉积的亲本必须同时携带 drive 和 Cas9。

胚胎抗性率的 `female` 和 `male` 分别表示母源和父源，不表示子代性别。
标量会同时设置两个率，但父源仅在 `use_paternal_deposition=True` 时启用。
因此，上例仅由母源沉积对每个剩余目标拷贝施加 1% 的编辑率。
两个已启用来源同时存在时，按顺序作用于剩余目标拷贝，总转换概率为
`1 - (1 - e_m) * (1 - e_p)`。

### ToxinAntidoteDrive - 毒素-解毒剂驱动（TARE/TADE）

`ToxinAntidoteDrive` 用于建模"驱动等位基因触发目标位点破坏，破坏等位基因产生适应度损失，而驱动等位基因提供救援"的系统。

```python
from natal.frontend.presets import ToxinAntidoteDrive

ta_drive = ToxinAntidoteDrive(
    name="TARE_Drive",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    conversion_rate=0.95,
    embryo_disruption_rate={"female": 0.30, "male": 0.0},
    viability_scaling=0.0,
    fecundity_scaling=1.0,
    viability_mode="recessive",
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9",
)

population.apply_preset(ta_drive)
```

参数说明：

1. `conversion_rate`：生殖系中 `target -> disrupted` 的转换概率，支持 `float`、`(female, male)` 或按性别字典
2. `embryo_disruption_rate`：胚胎期转换概率，可与 `cas9_deposition_glab` / `use_paternal_deposition` 联合建模母源/父源沉积效应
   - 如果设置了 `cas9_deposition_glab`，请确保 population 所属 species 在创建时通过 `gamete_labels` 注册了同名标签，否则应用预设时会触发 `KeyError`
3. `viability_scaling` 与 `viability_mode`：用于定义 `disrupted` 等位基因的毒素效应；TARE 常用 `viability_scaling=0.0` 且 `viability_mode="recessive"`
4. `fecundity_scaling` 与 `fecundity_mode`：定义繁殖力成本
5. `sexual_selection_scaling`（可选）：定义性选择效应；支持标量或二元组 `(default_male, carrier_male)`，配合 `sexual_selection_mode` 使用

加入性选择成本的示例：

```python
ta_drive_with_mating_cost = ToxinAntidoteDrive(
    name="TA_WithMatingCost",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    sexual_selection_scaling=(1.0, 0.8),
    sexual_selection_mode="dominant",
)
```

### PointMutation - 自发点突变

`PointMutation` 建模源等位基因自发突变为一个或多个目标等位基因。突变发生在每一个携带源等位基因的配子中，不依赖亲本基因型；每个目标都拿到你声明的速率——多目标之间是"同时竞争"，而不是互相吃掉份额：

```python
from natal.frontend.presets import PointMutation

# 单目标
mutation = PointMutation(
    "A2B",
    source_allele="A",
    target_allele="B",
    mutation_rate=1e-5,          # 1e-5 的源配子变成 B
    viability_scaling=0.98,      # 可选：目标等位基因的轻微适应度代价
)

# 多目标：声明的是有效速率，而不是级联份额
multi = PointMutation(
    "MultiMut",
    source_allele="A",
    target_alleles=["B", "C", "D"],
    mutation_rates=[1e-7, 5e-6, 1e-5],
)

population.apply_preset(mutation)
```

参数说明：

1. `source_allele`：发生突变的等位基因。所有携带它的配子都会转换，规则不带亲本基因型过滤（点突变是自发的）
2. `target_allele` / `mutation_rate` 与 `target_alleles` / `mutation_rates`：单目标与多目标两种声明形式；每个速率支持 `float`、`(female, male)` 二元组或按性别字典，缺失的性别键表示该性别不发生转换
3. `rate_mode`：`"strict"`（默认）把速率当作概率，和超过 1 时报错；`"proportional"` 把速率当作比例并缩放到和为 1，因此 `[2, 3, 5]` 与 `[0.2, 0.3, 0.5]` 是同一个模型
4. `viability_scaling` / `fecundity_scaling` / `sexual_selection_scaling` / `zygote_viability_scaling`（以及对应的 `*_mode`）：作用于整个目标等位基因组的适应度效应，默认中性

在目前实现中，突变只发生在生殖系（减数分裂产生配子时，受精之前）；预设不注册任何合子期修饰器。

同一个 ruleset 内的转换规则按级联执行：每条规则只能看到前一条规则剩下的源等位基因份额。若直接透传用户速率，`[0.3, 0.5, 0.1]` 中第二个目标的有效份额会变成 `0.5 × 0.7 = 0.35`。`PointMutation` 内部按 `r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)` 做补偿，因此 `A|A` 亲本的配子分布恰好是：

| 目标 | 声明速率 | 传给级联的速率 | 实际份额 |
|---|---|---|---|
| B | 0.3 | 0.3 | 0.3 |
| C | 0.5 | 0.5 / 0.7 ≈ 0.714 | 0.5 |
| D | 0.1 | 0.1 / 0.2 = 0.5 | 0.1 |
| A（未突变） | — | — | 0.1 |

这种"同时竞争"语义正是单个多目标 `PointMutation` 与叠加多个单目标预设的区别：后者的规则按注册顺序级联（先声明先得）。补偿按性别分别计算，因此按性别的速率在各自性别内独立竞争。

因此叠加多个单目标预设得到的是**顺序**模型，而不是同时模型。先声明 `W -> D`（速率 `mu`）、再声明 `D -> W`（速率 `nu`）时，级联给出 `q' = (1 - nu) (q + mu (1 - q))`，平衡 `mu (1 - nu) / (nu + mu (1 - nu))`；教科书"每条配子最多突变一次"的模型是 `q' = q (1 - nu) + mu (1 - q)`，平衡 `mu / (mu + nu)`。两者相差双突变项 `mu nu (1 - q)`：同一减数分裂内被转到 `D` 又转回 `W` 的配子，在级联里仍是 `W`，在同时模型里则算作 `D`（`mu = 0.02`、`nu = 0.06` 时为 0.2386 对 0.25）。要精确还原教科书模型，把**先声明**那条规则的速率放大：先声明 `mu / (1 - nu)`、再声明 `nu`（反向先声明则为 `nu / (1 - mu)` 加 `mu`），这样递推的斜率与截距同时配平。预设内部的 `r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)` 补偿不是跨预设配方——它只配平斜率，结果（0.2347）比不校正还远离教科书平衡点。速率在 `1e-3` 及以下时未校正偏差约 `1e-4`，可以忽略。

## 实用示例

### 简单点突变

```python
import natal as nt
from natal.frontend.presets import PointMutation

# 野生型等位基因 A 以 1e-4 的速率突变为 R
mutation = PointMutation(
    name="A2R",
    source_allele="A",
    target_allele="R",
    mutation_rate=1e-4,
)

# 构建种群并应用预设
species = nt.Species.from_dict("PointMutationSpecies", {
    "chr1": {"GeneA": ["A", "R"]}
})

pop = (nt.AgeStructuredPopulation.setup(species, name="MutationTest", stochastic=False)
       .age_structure(n_ages=5, new_adult_age=2)
       .initial_state({"female": {"A|A": [0, 0, 100, 0, 0]}})
       .presets(mutation)
       .build())

# 运行模拟
pop.run(n_steps=100)
```

### 多个预设组合使用

```python
import natal as nt
from natal.frontend.presets import HomingDrive, ToxinAntidoteDrive

# 物种需要声明预设中用到的全部等位基因
species = nt.Species.from_dict("MultiDriveSpecies", {
    "chr1": {"A": ["WT", "Drive", "Toxin", "Target", "Disrupted"]}
})

# 创建多个预设
drive1 = HomingDrive("Drive1", "Drive", "WT", drive_conversion_rate=0.95)
drive2 = ToxinAntidoteDrive("Drive2", "Toxin", "Target", "Disrupted", conversion_rate=0.90)

# 同时应用多个预设
pop = (nt.DiscreteGenerationPopulation.setup(species, name="MultiDriveTest")
       .presets(drive1, drive2)  # 应用多个预设
       .build())
```

## 深入学习

创建自定义预设是高级主题，详细内容参见以下专门文档：

- [设计你自己的预设](3_custom_presets.md)

## 相关章节

- [设计你自己的预设](3_custom_presets.md) - 详细的转换规则系统和预设设计
- [基因型模式匹配](2_genotype_patterns.md) - 语法规则与 pattern 设计
- [种群观测规则](2_data_output.md) - pattern 在观察分组中的使用
- [Modifier 机制](3_modifiers.md) - 底层修饰器原理
- [快速开始](1_quickstart.md) - 基础使用教程
