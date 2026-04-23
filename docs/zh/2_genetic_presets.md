# 遗传预设（Genetic Presets）

`Genetic Presets`（遗传预设）是 NATAL 框架中用于定义可重用遗传修饰的机制，支持基因驱动、突变系统和其他遗传修饰的快速配置。

## 概述

**遗传预设**（Genetic Presets）提供了一种标准化的方式来定义遗传修饰规则，包括：

- 修改配子生成规则（如基因驱动的过度分离）
- 改变合子发育过程（如胚胎抗性形成）
- 调整适应度参数（如驱动等位基因的成本）

## 在构建器中应用预设

```python
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="TestPop")
       .presets(preset1, preset2)  # 可以应用多个预设
       .build())
```

## 内置预设

### HomingDrive - 同源重组基因驱动

`HomingDrive` 实现 CRISPR/Cas9 类型的同源重组基因驱动：

```python
from natal.genetic_presets import HomingDrive

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
# 性别特异性参数
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # 性别差异
    late_germline_resistance_formation_rate=(0.02, 0.04),  # 元组形式 (female, male)
    embryo_resistance_formation_rate=0.01,
    functional_resistance_ratio=0.2,  # 20%的抗性等位基因是功能性的

    # 适应度成本
    viability_scaling=0.9,      # 10%生存力成本
    fecundity_scaling=0.95,     # 5%繁殖力成本
    sexual_selection_scaling=0.85  # 15%性选择劣势
)
```

### ToxinAntidoteDrive - 毒素-解毒剂驱动（TARE/TADE）

`ToxinAntidoteDrive` 用于建模"驱动等位基因触发目标位点破坏，破坏等位基因产生适应度损失，而驱动等位基因提供救援"的系统。

```python
from natal.genetic_presets import ToxinAntidoteDrive

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

## 实用示例

### 简单点突变

```python
from natal.genetic_presets import HomingDrive
from natal.population_builder import AgeStructuredPopulationBuilder

# 创建基因驱动
drive = HomingDrive(
    name="DemoDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate=0.95
)

# 构建种群并应用预设
species = Species.from_dict("TestSpecies", {
    "chr1": {"GeneA": ["WT", "Drive"]}
})

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="DriveTest", stochastic=False)
       .age_structure(n_ages=5)
       .initial_state({"female": {"WT|WT": [0, 0, 100, 0, 0]}})
       .presets(drive)
       .build())

# 运行模拟
pop.run(ticks=100)
```

### 多个预设组合使用

```python
from natal.genetic_presets import HomingDrive, ToxinAntidoteDrive

# 创建多个预设
drive1 = HomingDrive("Drive1", "Drive", "WT", conversion_rate=0.95)
drive2 = ToxinAntidoteDrive("Drive2", "Toxin", "Target", conversion_rate=0.90)

# 同时应用多个预设
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="MultiDriveTest")
       .presets(drive1, drive2)  # 应用多个预设
       .build())
```

## 深入学习

创建自定义预设是高级主题，详细内容请参考以下专门文档：

- [设计你自己的预设](3_custom_presets.md)

## 相关章节

- [设计你自己的预设](3_custom_presets.md) - 详细的转换规则系统和预设设计
- [基因型模式匹配](2_genotype_patterns.md) - 语法规则与 pattern 设计
- [种群观测规则](2_data_output.md) - pattern 在观察分组中的使用
- [Modifier 机制](3_modifiers.md) - 底层修饰器原理
- [快速开始](1_quickstart.md) - 基础使用教程
