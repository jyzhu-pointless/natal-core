# NATAL Core 文档

**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.60.0+-orange.svg)](https://numba.pydata.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** 是一个高性能的前向时间群体遗传学模拟引擎，支持可配置的物种生命周期。它支持年龄结构化和离散世代种群、精子储存、遗传预设、基于 hook 的干预措施以及 Numba 加速的模拟内核。NATAL Core 尤其适用于**对昆虫种群中的基因驱动（gene drive）系统进行建模**，但其灵活的架构也使其能够应用于广泛的群体遗传学场景。

NATAL Core 是 NATAL 项目的一部分。完整项目还包括 **NATAL Inferencer**，这是一个基于 NATAL Core 的群体遗传学模型参数推断工具包。

## 主要特性

- 🪲 支持前向时间模拟，可灵活配置种群的生命周期（年龄结构化种群与离散世代种群）
- 🧬 可定义遗传结构，包括染色体、基因座和等位基因
- 🚀 基于 Numba 加速的计算核心，性能优秀
- 🧩 内置多种基因驱动预设，特别是 homing drive 和 toxin-antidote drive
- 🪝 提供 Hook 系统，可在模拟过程中插入自定义干预逻辑
- 🔍 配备观察与过滤工具，便于后续分析
- 🗺️ 支持多 deme（亚种群）空间模拟

## 安装

### 1. 创建虚拟环境

强烈建议使用虚拟环境来管理依赖项。

请选择以下命令之一。**推荐使用 Python 3.12**，但任何 Python 版本 >= 3.9 应该都可以工作。

```bash
uv venv --python 3.12 .venv            # uv（推荐）
python -m venv .venv                   # venv（请确保使用 Python >= 3.9）
conda create -n natal-env python=3.12  # conda
```

在 Windows 上，你可以运行 `py -3.12 -m venv .venv` 来指定 Python 3.12 作为虚拟环境的解释器。

### 2. 激活虚拟环境

Linux / macOS：

```bash
source .venv/bin/activate    # uv / venv
conda activate natal-env     # conda
```

Windows：

```powershell
.venv\Scripts\activate       # uv / venv
conda activate natal-env     # conda
```

### 3. 安装 NATAL Core

```bash
uv pip install natal-core
# 或
pip install natal-core
```

## 一个最简示例

```python
import natal as nt
from natal.ui import launch

# 1. 定义物种的遗传架构
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {"loc1": ["WT", "Dr", "R2", "R1"]}
    },
    gamete_labels=["default", "cas9_deposited"]
)

# 2. 使用内置预设定义驱动
drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.9,
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    viability_scaling=1.0,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

# 3. 使用 hook 定义释放事件
@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10")
    ]

# 4. 构建一个随机交配种群
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="TestPop",
        stochastic=True
    )
    .initial_state(
        individual_count={
        "male": {"WT|WT": 50000}, "female": {"WT|WT": 50000}
        }
    )
    .reproduction(
        eggs_per_female=100
    )
    .competition(
        low_density_growth_rate=6.0,
        carrying_capacity=100000,
        juvenile_growth_mode="concave"
    )
    .presets(drive).hooks(release_drive_carriers).build())

# 5. 启动交互式 WebUI 并运行模拟
launch(pop)
```

更多可即时使用的示例，请参阅 GitHub 仓库中的 [demos](https://github.com/jyzhu-pointless/natal-core/tree/main/demos) 目录。

## 文档目录

推荐先阅读第一部分，上手后以实际项目为驱动阅读第二部分，再根据实际需要选择性阅读第三、四部分。

### 第一部分：快速入门

> 本部分介绍 NATAL Core 的基本概念和使用方法，帮助你快速上手。

1. [快速入门：15 分钟上手 NATAL](1_quickstart.md)

### 第二部分：实用组件

> 本部分介绍 NATAL Core 的主要组件，它们是日常使用的主要功能。

2. [遗传结构与实体](2_genetics.md)
3. [种群初始化](2_population_initialization.md)
4. [随机交配种群](2_population.md)
5. [遗传预设使用指南](2_genetic_presets.md)
6. [Hook 系统](2_hooks.md)
7. [模式匹配与可扩展配置](2_genotype_patterns.md)
8. [提取种群模拟数据](2_data_output.md)

### 第三部分：进阶指南

> 本部分介绍 NATAL Core 的高级功能，包括空间模拟和更多自定义配置。

9. [空间模拟指南](3_spatial_simulation.md)
10. [设计你自己的预设](3_custom_presets.md)
11. [Modifier 机制](3_modifiers.md)
12. [高级 Hook 教程](3_advanced_hooks.md)

### 第四部分：内部实现

> 本部分介绍 NATAL Core 中不直接面向用户的底层实现机制，帮助你深入理解 NATAL Core 的工作原理。

<!--TODO: numba 相关内容可适当提前-->

13. [IndexRegistry 索引机制](4_index_registry.md)
14. [PopulationState 与 PopulationConfig](4_population_state_config.md)
15. [模拟内核深度解析](4_simulation_kernels.md)
16. [Numba 优化指南](4_numba_optimization.md)


## API 文档

- [完整 API 索引](api/index.md)

## 链接

- GitHub 仓库：https://github.com/jyzhu-pointless/natal-core
- PyPI 包：https://pypi.org/project/natal-core/

## 许可证

本项目采用 MIT 许可证。
