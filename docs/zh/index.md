# NATAL Core 文档首页

[![NATAL](https://img.shields.io/badge/NATAL-dev-purple.svg)](https://github.com/jyzhu-pointless/natal-core)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.60.0+-orange.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![logo](./natal-brand.svg)

NATAL Core 是一个用于群体遗传动力学模拟的 Python 工具包，核心计算路径基于 Numba 加速。

## 核心特性

- **🧬 遗传结构：** 支持定义完整的基因组-染色体-位点-基因层次化体系
- **🪲 群体模型：** 支持年龄结构群体与离散世代群体，支持各类生态学和遗传学参数
- **🌐 空间模拟：** 支持空间种群模拟，使用六边形和正方形网格的亚种群
- **🚀 高性能内核：** Numba JIT 加速的模拟内核，空间种群下内置并行计算
- **🧩 可扩展机制：** Modifier、Preset 和 Hook 多重扩展体系，内置常用基因驱动
- **🔍 观测与筛选：** 支持灵活的模式匹配筛选机制，用于定义 Preset 和观测规则

## 安装

### 1. 创建并激活虚拟环境

推荐使用虚拟环境管理依赖，以下命令可任选其一。

```bash
# uv（推荐）
uv venv --python 3.12 .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

```bash
# venv
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

```bash
# conda
conda create -n natal-env python=3.12
conda activate natal-env
```

### 2. 安装 NATAL Core

```bash
uv pip install natal-core
# 或
pip install natal-core
```

## 学习路径（按推荐顺序）

1. [快速开始：15 分钟上手 NATAL](quickstart.md)
2. [遗传结构与实体](genetic_structures.md)
3. [Builder 系统详解](builder_system.md)
4. [IndexRegistry 索引机制](index_registry.md)
5. [PopulationState 与 PopulationConfig](population_state_config.md)
6. [模拟内核深度解析](simulation_kernels.md)
7. [Numba 优化指南](numba_optimization.md)
8. [Modifier 机制](modifiers.md)
9. [Hook 系统](hooks.md)
10. [遗传预设使用指南](genetic_presets.md)
11. [Spatial 模拟指南](spatial_simulation_guide.md)
12. [种群观测规则](observation_rules.md)
13. [基因型模式匹配](genotype_patterns.md)
14. [设计自己的 Preset（1）：从等位基因转换规则开始](allele_conversion_rules.md)
15. [设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围](genotype_filter.md)
16. [设计自己的 Preset（3）：封装、验证与发布前检查](preset_encapsulation_and_validation.md)

## API 文档

- [Genetic Structures API](api/genetic_structures.md)
- [Population Builder API](api/population_builder.md)
- [Simulation Kernels API](api/simulation_kernels.md)
- [完整 API 目录](api/index.md)

## 源码仓库

- [GitHub: jyzhu-pointless/natal-core](https://github.com/jyzhu-pointless/natal-core)
