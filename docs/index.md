# NATAL Core 文档首页

NATAL Core 是一个用于群体遗传动力学模拟的 Python 工具包，核心计算路径基于 Numba 加速。

## 核心特性

- 遗传结构建模：支持染色体、位点、等位基因与基因型模式
- 群体模型：支持年龄结构群体与离散世代群体
- 高性能内核：Numba JIT 加速的模拟 Kernel
- 可扩展机制：Modifier 与 Hook 双扩展体系
- 观测与筛选：Sampler、过滤规则与模式匹配协同

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
6. [Simulation Kernels 深度解析](simulation_kernels.md)
7. [Numba 优化指南](numba_optimization.md)
8. [Modifier 机制](modifiers.md)
9. [Hook 系统](hooks.md)
10. [遗传预设使用指南](genetic_presets.md)
11. [Spatial 模拟指南](spatial_simulation_guide.md)
12. [Samplers：观察过滤系统](samplers_observation.md)
13. [模式匹配与可扩展配置](genotype_pattern_matching_design.md)
14. [设计自己的 Preset（1）：从等位基因转换规则开始](allele_conversion_rules.md)
15. [设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围](genotype_filter_implementation.md)
16. [设计自己的 Preset（3）：封装、验证与发布前检查](preset_encapsulation_and_validation.md)

## API 文档

- [Genetic Structures API](api/genetic_structures.md)
- [Population Builder API](api/population_builder.md)
- [Simulation Kernels API](api/simulation_kernels.md)
- [完整 API 目录](api/)

## 源码仓库

- [GitHub: jyzhu-pointless/natal-core](https://github.com/jyzhu-pointless/natal-core)
