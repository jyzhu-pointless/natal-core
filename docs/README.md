# NATAL 完整文档

欢迎来到 **NATAL** 框架的完整文档。这套文档致力于为用户提供从快速上手到深入理解的全面指导。

## 📚 文档结构

### 入门与基础

- **[快速开始](01_quickstart.md)** - 15分钟快速上手指南，基于 `mosquito_population.py` 的实际示例
- **[遗传结构与实体](02_genetic_structures.md)** - 理解 Species/Chromosome/Locus 及 Genotype 字符串化、全局缓存机制

### 核心算法与数据结构

- **[Simulation Kernels 深度解析](03_simulation_kernels.md)** - 逐步讲解 `run_tick`、`run_reproduction`、`run_survival`、`run_aging` 算法
- **[PopulationState & PopulationConfig](04_population_state_config.md)** - "编译"步骤、NumPy/Numba 底层设计
- **[IndexCore 索引机制](05_index_core.md)** - 遗传学对象与基因型、单倍基因型+label 的映射关系

### 高级功能

- **[Modifier 机制](06_modifiers.md)** - 如何修改映射矩阵、基因驱动实现、细胞质标记
- **[Hook DSL 系统](07_hooks_dsl.md)** - 声明式/选择器/原生 Numba 三种 Hook 写法
- **[Numba 优化指南](08_numba_optimization.md)** - 开关机制、编译耗时、缓存策略

### 数据与推断

- **[Samplers & 观察过滤](10_samplers_observation.md)** - 灵活的数据提取工具，用于统计推断和数据同化

### PMCMC 入口说明（重要）

- PMCMC 推断脚本统一为 `pmcmc_inference_multi.py`。
- 旧脚本 `pmcmc_inference.py` 已弃用并移除。
- 建议所有批处理、示例和实验命令都基于 `pmcmc_inference_multi.py`。

### 参考资料

- **[API 完整参考](09_api_reference.md)** - 所有主要类和方法的完整参考

---

## 🎯 快速导航

### 按使用场景

**初学者**: 先从 [快速开始](01_quickstart.md) 开始，理解最基本的 API 调用。

**模型开发者**: 了解 [遗传结构](02_genetic_structures.md) 和 [Modifier 机制](06_modifiers.md)，学会定义复杂的遗传学模型。

**数据分析师**: 使用 [Samplers 系统](10_samplers_observation.md) 进行灵活的观察过滤和数据提取。

**性能优化师**: 阅读 [Simulation Kernels](03_simulation_kernels.md)、[PopulationState & Config](04_population_state_config.md) 和 [Numba 优化](08_numba_optimization.md)。

**高级用户**: 深入学习 [Hook DSL](07_hooks_dsl.md) 和 [IndexCore](05_index_core.md)，实现自定义模拟逻辑。

### 核心概念速查

| 概念 | 说明 | 详细文档 |
|------|------|---------|
| **Species/Chromosome/Locus** | 遗传架构的声明式定义 | [遗传结构](02_genetic_structures.md) |
| **Genotype** | 基因型实体，支持字符串化 (如 "WT\|Drive") | [遗传结构](02_genetic_structures.md) |
| **PopulationState** | 运行时种群状态（个体计数、精子存储） | [State & Config](04_population_state_config.md) |
| **PopulationConfig** | 编译后的配置（映射矩阵、适应度） | [State & Config](04_population_state_config.md) |
| **simulation_kernels** | 纯函数化模拟核心，支持 Numba 加速 | [Kernels 深度解析](03_simulation_kernels.md) |
| **Hook DSL** | 声明式钩子系统，支持三种写法 | [Hook DSL](07_hooks_dsl.md) |
| **Modifier** | 配子/合子级别的遗传修饰 | [Modifier 机制](06_modifiers.md) |
| **ObservationFilter** | 灵活的数据分组和提取工具 | [Samplers 系统](10_samplers_observation.md) |
| **IndexCore** | 遗传学对象 ↔ 整数索引的映射注册表 | [IndexCore 机制](05_index_core.md) |
| **Modifier** | 修改映射矩阵，实现基因驱动等复杂遗传现象 | [Modifier 机制](06_modifiers.md) |
| **Gamete Label** | 标记配子的附加维度（如"Cas9沉积") | [Modifier 机制](06_modifiers.md) |
| **Hook DSL** | 声明式钩子系统，在模拟特定阶段注入逻辑 | [Hook DSL](07_hooks_dsl.md) |

---

## 🚀 典型工作流程

### 1️⃣ 定义模型（10 分钟）

```python
from natal.genetic_structures import Species

# 声明式定义遗传架构
sp = Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "drive": ["WT", "Drive", "Resistance"]
        }
    }
)

# 或使用链式 API
sp = Species("Mosquito")
sp.add("chr1").add("drive").add_alleles(["WT", "Drive"])
```

### 2️⃣ 初始化种群（"编译"）（5 分钟）

```python
from natal.nonWF_population import AgeStructuredPopulation

pop = AgeStructuredPopulation(
    species=sp,
    n_ages=8,
    initial_individual_count={
        "female": {"WT|WT": [0, 600, 600, ...]},
        "male": {"WT|Drive": [0, 300, 300, ...]},
    },
    survival_rates=[1.0, 1.0, 5/6, ...],
    expected_eggs_per_female=100,
    use_sperm_storage=True,
    gamete_labels=["default", "Cas9_deposited"],
)
```

### 3️⃣ 定义遗传规则（Modifier）（5 分钟）

```python
# 基因驱动：D|w 杂合子的 D 配子比例 > w
def gene_drive_mod(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        }
    }

pop.set_gamete_modifier(gene_drive_mod, hook_name="drive")
```

### 4️⃣ 定义模拟逻辑（Hook）（5 分钟）

```python
from natal.hook_dsl import hook, Op

@hook(event='first')
def release_drive():
    return [
        Op.add(genotypes='Drive|*', ages=[2, 3, 4, 5, 6, 7], 
               delta=50, when='tick == 10')
    ]

release_drive.register(pop)
```

### 5️⃣ 运行模拟（1 分钟）

```python
pop.run(n_steps=200, record_every=10)

# 查看结果
print(f"最终种群: {pop.get_total_count()}")
print(f"记录点数: {len(pop.history)}")
```

---

## 📖 关键设计特性

### 🧩 分层架构

```
高层用户界面 (AgeStructuredPopulation)
    ↓ [编译] 创建 PopulationConfig
中层配置 (PopulationConfig + PopulationState)
    ↓ [导出] 导出数组 + 配置
底层数值计算 (simulation_kernels)
    ↓ [Numba JIT]
性能关键路径 (numpy/numba 纯数值计算)
```

**优势**:
- 高层用户友好，低层完全数值化
- 支持 Numba JIT 编译，性能优异
- 可导出状态进行 Monte Carlo 批量模拟
- 支持动态修改 Modifier 和 Hook

### 🔗 对象与索引分离

**遗传学层面**（面向用户）:
- `Species`, `Chromosome`, `Locus`, `Gene`
- `Genotype`, `HaploidGenotype`

**索引层面**（面向计算）:
- `IndexCore` 维护 object ↔ integer index 映射
- 所有 numpy 数组都使用整数索引
- Modifier 和 Hook 基于索引进行操作

### ⚡ Numba 加速原理

1. **声明式配置**: 用户写高层 Python 代码
2. **编译步骤**: 生成 numpy 数组和 Numba 兼容的配置
3. **纯函数核心**: `simulation_kernels` 全部是 `@njit` 函数
4. **缓存优化**: 首次调用时编译，后续调用直接执行

---

## ⚠️ 常见问题

### Q: 什么时候需要了解 IndexCore?
**A**: 通常不需要直接接触。只有在编写自定义 Modifier 或 Hook 时，才需要通过 `pop.registry` 或 `pop._index_core` 获取索引。

### Q: Gamete Label 有什么用?
**A**: 用来标记配子的附加信息（如是否携带 Cas9）。可以有多个 label，每个 label 对应一个维度。在 Modifier 中可以根据 label 修改配子分布。

### Q: 为什么初始化很慢?
**A**: 初始化时要构建两个大映射矩阵（基因型→配子、配子→合子），复杂度与基因型数量的乘积有关。这只发生一次，之后的每个 tick 都很快。

### Q: 如何进行大规模 Monte Carlo 模拟?
**A**: 使用 `pop.export_state()` 导出状态和配置，然后调用 `simulation_kernels.batch_ticks()` 进行并行模拟。详见 [Simulation Kernels](03_simulation_kernels.md)。

### Q: Numba 编译需要多长时间?
**A**: Numba 默认启用以获得最佳性能。首次编译通常需要 1-5 秒。后续运行会使用缓存。要禁用 Numba 进行调试，使用：

```python
from natal.numba_utils import numba_disabled

# 临时禁用（推荐）
with numba_disabled():
    pop.run(n_steps=10)

# 或全局禁用
from natal.numba_utils import disable_numba
disable_numba()
```

---

**准备好了吗？** [从快速开始开始 →](01_quickstart.md)
