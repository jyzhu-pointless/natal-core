# 空间执行与迁移

空间容器中的 deme 是局部种群单元。容器拥有共同时间线和堆叠状态；逐个在 Python 中调用 deme 的 run，无法替代容器的统一调度与迁移。

## 共同布局与异构参数

[SpatialPopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/builder.py) 组织多个局部声明与共同发布。压缩时必须考虑共同运行布局，不能让不同 deme 的 ZType 索引 1 指向不同类型后再直接堆叠。

原生个体数量按 `(deme, sex, age, ZType)` 排列；年龄结构精子存储按 `(deme, age, female ZType, male ZType)` 排列。局部 deme 可以具有不同生态参数，遗传数据则通过变体组织。

后端的 [ecology_columns_from_drafts() 与 genetics_variant_bank()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) 是理解这两种组织方式的入口：生态值形成按 deme 取值的列，遗传张量形成可复用的变体库。`fork_variant()` 和 `refresh_variant_tensors()` 关联遗传分叉与更新；不能因为两个 deme 初始共享遗传数组，就默认允许写入时互相影响。

## 构建期折叠迁移结构

[fold_migration_csr()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/migration.py) 将邻接矩阵或迁移核转成 `MigrationCSR`。CSR 是压缩稀疏行格式：

| 数组 | 含义 |
| --- | --- |
| `indptr` | 长度为 D+1；每个源 deme 的边位于相邻两个指针之间 |
| `dest_idx` | 每条边的目标 deme |
| `weights` | 与目标一一对应的边权重 |
| rate 列 | 独立的 `(D, 2, A)` 外迁率，按源、性别、年龄选择 |

邻接模式按目标升序保存原始权重。迁移核按核的行优先顺序访问偏移，处理越界或环绕，再归一化。环绕可能产生重复目标；实现保留这些条目与访问顺序，以维持浮点累加行为。不要擅自排序或合并条目后声称完全等价。

`adjust_on_edge` 当前的分母选择会在最终行归一化中抵消，因此它对目的地分布通常只剩舍入差异。这个实现事实与“边缘必然损失外迁质量”的直觉不同，修改前应先读 `_kernel_row_entries()` 的具体分支。

## 执行与迁移不是原地扫一遍

[kernels/spatial.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/spatial.rs) 的 `schedule_deme_ticks()` 负责局部 tick 调度；`run_spatial_tick_heterogeneous()` 和 `run_spatial_tick_discrete()` 组织对应模型路径。迁移在空间生命周期流程中作用于各 deme 更新后的状态。

`migrate_csr_deterministic()` 从旧输入读，在新输出中累加。对每个源先计算外迁量，再按 CSR 目的地分配，最后把剩余量留在源。新缓冲区避免同一 tick 刚迁入的个体又作为另一个源被迁出。

例如仅考虑一种个体类别，A 有 100、B 有 0，A 的外迁率为 0.2 且唯一目标为 B，那么迁移后是 80 和 20。这只说明迁移阶段的守恒，不包含此前繁殖或死亡。

年龄结构路径把未交配雌性和按雄性类型记录的已交配雌性分别处理，再按雌性迁移率移动相关精子存储。`migrate_csr_stochastic_rngs()` 则通过源 deme 的 RNG 采样外迁量并分配目的地；不能把精子当作独立于雌性的个体再次迁移。

## 所有权与验证入口

[SpatialPopulation](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/spatial/population.py) 管理会话和容器级操作；局部 deme 的参数访问不能赋予它独立 run、reset 或恢复共享时间线的权限。

- [test_spatial_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_publication.py)：共同发布和变体关系。
- [test_spatial_migration_conservation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_migration_conservation.py)：迁移守恒。
- [test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py)：deme 执行限制、局部参数原子性与共享时钟。

迁移修改需要分别检查总量、雌性与存储对应关系、边缘和重复目标、源遍历顺序、随机流及停止边界。生命周期前后总量不相等，并不能单独证明迁移不守恒。
