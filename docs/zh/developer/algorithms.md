# 关键数值算法（导航）

本章原先集中描述数值循环，现在相关内容已经拆入各自的专题章节。这一页保留为导航入口：它给出每个算法的落点、仍在正文中重要的数值约束，以及旧的代码与测试链接。

## 算法的落点

| 主题 | 现在的入口 | 读完能够回答 |
| --- | --- | --- |
| 一次繁殖的完整推导（M、F、P、配对、产卵、遗传损失） | [一次繁殖如何计算](reproduction.md) | 200 个后代是怎么一步步算出来的 |
| 生存、密度调节顺序与世代替换 | [生存与世代更替如何计算](survival.md) | 幼体如何被筛选，旧成体何时消失 |
| 年龄权重、长期精子存储与年龄推进 | [年龄结构与长期精子存储](age_structure.md) | 世代重叠与存储精子如何参与繁殖 |
| 密度曲线、承载量与平衡量 | [密度调节与平衡量如何计算](density_regulation.md) | `x`、`g(x)`、`C*`、`s*` 各自是什么 |
| 抽样点、随机流与复现范围 | [随机采样与可复现性](randomness.md) | 哪些边界跳过抽样，同一个种子承诺了什么 |
| 融合 Wright–Fisher 一步 | [融合 Wright–Fisher 执行路径](wright_fisher.md) | 哪些阶段被合并，哪些 Hook 不再运行 |
| 遗传映射如何从声明生成 | [遗传预设与转换规则如何编译](genetic_compilation.md) | 基线与规则的应用顺序 |
| 发布时的坐标变换 | [可达性、索引压缩与模型发布](publication.md) | 数组如何在最终轴上重建 |

## 仍然需要在正文里记住的数值约束

这些约束不适合拆进单一专题，但改动算法时都会被碰到：

- **后代张量的累加顺序固定**。[offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs) 的 `compute_offspring_tensor_flat()` 按 `(gf, gm, go, hf, hm)` 嵌套求 `P[gf, gm, go] = Σ meiosis_f[gf, hf] · meiosis_m[gm, hm] · fusion[hf, hm, go]`，跳过为零的项，并显式保持逐项累加顺序。源码注释禁止重结合、向量化或融合乘加：Python 侧与原生侧要求位级一致。
- **概率损失不重新归一化**。P 的某个亲本切片总和可以小于 1，这是允许表达的损失；只有随机路径在"先按总和抽样存活卵数"之后才用条件概率分配类型。
- **零权重行不参与除法**。交配概率的行和必须有限且大于阈值才做归一化，否则整行置零，避免"没有可用雄性"变成 NaN 或均匀交配。
- **密度调节先于生存**。顺序改变会改变同一组参数的结果，见[密度调节与平衡量如何计算](density_regulation.md)。
- **离散抽样先取整**。离散随机模式把数量四舍五入为整数再抽样；连续采样保留小数质量。两者结果不可逐位比较。

## 手算校验的例子

这些例子在各专题里给出完整上下文，这里只列出可直接口算的结果：

| 例子 | 结果 |
| --- | --- |
| 双亲都以 1/2 产生 A 或 a 配子 | 一对亲本的 P 切片为 `(1/4, 1/2, 1/4)` |
| 权重 20 与 10 的两个雄性 | 交配对象概率 2/3 与 1/3，不等于"所有雌性都交配" |
| 100 个 A\|a 成体、每雌 2 卵、幼体生存率 0.5 | `early` 每性别 `[100, 100]`，`late` 每性别 `[50, 100]` |

## 代码与测试入口

- [offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs)：后代张量推导。
- [discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs)：离散阶段与融合一步。
- [age_structured.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/age_structured.rs)：年龄结构阶段与精子存储。
- [density_regulation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/density_regulation.rs) 与 [equilibrium.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/equilibrium.rs)：曲线与平衡量。
- [rng.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/rng.rs)：抽样分布与随机流。
- [test_offspring_alignment.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_alignment.py)、[test_offspring_single_derivation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_single_derivation.py)、[test_rust_discrete_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_rust_discrete_lifecycle.py)：数值与生命周期合同。

从[一次繁殖如何计算](reproduction.md)开始读数值主线，或从[项目架构与职责边界](architecture.md)重新建立整体认识。
