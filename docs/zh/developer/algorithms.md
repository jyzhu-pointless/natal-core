# 关键数值算法

繁殖部分的完整推导参见[一次繁殖如何计算](reproduction.md)，包括交配权重、配对数量、遗传损失、性别与适应度的阶段差异。

本章把数值含义连接到具体循环。以下公式描述实现中的局部计算，不替代完整生命周期；fitness、交配率、产卵、生存和密度调节发生在不同位置，不能全部合并成一个未注明顺序的乘数。

## 从遗传映射推导后代概率

在[offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs) 中，`compute_offspring_tensor_flat()` 将减数分裂表 M 和配子融合表 F 合成为 P。令 i、j、k 为雌性亲本、雄性亲本和后代 ZType，u、v 为雌雄配子 GType：

\[
P_{ijk}=\sum_{u=0}^{G-1}\sum_{v=0}^{G-1}M_{0iu}M_{1jv}F_{uvk}.
\]

M 的 shape 是 `(2, Z, G)`，F 是 `(G, G, Z)`，P 是 `(Z, Z, Z)`。Rust 使用行优先的一维数组：P 的偏移为 `(i * Z + j) * Z + k`，雄性减数分裂表的偏移为 `(Z + j) * G + v`。

循环顺序是 i、j、k、u、v，M 中为零的项直接跳过。源码明确保留逐项累加顺序，避免浮点重结合或融合乘加改变位级结果。因此，将数学上等价的收缩表达式替换进去之前，需要检查数值一致性要求。

作为手算校验，设双亲都以 1/2 概率产生 A 或 a 配子，融合将 AA、Aa/aA、aa 分别映射为三个后代类型，则一对亲本的 P 切片为 `(1/4, 1/2, 1/4)`。这只是中性遗传映射算例，尚未乘入后代数量或合子适应度。

## 交配概率与数量分开计算

[discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs) 的 `compute_mating_probability()` 先构造权重，再逐雌性类型归一化。若 S 是性选择 fitness，m 是传入函数的雄性数量向量：

\[
w_{ij}=S_{ij}m_j,\qquad q_{ij}=w_{ij}/\sum_j w_{ij}.
\]

只有行总和有限且大于 `EPS` 才执行除法，否则该行全部置零。这避免“无可用雄性”变成 NaN 或均匀交配。若某行权重为 20 和 10，交配对象概率就是 2/3 和 1/3；这并不意味着所有雌性都会交配。

`mate_discrete()` 继续根据雌性数量和交配率计算或采样配对数，`fertilize_discrete()` 再处理产卵、亲本 fecundity 与后代类型分配。年龄结构版本的 `compute_mating_probability_matrix()`、`sample_mating()`、`fertilize()` 还处理年龄权重与精子存储，不能直接用离散配对缓冲替代。

## 从配对数走到下一代

在确定性离散路径中，令配对数为 Cᵢⱼ、繁殖概率为 b、每雌性产卵数为 e、雌雄 fecundity 为 fᵢ 和 fⱼ，则该配对的期望卵数为 Cᵢⱼ·b·e·fᵢ·fⱼ。`fertilize_discrete()` 将它乘 Pᵢⱼₖ 并按后代类型累加。

P 的一个亲本切片总和可以小于 1，表示遗传融合路径中损失的质量；不能无条件归一化来抹去损失。随机路径先用该总和对卵数进行存活抽样，再对存活后代使用条件归一化概率分配类型。随后按性染色体兼容性或全局性别比例分配性别，雄性取总数减雌性的剩余值。

`reproduction()` 接着施加 `zygote_viability_fitness`。`survival()` 先调用 `recruit_juveniles()` 做密度调节，再按“age-0 基础生存率 × 对应性别、年龄、类型的 viability”保留个体；最后 `aging()` 把它们变为成体。这说明合子适应度与普通 viability 的位置不同。

一个贯穿阶段的手算场景是：100 对已交配的中性 A|a 双亲，b=1、e=2，双亲 fecundity 均为 1，性别比为 1/2，无密度调节，合子适应度为 1，雌雄 age-0 生存率均为 1/2，普通 viability 为 1。遗传分配产生 AA、Aa、aa 共 `(50, 100, 50)`；分性别后每个性别为 `(25, 50, 25)`；survival 后每个性别为 `(12.5, 25, 12.5)`，aging 将这组数移到 age-1。小数是确定性期望数量，不是对整数随机结果的承诺。

## 密度曲线与实际缩放不是同一个量

[density_regulation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/density_regulation.rs) 把曲线与内核输入分开。令 x 为实际竞争强度与平衡竞争强度之比，r 为低密度增长率：

| mode | 曲线 g(x) |
| --- | --- |
| 0：无调节 | 1 |
| 1：fixed | min(1, 1/x)，x ≤ 0 时为 1 |
| 2：linear / logistic | max(0, r − (r − 1)x) |
| 3：Beverton–Holt | r / (1 + (r − 1)x) |
| 4：Ricker | r^(1 − x) |

`scaling_factor()` 分派曲线；生命周期使用 `regulation_scaling()`。后者在模式 2–4 中还乘平衡生存率，且在平衡竞争强度非正或 NaN 时返回零。fixed 路径直接计算 `equilibrium / actual`，避免用 `1 / (actual / equilibrium)` 引入额外舍入。

例如 r=2、x=2 时，linear、Beverton–Holt、Ricker 的曲线值分别为 0、2/3、1/2；最终招募缩放还依赖平衡生存率。年龄结构模型的竞争强度与年龄权重有关，不能一律把 x 写成“总种群数 / K”。模式 5 及以上目前没有自定义曲线注册机制。

## 随机路径需要检查退化分支

[rng.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/rng.rs) 集中实现 `binomial()`、`poisson()`、`multinomial()` 及连续对应版本。确定性路径使用期望数量；离散随机路径采样；连续采样采用另一组数值规则，并不意味着完整模拟自动可微。

例如 `continuous_binomial()` 在概率接近零或一时直接返回边界值，在 n ≤ 1 + EPS 时返回 n·p；其他情况用两个按固定顺序抽取的 Gamma 值构成 Beta 比例，再乘 n。交换抽样顺序会改变后续随机流，即使边缘分布仍然相同。

年龄结构的 `sample_survival_with_sperm()` 同时维护雌性数量与精子存储。测试不能只核对种群总量，还应检查存储对应的已交配雌性不会超过雌性数量，以及年龄推进后的对应关系。

## 验证入口

- [offspring 单元测试](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/tests/unit/kernels/offspring.rs) 与 [test_offspring_alignment.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_alignment.py)：概率推导与轴对齐。
- [密度曲线单元测试](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/tests/unit/kernels/density_regulation.rs) 与 [test_density_zero_equilibrium.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_density_zero_equilibrium.py)：曲线与零平衡行为。
- [test_mgdrive1_compatible_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_mgdrive1_compatible_lifecycle.py)：生命周期计算的参考场景。

算法修改应分别验证有依据的预期值、非负性或守恒等约束、随机统计性质与要求保持的复现性。只对比两份采用相同公式的实现，可能同时保留同一个错误。
