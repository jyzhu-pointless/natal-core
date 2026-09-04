# 后端选择与性能

`natal` 提供两个执行后端。默认的 `auto` 在原生扩展可用时选择它，否则回退到始终可用的纯 Python 参考实现：

| 后端 | 选择器 | 说明 |
|------|--------|------|
| Rust（原生扩展） | `backend="rust"` | 编译产物 `natal._engine_rs`，最快的路径；用 `maturin develop` 构建。 |
| 纯 Python 参考实现 | `backend="python"` | 始终可用；参考语义（oracle），最适合调试。 |
| 自动 | `backend="auto"`（默认） | 扩展可导入时选 Rust，否则选参考实现。 |

## 选择后端

后端在构建时按种群选择：

```python
from natal.frontend.genetics import Species
from natal.frontend.population.age_structured import AgeStructuredPopulation

species = Species.from_dict("demo", {"chr1": {"loc": ["WT", "Dr"]}})
pop = AgeStructuredPopulation.setup(
    species, stochastic=False, backend="auto",  # auto / rust / python
).age_structure(4, 2).reproduction(eggs_per_female=50).build()

print(pop.using_rust_backend)  # 原生扩展生效时为 True
```

`pop.enable_rust_backend(seed=...)` 在构造后为种群启用 Rust；`pop.disable_rust_backend()` 回到参考实现。

已退役的编译后端选择器（`backend="numba"`）会抛出带迁移提示的 `ValueError` —— 请改用 `"rust"` 或 `"python"`。

## 两个后端的共同点

两个后端执行相同的 tick 顺序（first hook → 繁殖 → early hook → 密度调节与存活 → late hook → 年龄推进）与相同确定性算术。确定性（`stochastic=False`）轨迹在两者间逐位一致；随机运行使用独立的 RNG 流。

Hook 在两条路径上都经过同一个声明式 CSR 解释器；单参数 Python 回调会被桥接进原生会话。

## 性能建议

- Rust 后端是大型模型（基因型多、deme 多、运行长）的主要性能手段。
- 参考实现的复杂度为 O(基因型² × 年龄 × deme)/tick，定位是黄金参照与调试目标，而非生产吞吐。
- `compress=True` 会压缩基因型轴到可达类型，对两个后端都有加速。
