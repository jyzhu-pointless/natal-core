# SpatialPopulation 初始化优化计划

## 现状与瓶颈

### 1. 每个 deme 独立走 builder 流程

```python
# 当前写法（51×51 = 2601 demes）
demes = [build_deme(species, idx, ...) for idx in range(2601)]
```

每个 `build_deme` 调用经过：
- Species genotype 解析（索引查找）
- Numba hook 编译（`_compile_hooks` → `CompiledEventHooks.from_compiled_hooks`）
- config/fitness 数组分配与填充
- `_finalize_hooks` 触发 codegen

同构 deme 之间这些步骤完全重复。实测 1000 demes ~1s，2601 demes ~2.6s。

### 2. 全局 Config 共享是手动操作

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
    deme.import_config(shared_config)
```

- 忘记调用时每个 deme 持有独立 config，浪费内存
- `import_config` 只能共享 scalar 字段，Numba 编译产物各自独立

### 3. 异构配置缺乏批量表达

当前需要在循环中写条件分支：

```python
def build_deme(..., idx):
    k = 10000 if idx < n_demes // 2 else 5000
    ...
    .competition(carrying_capacity=k)
```

没有声明式的批量接口来定义空间渐变模式（左高右低、中心高四周低等）。

### 4. Deme 间缺乏延迟初始化

所有 deme 必须在构造 `SpatialPopulation` 之前全部就绪。

---

## 设计目标

1. **保留 Builder 链式 API 语法** — 不改现有 builder 的调用风格
2. **`batch_setting` 向量化表达异构参数** — 在链式调用中嵌入异构数据
3. **零重复构建同构 deme** — 共享 template 的编译产物
4. **渐进式复杂度** — 同构场景一行够用，异构场景加参数

---

## 核心设计：SpatialBuilder + batch_setting

### `batch_setting` 包装器

`batch_setting` 是标志性的参数包装器，表示"这是一个跨 deme 变化的参数"：

```python
batch_setting(value)                          # 所有 deme 相同（但显式标记为可扩展）
batch_setting([10000, 5000, 8000, ...])        # 列表/数组：按 deme 索引
batch_setting(spatial=lambda i, x, y: ...)     # 回调：接收 (deme_id, x, y)，返回标量
batch_setting({
    0: 10000,                                  # 只指定部分 deme
    range(1, 10): 5000,                        # range 映射
    "rest": 8000,                              # 其余默认
})
```

当 builder 的某个参数是 `batch_setting` 对象时，内部自动切换为 spatial 批量模式。

### SpatialBuilder 链式 API

```python
pop = SpatialPopulation.builder(species, n_demes=N, topology=HexGrid(rows=N, cols=N)) \
    .setup(name="spatial_demo") \
    .initial_state(
        female={"WT|WT": 5000, "Dr|WT": 50},      # 非 batch → 所有 deme 相同
        male={"WT|WT": 5000, "Dr|WT": 50},
    ) \
    .reproduction(eggs_per_female=50) \
    .competition(
        carrying_capacity=batch_setting(spatial=lambda i, x, y: 10000 if x < N//2 else 5000),
        juvenile_growth_mode="concave",
        low_density_growth_rate=6.0,
    ) \
    .presets(drive) \
    .fitness(fecundity={"R2::!Dr": 1.0, "R2|R2": {"female": 0.0}}) \
    .migration(kernel=..., migration_rate=0.1) \
    .build()
```

### SpatialBuilder 内部流程

```
build() 调用时:
1. 扫描所有参数，提取 batch_setting 对象
2. 如果没有 batch_setting → 同构优化路径（见下方 Phase 1a）
3. 如果有 batch_setting:
   a. 展开每个 batch_setting 为 per-deme 值列表（n_demes 长度）
   b. 根据 config 等价性对 deme 分组
   c. 对每组：
      - 用首个 deme 的 config 构建 template deme（走 builder 全流程一次）
      - 组内其余 deme 从 template 克隆（浅拷贝 config、registry、hooks）
   d. 用克隆好的 demes + topology/migration 构造 SpatialPopulation
```

---

## 实施路径

### Phase 1a：同构 SpatialBuilder（无 batch_setting）

没有 `batch_setting` 时，所有 deme 完全一致。此时 SpatialBuilder 只需 build 一个 template，然后 N 次浅拷贝。

```python
pop = SpatialPopulation.builder(species, n_demes=2601, ...) \
    .setup(...).initial_state(...).reproduction(...).competition(...) \
    .presets(drive).build()

# 以下两句是完全等价的表达（拆解内部逻辑）：
# template = DiscreteGenerationPopulation.builder(...).build()
# demes = [clone(template) for _ in range(2601)]
# pop = SpatialPopulation(demes, topology=...)
```

预期：2601 demes ~50ms（不含 template 首次 build 的 2-3ms）。

### Phase 1b：异构 SpatialBuilder（含 batch_setting）

检测到至少一个 `batch_setting` 参数时，按 config 等价性分组。

```python
pop = SpatialPopulation.builder(species, n_demes=4, ...) \
    .setup(...).initial_state(...) \
    .competition(carrying_capacity=batch_setting([10000, 8000, 6000, 4000])) \
    .build()

# 内部：4 种不同 config → 4 个 template → 各自浅拷贝（每组只构建一个 template）
# 如果只有 2 组异构（如前 2 个 K=10000，后 2 个 K=5000）→ 只 build 2 个 template
```

### Phase 1c：`batch_setting.spatial` 便捷 API

空间渐变参数可通过 `batch_setting.spatial` 快速表达：

```python
batch_setting.spatial(lambda x, y: 10000 if abs(x) < 5 else 5000, topology=hex_grid)
```

接收拓扑坐标，隐式填充所有 deme 位置。

### Phase 1d：`set_hook` 在 SpatialBuilder 中的集成

```python
SpatialPopulation.builder(...) \
    ...
    .set_hook(event="early", fn=my_hook, deme=batch_setting([0, 1, 2])) \
    .build()
```

Deme 选择器自动展开为每个目标 deme 的 `set_hook` 调用。

---

## 后续 Phase（按需推进）

### Phase 2：轻量 per-deme 构造器

当性能需求超过 builder 的 DSL 便利性时，提供跳过 builder 的数组级构造：

```python
# 内部构造路径（用户不需要直接调用）
DemeFactory.quick(
    species=species,
    individual_count=np.array(...),  # (n_sexes, n_ages, n_genotypes)
    config=PopulationConfig(...),
    registry=shared_registry,
)
```

### Phase 3：SpatialPopulation 直接接收数组

```python
SpatialPopulation(
    demes=...,              # 传统方式
    species=species,        # 二选一：直接提供数组
    individual_counts=...,  # (n_demes, n_sexes, n_ages, n_genotypes)
    config=...,             # shared or bank
    topology=...,
    migration=...,
)
```

---

## `batch_setting` 的探测与展开机制

### Detect

```python
class batch_setting:
    """Marker wrapper for per-deme varying parameters."""
    
    _kind: Literal["scalar", "array", "spatial", "partial"]
    _data: Any
    
    def __init__(self, value):
        if callable(value) and "spatial" in hint: ...
        ...
```

Builder 的每个 setter 方法检查参数类型：

```python
def competition(self, carrying_capacity=None, ...):
    if isinstance(carrying_capacity, batch_setting):
        self._batch_params["carrying_capacity"] = carrying_capacity
    else:
        self._params["carrying_capacity"] = carrying_capacity
    return self
```

### Expand

`build()` 时将 `batch_setting` 展开为 `n_demes` 长度的列表：

```python
# 展开算法
for name, bs in self._batch_params.items():
    if bs._kind == "scalar":
        values = [bs._data] * n_demes
    elif bs._kind == "array":
        values = list(bs._data)  # 长度必须 == n_demes
    elif bs._kind == "spatial":
        values = [bs._fn(self._topology.from_index(i)) for i in range(n_demes)]
    ...
```

### Group

按展开后的 config 等价性分组：

```python
groups: dict[tuple, list[int]] = {}
for i in range(n_demes):
    cfg_key = tuple(
        params[name][i] if isinstance(params[name], list) else params[name]
        for name in config_fields
    )
    groups.setdefault(cfg_key, []).append(i)
```

---

## 性能预期

| 场景 | 当前 | 优化后 |
|------|------|--------|
| 同构 2601 demes | ~2.6s | ~50ms（+ template 首次 2-3ms）|
| 2 组异构 2601 demes | ~2.6s | ~55ms（+ 2 个 template 各 2-3ms）|
| 全异构 2601 demes | ~2.6s | ~2.6s（无法跳过 builder，但语法简化）|

---

## 技术风险

| 风险 | 缓解 |
|------|------|
| batch 展开后 config 分组 key 不可哈希（含 NumPy 数组）| 用 `id(arr)` 或序列化摘要 |
| 克隆 demes 时 `_compiled_hooks` 共享引用导致状态泄漏 | Copy-on-write：`set_hook`/`remove_hook` 时按需复制 |
| `PopulationConfig` 是否支持 `_replace`？ | 目测是 NamedTuple，确认后可用 |
| SpatialBuilder 与现有 `DiscreteGenerationPopulationBuilder` 的关系 | SpatialBuilder 内部持有 per-deme builder，复用其校验逻辑 |

---

## 接口设计原则

1. **不破坏现有 API** — `DiscreteGenerationPopulation.builder()` 和现有`SpatialPopulation.__init__` 不动
2. **`batch_setting` 是可选增强** — 不加 demo 照样跑，加了语法更简洁
3. **零拷贝优先** — 同构 config/hooks 共享引用，只复制差异行为
4. **验证前置** — `build()` 时校验 batch_setting 长度/坐标与 n_demes 一致
