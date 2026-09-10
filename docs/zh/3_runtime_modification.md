# 运行时参数修改

种群构建完成后，所有参数都可以在模拟运行时动态修改——无需重建。覆盖三种场景：

- **between-tick**：Python 侧通过 `pop.update()` 或 `pop.params.<name> = v` 修改
- **hook 内**：通过回调 Hook 的 `pop.params`（`TickContext`）写入，或 `Op.set_param` 声明式调度
- **spatial**：per-deme 写入（`pop.params.tensor_write` / `deme(i).write_ecology`）

**怎么选：**默认使用 Hook 声明式更新——Hook 明确写出每次修改发生的 tick、
事件阶段和执行顺序，写入随事件原子提交（失败自动回滚）。在两次运行之间调参、
做参数扫描或运行前微调时，使用 between-tick 写入面（`pop.update()` /
`pop.params`）——它们是完全支持的一等场景，不是弃用路径。

---

## 1. between-tick 修改：`pop.update()`

`pop.update()` 返回**绑定到运行种群**的 `RuntimeUpdater`。域方法语法与构建时完全一样；每次调用经参数路由表校验后提交到运行种群（draft 与 Rust 会话同步），同一 tick 的后续阶段立即可见：

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .build()
)

# 单个参数
pop.update().competition(carrying_capacity=5000)

# 链式多个参数
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# 自定义字段（回调内写入；回调外读 pop.config.custom）
pop.update().custom(temperature=35.0)
```

每次调用都提交到运行种群，并在值实际变化时追加一条参数日志；它操作的是运行种群，不是查询快照。

## 2. between-tick 修改：`pop.params` 参数面

`pop.params.<name> = value` 是 between-tick 写入的首选通道：属性写入经 jsonc 边界校验，
同时到达 draft、Rust 会话与参数快照日志。读取返回当前值：

```python
pop.params.carrying_capacity = 5000.0
print(pop.params.carrying_capacity)  # 当前值
# 向量/张量参数走专用通道
pop.params.tensor_write("survival_rates", np.ones((2, 2)))
```

每次实际写入追加一条 `(tick, name, old, new)` 到 `pop.params_log`：

```python
for row in pop.params_log:
    print(f"tick={row[0]}  {row[1]}  {row[2]} -> {row[3]}")
```

---

## 3. `set_param()`：草稿层底层接口

`set_param()` 只写你传入的 **draft（草稿）**，不会写入任何运行种群。`pop.config` 是查询快照，对它调用 `set_param()` 不会改变种群；要修改运行种群请用 `pop.update()` 或 `pop.params`。它适合纯草稿场景（脚本、notebook、离线构造配置）：

```python
from natal.frontend.builder import set_param

# 生态标量是 NamedTuple 槽位：必须接住返回值并重新绑定
draft = pop.config
draft = set_param(draft, "competition.carrying_capacity", 5000.0)

# 全名、短名、别名均可
draft = set_param(draft, "carrying_capacity", 5000.0)
draft = set_param(draft, "reproduction.eggs_per_female", 100.0)
draft = set_param(draft, "eggs_per_female", 100.0)  # 别名
```

要点：

1. 名称经 `parameters.jsonc` 注册表解析：全名 → 短名 → 别名。
2. 生态标量走 `NamedTuple._replace`，**必须重新绑定返回值**；数组类字段（custom 槽位、向量/张量内容）原地修改并返回同一个 draft。
3. 传入 `pop.config` 得到的是隔离快照，写入只落在该快照上；即使接住返回值也不会影响运行种群。
4. 均衡指标（expected_competition_strength / expected_survival_rate）按需现算（derive），没有存储副本需要同步。

---

## 4. Hook 内修改

回调 Hook 的 `pop.params` 与 `pop.update()` 是同一套写入器栈，写入语义一致：

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="early")
def heatwave(pop: TickContext) -> int:
    if pop.tick == 10:
        pop.params.carrying_capacity = 2000.0
        pop.params.eggs_per_female = 100.0
        pop.params.sex_ratio = 0.55
    return 0
```

规则与注意点：

- 属性写入目标为生态标量（`carrying_capacity`、`eggs_per_female`、`sex_ratio`、
  `sperm_displacement_rate`、`low_density_growth_rate` 等，即 `Op.set_param` 的
  5 参数目标表 + 同类生态标量）；越界值抛 `ValueError`，写入后同一 tick 后续阶段
  立即可见。
- 向量/张量参数用 `pop.params.tensor_write(name, values)`。
- 均衡指标不再存储于配置中：任何读取（`pop.params.expected_competition_strength`
  等）都会按当前生态现算，写入生态参数后立即反映，无需手动同步。
- 声明式的 `Op.set_param("carrying_capacity", "K * 0.95", every=10)` 等价于按计划
  执行同一写入链（无需任何 Python 代码），详见 [Hook 系统](2_hooks.md)。

---

## 5. 自定义字段 `config.custom`

`custom` 是 0-d structured numpy array。构建时通过 `.custom()` 注册字段和初始值；
读取走 `pop.config.custom["name"]`（查询快照），写入统一走 `pop.update().custom()`：

```python
# 构建
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0)
    .build()
)

# Hook 内：写入经事件事务提交，回调成功才生效
@nt.hook(event="early")
def seasonal_hook(ctx: TickContext) -> int:
    if ctx.tick == 0:
        ctx.update().custom(temperature=35.0)
    return 0

# 回调外
pop.update().custom(temperature=35.0, season_idx=1)
print(pop.config.custom["temperature"])  # 35.0
```

支持 `bool`、`float`、`int`，也支持任意维数数组；类型与形状保留。

> **注意**：自定义字段不在参数路由表中。回调内**没有公开的读取入口**：
> `TickContext` 不提供 `config`，`ctx.state` 也没有 `config` 属性，`ctx.params`
> 与 `Op.set_param` 都只接受注册参数（`Op.set_param` 遇到自定义字段在编译期抛
> `ValueError`）。请在回调外读取 `pop.config.custom["name"]`，回调内只做写入
> （`ctx.update().custom(...)`）。草稿层 `set_param(draft, "temperature", v)` 能写
> custom 槽位，但和所有 `set_param` 调用一样不提交到运行种群。

---

## 6. 空间种群 per-deme 修改

**`SpatialPopulation.update()` 链式接口已删除。** 空间种群的运行时写入走两个入口：

### 6.1 `pop.params`（between-tick 批量写入推荐）

`pop.params` 只读返回 `(n_demes, ...)` 生态列的写保护视图；`tensor_write` 校验形状后
通过共享写入通道按 deme 路由（列 + deme draft + Rust 会话列同步）：

```python
from natal.frontend.spatial import batch_setting

# 全部 deme 同值：per-deme 形状广播
pop.params.tensor_write("survival_rates", np.ones((2, 2)))

# 每 deme 独立值：(n_demes, ...) 全列
pop.params.tensor_write("carrying_capacity", np.array([5000.0, 5000.0, 8000.0, 8000.0]))

# migration_rate 保留构建期语法糖：标量 / 按性别映射 / (n_ages,) / (S, A) / (n_demes, S, A)
pop.params.tensor_write("migration_rate", {"F": 0.2, "M": 0.05})

# 数值读取（写保护视图）
print(pop.params.migration_rate.shape)  # (n_demes, S, A)
```

`migration_rate` 在运行时契约中的形状为 `(n_demes, S, A)`（A 为年龄轴；标量/映射等
小形状自动广播，标量语义为成年年龄两性取该值、幼年取 0）。

### 6.2 `deme(i).write_ecology` / `write_genetics`（单 deme 写入）

`pop.deme(i)` 返回 `DemeSlice` 视图：其 `config`/`state`/`registry`/`name` 等读取
全部委托给底层 deme 对象；写入走两个专用方法：

- `write_ecology(field, value)`：同时写入生态列与该 deme 的 draft（按字段
  clone-on-write），任何执行路径（Python 分派与 Rust 会话列）都看到同一值。
- `write_genetics(field, values)`：先 fork 该 deme 的遗传变体（Rust 侧），再分离
  draft 表，保证共享这些表的其他 deme 数值逐位不变。

```python
pop.deme(3).write_ecology("carrying_capacity", 8000.0)
pop.deme(3).write_genetics("viability_fitness", new_table)
```

### 6.3 batch_setting 单一入口

构建时 `batch_setting([...])` 是 per-deme 异构参数的**唯一**声明入口：kind 由值的
种类推断（`"scalar"` / `"array"` / lambda 的 `"spatial"`），同构路径与异构路径在
`build()` 时自动分叉。fitness/presets 不支持 `batch_setting`（它们修改 config 内部
ndarray，不适合标量表达）；`spatial` kind 的 lambda 需要 builder 传入 topology。

---

## 7. 底层机制

运行期写入入口（`pop.update()`、`pop.params`、hook 内 `ctx.params`/`ctx.update()`、
声明式 `Op.set_param`）最终都落到同一条链上：

```
运行期写入入口
  → 路由表解析名称 + jsonc 边界校验
  → draft 字段更新（生态标量走 NamedTuple._replace）
  → Rust 会话参数同步 + 参数快照日志 (tick, name, old, new)
  → 读取时现算均衡指标（derive，无存储副本）
```

草稿层 `set_param(draft, name, value)` 只完成其中的 draft 更新，不经过会话同步，
也不会写入运行种群。

生态标量（K、eggs、sex_ratio、sperm_displacement_rate、low_density_growth_rate、
juvenile_growth_mode、generation_time）在运行合同中均为普通标量；均衡指标
（expected_competition_strength、expected_survival_rate）为只读派生值，经
`pop.params.<name>` 读取（现算），直接写入会抛 AttributeError。

### `set_config()` — 整体配置替换

`pop.set_config(new_config)` 一次性替换种群的整个配置对象。适用于从头重建配置后
（例如修改了 custom 字段结构）。新配置必须与原有配置类型相同（`ModelDraft`，
且离散模型须满足离散归一化不变量）。

PopulationBuilder 的 `custom()` 方法在添加新字段时会触发此路径：它会重建 custom
结构化数组并调用 `set_config()` 将新配置写回种群。

---

## 8. 参数参考

参数按领域分组，与 PopulationBuilder 链式 API 方法对应。

| 领域 | 参数名 | 别名 | 适用模型 | set_param |
|---|---|---|---|---|
| setup | `stochastic` | — | both | ❌ 构建时 |
| setup | `continuous_sampling` | — | both | ❌ 构建时 |
| setup | `fixed_egg_count` | — | both | ❌ 构建时 |
| setup | `has_sex_chromosomes` | — | both | ❌ 构建时 |
| age_structure | `n_ages` | — | age-structured | ❌ 构建时 |
| age_structure | `new_adult_age` | — | age-structured | ❌ 构建时 |
| age_structure | `generation_time` | — | age-structured | ❌ 构建时 |
| survival | `female_age_based_survival` | — | age-structured | ✅ |
| survival | `male_age_based_survival` | — | age-structured | ✅ |
| reproduction | `eggs_per_female` | `expected_eggs_per_female` | both | ✅ |
| reproduction | `sex_ratio` | — | both | ✅ |
| reproduction | `sperm_displacement_rate` | — | both | ✅ |
| competition | `carrying_capacity` | — | both | ✅ |
| competition | `low_density_growth_rate` | — | both | ✅ |
| competition | `juvenile_growth_mode` | `growth_mode` | both | ✅ |
| fitness | `viability` | — | both | ❌ 张量 |
| fitness | `fecundity` | — | both | ❌ 张量 |
| fitness | `sexual_selection` | — | both | ❌ 张量 |
| fitness | `zygote_viability` | — | both | ❌ 张量 |
| migration | `migration_rate` | — | spatial | 仅空间 |

## 9. 新旧对比

| | 旧（Builder / njit 时代） | 新（PopulationBuilder） |
|---|---|---|
| 构建后修改 | 不支持 | `pop.update()` |
| Hook 内修改 | `(state, config, deme_id)` 直接写 | `pop.params`（TickContext）或 `Op.set_param` |
| 参数审计 | 无 | `pop.params_log`（(tick, name, old, new)） |
| 自定义字段 | ConfigMutator（已删除） | `config.custom` |
| 空间 per-deme 写入 | `SpatialPopulation.update()` 链（已删除） | `pop.params.tensor_write` + `deme(i).write_ecology` |
| 底层接口 | 无 | `set_param(config, name, value)` |
