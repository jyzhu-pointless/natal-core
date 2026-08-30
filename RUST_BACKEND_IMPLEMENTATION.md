# Rust 后端实现现状与使用说明

> 更新日期：当前分支 `feat/rust-engine-backend`
> 状态：核心生命周期、离散/WF、空间模拟、CSR hook、迁移均已接入；P5 custom hook 机器码桥接仍为探索阶段。

---

## 1. 当前做了什么

### 1.1 总体架构

- 使用 **PyO3 + maturin** 构建 Rust 原生扩展 `natal._engine_rs`。
- Python 侧保留 Numba 作为默认后端和 golden reference。
- Rust 后端通过 `EngineSession` / `DiscreteEngineSession` / `SpatialEngineSession` / `HeterogeneousSpatialEngineSession` 暴露给 Python。
- NumPy 数组以零拷贝视图跨 FFI 边界，Rust 内部直接修改 flat slice。

### 1.2 已实现功能

| 模块 | 内容 |
|---|---|
| Age-structured lifecycle | reproduction、survival、aging、完整 tick、批量 run、history 记录 |
| Discrete-generation lifecycle | 离散代 reproduction / survival / aging、完整 tick、批量 run |
| Wright-Fisher | 融合 tick（first hook + WF update）、multinomial / Poisson / deterministic 三种模式 |
| CSR 声明式 hook | Rust 内解释 10 种 opcode、RPN condition、deme selector、stop 语义 |
| Spatial 同质 | Rayon 并行 per-deme lifecycle，每 deme 独立 RNG（`seed + deme_id`） |
| Spatial 异质 | config bank + `deme_config_ids`，每个 deme 使用自己的 config 与 RNG |
| 空间迁移 | adjacency / topology-kernel 的 deterministic 与 stochastic 两种路径 |
| 随机迁移 RNG | 已改为 per-deme RNG（`seed + deme_id`），与 lifecycle 策略一致 |
| 后端选择 | `backend="auto"` / `"rust"` / `"numba"` / `"python"` |
| Rust 文档 | 所有 Rust 源码已补充详细 Google-style 文档与业务逻辑注释 |

### 1.3 已接入的 Python API

- `AgeStructuredPopulation.enable_rust_backend(seed=...)`
- `DiscreteGenerationPopulation.enable_rust_backend(seed=...)`
- `SpatialPopulation.enable_rust_backend(seed=...)`
- `Population.setup(..., backend="auto" | "rust" | "numba" | "python")`
- `pop.using_rust_backend`
- `pop.disable_rust_backend()`
- `RustLifecycleBackend` / `RustDiscreteLifecycleBackend` / `RustSpatialLifecycleBackend` / `RustHeterogeneousSpatialLifecycleBackend`

### 1.4 测试与工具

- `tests/test_rust_lifecycle.py`
- `tests/test_rust_discrete_lifecycle.py`
- `tests/test_rust_population_integration.py`
- `tests/test_rust_spatial_backend.py`
- `tests/test_packaging.py`
- `demos/rust_backend_demo.py`
- `benchmarks/rust_backend_benchmark.py`
- `benchmarks/rust_backend_small_benchmark.py`
- `benchmarks/rust_backend_spatial_benchmark.py`
- `scripts/check_rust.py`
- `scripts/build_rust_wheel.py`
- `scripts/ci_full.py`

---

## 2. 怎样使用

### 2.1 构建 Rust 扩展

推荐开发模式：

```bash
maturin develop --skip-install --release
```

或安装为 Python 包：

```bash
pip install .
```

检查是否可用：

```python
from natal.engine.backends.rust_backend import rust_backend_available
print(rust_backend_available())
```

### 2.2 Age-structured 示例

```python
import natal as nt

sp = nt.Species.from_dict(
    name="demo",
    structure={"chr1": {"loc": ["A", "B"]}},
    gamete_labels=["default"],
)

pop = (
    nt.AgeStructuredPopulation.setup(sp, stochastic=False)
    .initial_state(individual_count={
        "female": {"A|A": 200},
        "male": {"A|A": 150},
    })
    .reproduction(eggs_per_female=10.0)
    .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
    .competition(juvenile_growth_mode=1, carrying_capacity=500)
    .build()
)

pop.enable_rust_backend(seed=2026)
pop.run(10, record_every=1, clear_history_on_start=True)

print(pop.using_rust_backend)
```

### 2.3 Discrete / Wright-Fisher 示例

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
    .initial_state(individual_count={
        "female": {"A|A": [0.0, 100.0]},
        "male": {"A|A": [0.0, 100.0]},
    })
    .reproduction(eggs_per_female=8.0)
    .competition(juvenile_growth_mode=0)
    .build()
)

pop.enable_rust_backend(seed=7)
pop.run(5)
```

如果 `config.extreme_speed_mode > 0`，Rust 自动走 Wright-Fisher 融合路径。

### 2.4 Spatial 示例

```python
demes = [build_deme(i) for i in range(4)]
spatial = nt.SpatialPopulation(demes=demes, migration_rate=0.2)
spatial.enable_rust_backend(seed=9)
spatial.run(10)
```

- 同质空间：所有 deme 共享一个 config。
- 异质空间：不同 deme 可通过 `deme_config_ids` 使用不同 config。
- Rust 空间路径支持 age-structured 与 discrete（含 WF）deme。
- 每 deme lifecycle 使用独立 RNG，随机迁移也已使用 per-deme RNG。

### 2.5 直接使用 backend 类

```python
from natal.engine.backends.rust_backend import RustLifecycleBackend

backend = RustLifecycleBackend(pop.config, seed=0)
next_state, result = backend.run_tick(pop.state)
```

### 2.6 运行测试与基准

```bash
# Rust 门禁
python scripts/check_rust.py

# Rust 相关测试
pytest tests/test_rust_lifecycle.py tests/test_rust_discrete_lifecycle.py \
       tests/test_rust_population_integration.py tests/test_rust_spatial_backend.py -q

# Demo
python demos/rust_backend_demo.py

# Benchmark
python benchmarks/rust_backend_benchmark.py
python benchmarks/rust_backend_small_benchmark.py
python benchmarks/rust_backend_spatial_benchmark.py

# 一键 CI（Rust + pytest + pyright + ruff + wheel）
python scripts/ci_full.py
```

---

## 3. 当前限制

- Rust 后端目前只支持 **CSR 声明式 hook**；custom hook（`@hook(custom=True)` 或带 `(state, config, deme_id)` 的 njit 函数）存在时自动回退 Numba。
- 尚未支持 Python lambda 自定义 juvenile density regulation curve（方案已写入 `DENSITY_CURVE_PLAN.md`，未实现）。
- 随机数与 Numba 是统计等价，不是逐位一致。
- rust-analyzer 可选门禁在部分 macOS 环境仍会报告环境相关 false positive，非阻塞。
- `.github/workflows` 因 `.gitignore` 限制尚未落盘，CI 脚本已本地可用。

---

## 4. 后续待办

### 4.1 短期

- [ ] 实现 `DENSITY_CURVE_PLAN.md`：Python lambda 自定义 juvenile density regulation curve（cfunc 桥接）。
- [ ] 将 P5 custom hook `@cfunc` 桥接从探索转为正式实现，使 Rust 后端支持 custom hook。
- [ ] 为空间随机迁移补充 per-deme RNG 的统计测试与文档（代码已改，测试待补充/确认）。
- [ ] 完善 Rust 后端在 `backend="auto"` 下的自动选择与回退策略。

### 4.2 中期

- [ ] 增加更多 Rust 性能 benchmark 基线，防止回归。
- [ ] 为离散空间、WF 空间补充更完整的 parity 测试。
- [ ] 考虑将 `.github/workflows` 落盘并接入 CI（需调整 `.gitignore` 策略）。

### 4.3 长期

- [ ] P6 GPU 后端抽象（当前不建议启动）。
- [ ] 更多高级 hook 类型（selector hook 等）的 Rust 支持。
- [ ] 任意 Python 回调（density curve、custom hook 等）的统一 cfunc 桥接框架。

---

## 5. Spatial benchmark 结果

新增脚本：`benchmarks/rust_backend_spatial_benchmark.py`

本地测试环境：

- 16 demes（4×4 von Neumann 邻接）
- 3 个双等位基因位点
- 每个 deme 初始成年个体较多
- 每个模式跑 20 ticks，重复 3 次取中位数

> 以下为本地一次 3 次重复取中位数的结果；不同机器、负载和缓存状态下会略有波动。

### Deterministic

| path | total_ms | per_tick_ms | speedup |
|---|---:|---:|---:|
| numba run(n) | 38.6 | 1.930 | — |
| rust run(n) | 12.9 | 0.646 | 2.99x |
| numba run_tick loop | 268.0 | 13.402 | — |
| rust run_tick loop | 12.7 | 0.636 | 21.08x |

### Stochastic

| path | total_ms | per_tick_ms | speedup |
|---|---:|---:|---:|
| numba run(n) | 44.5 | 2.225 | — |
| rust run(n) | 16.6 | 0.832 | 2.68x |
| numba run_tick loop | 273.2 | 13.662 | — |
| rust run_tick loop | 16.6 | 0.829 | 16.48x |

### 结论

- Rust 空间批量 `run(n)` 比 Numba 快约 **3 倍**。
- Rust 空间逐 tick 循环比 Numba 快约 **16–22 倍**（因为避免了 Numba 每次调用的 wrapper 开销）。
- 随机模式与确定性模式趋势一致。

---

## 6. 相关文档

- `RUST_BACKEND_PLAN.md`：P0–P6 推进计划与状态。
- `DENSITY_CURVE_PLAN.md`：自定义 juvenile density regulation curve 方案。
- `P5_EXPLORATION.md`：Numba `@cfunc` 机器码桥接探索报告。
- `research/p5_cfunc_bridge/`：P5 spike 代码。
