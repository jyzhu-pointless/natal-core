# 自定义 Juvenile Density Regulation Curve 方案

> 状态：方案设计，暂不实现。
> 目标：支持用户通过 Python lambda 定义任意 juvenile density regulation curve，并在 Rust 后端中以 C ABI 机器码方式直接调用。

---

## 1. 目标

允许用户通过 Python lambda 定义任意 juvenile density regulation curve，并在 **Rust 后端**中以 **C ABI 机器码**方式直接调用，避免每 tick 调回 Python 的开销。

---

## 2. 用户 API 设计

推荐扩展 `.competition()`：

```python
pop = (
    nt.AgeStructuredPopulation
    .setup(...)
    .competition(
        juvenile_growth_mode=lambda ratio: max(0.0, 1.0 - ratio),
        expected_competition_strength=100.0,
    )
    .build()
)
```

- `juvenile_growth_mode` 现在允许：
  - `int`：保留现有 0/1/2/3
  - `str`：保留现有 `"logistic"` / `"fixed"` 等
  - `Callable[[float], float]`：自定义 curve
- 传入 callable 时，内部使用一个保留 mode 值（例如 `CUSTOM_CURVE = 100`），并把 callable 单独保存在 Python 侧，不塞进 `PopulationConfig` NamedTuple。

---

## 3. Lambda 签名约定

建议第一期采用统一签名：

```python
def curve(ratio: float) -> float:
    ...
```

其中：

- `ratio = actual_competition / expected_competition_strength`
- 返回值为 juvenile scaling factor（非负 float）

这样：

- 与现有 logistic / Beverton-Holt 的输入口径一致
- cfunc 签名简单：`float64(float64) -> float64`
- 用户很容易写出 fixed、线性、凹形、自定义阈值等曲线

如果需要更灵活，后续可扩展为：

```python
def curve(total_juveniles, expected_strength, carrying_capacity, low_density_growth_rate) -> float
```

但第一期建议先用 `ratio -> factor`。

---

## 4. cfunc 桥接机制（基于 P5 探索）

复用 P5 已验证的路线：

1. Python 侧用 Numba `@cfunc` 编译用户 lambda：

   ```python
   @cfunc("float64(float64)")
   def _curve_cfunc(ratio):
       return user_lambda(ratio)
   ```

2. 拿到 `_curve_cfunc.address`，即 C ABI 函数指针。

3. 通过 PyO3 新方法传给 Rust session：

   ```rust
   session.set_density_curve(fn_ptr: usize)
   ```

4. Rust 内部保存为：

   ```rust
   Option<extern "C" fn(f64) -> f64>
   ```

5. 在 `scaling_factor()` 中：

   ```rust
   if let Some(curve) = cfg.density_curve {
       return curve(competition_ratio);
   }
   ```

---

## 5. Rust 侧改动点

| 文件 | 改动 |
|---|---|
| `rust/src/config.rs` | `SimConfig` 增加 `density_curve: Option<extern "C" fn(f64) -> f64>` |
| `rust/src/discrete.rs` | `DiscreteConfig` 增加同样的字段 |
| `rust/src/lifecycle.rs` | `scaling_factor()` 增加 custom curve 分支 |
| `rust/src/discrete.rs` | 同样增加 custom curve 分支 |
| `rust/src/session.rs` | `EngineSession` 增加 `set_density_curve()` |
| `rust/src/discrete_session.rs` | `DiscreteEngineSession` 增加 `set_density_curve()` |
| `rust/src/spatial_session.rs` | 同质/异质空间 session 增加 `set_density_curve()` / config bank 支持 |
| `rust/src/lib.rs` | 注册新方法 |

---

## 6. Python 侧改动点

| 文件 | 改动 |
|---|---|
| `src/natal/configurator/age_structured.py` | `.competition()` 接受 callable |
| `src/natal/configurator/discrete.py` | 同上 |
| `src/natal/engine/backends/rust_backend.py` | 新增 `compile_density_curve_cfunc()`；backend 保存 cfunc 引用并调用 `set_density_curve()` |
| `src/natal/population/age_structured.py` | 创建 Rust backend 时传入 curve |
| `src/natal/population/discrete_generation.py` | 同上 |
| `src/natal/spatial/population.py` | 空间 config bank 支持 per-config curve |
| `src/natal/data/config.py` 或 configurator | 增加 side-channel 存储 callable，不污染 NamedTuple |

---

## 7. 后端选择与回退策略

| 情况 | 行为 |
|---|---|
| 使用 Rust 后端 + custom curve | ✅ 走 cfunc 机器码，性能好 |
| 使用 Numba 后端 + custom curve | ⚠️ Numba nopython 无法直接执行任意 Python lambda，需要回退到纯 Python 路径，或要求启用 Rust |
| 使用纯 Python 后端 + custom curve | ✅ 直接调用 lambda，最灵活但最慢 |

建议：

- `backend="auto"` 时，如果存在 custom curve 且 Rust 可用，优先 Rust；
- 如果强制 `backend="numba"` 且存在 custom curve，则报错或自动回退纯 Python；
- 默认仍保持无 custom curve 时行为完全不变。

---

## 8. 空间模型支持

- 同质空间：所有 deme 共享同一个 curve 指针。
- 异质空间：每个 config bank 条目可以有自己的 curve 指针；`HeterogeneousSpatialEngineSession` 需要按 config index 保存 curve 指针数组。
- 随机/确定性迁移不受影响。

---

## 9. 生命周期管理

- Python 侧必须持有编译后的 `@cfunc` 对象，防止函数指针被 GC。
- 建议放在 `RustLifecycleBackend` / `RustDiscreteLifecycleBackend` / 空间 backend 实例上。
- backend 重建/刷新时，重新编译或复用缓存的 cfunc。

---

## 10. 测试与验收

1. `test_density_curve_cfunc_compile`：lambda 能编译成 cfunc 并调用。
2. `test_age_structured_custom_curve_matches_python`：Rust 自定义 curve 与纯 Python 参考实现逐元素一致。
3. `test_discrete_custom_curve_matches_python`：离散模型同样一致。
4. `test_spatial_custom_curve_homogeneous`：同质空间一致。
5. `test_spatial_custom_curve_heterogeneous`：异质空间不同 deme 不同 curve。
6. `test_backend_auto_uses_rust_for_custom_curve`。
7. 回归：无 custom curve 时所有现有测试不变。

---

## 11. 实施阶段建议

1. **P5-cfunc 最小落地**：实现 `float64(float64)` 的 cfunc 编译与 Rust 调用。
2. **Age-structured 接入**：`SimConfig` + `EngineSession` + `scaling_factor`。
3. **Discrete 接入**：`DiscreteConfig` + `DiscreteEngineSession`。
4. **Configurator API**：`.competition()` 接受 callable。
5. **空间接入**：同质 + 异质。
6. **后端回退策略**：auto / numba / python。
7. **测试与文档**。

---

## 12. 风险与注意点

- lambda 必须是 Numba 可编译的纯数值函数；如果用了不支持的特性，编译期会报错，需要给出清晰提示。
- cfunc 函数指针生命周期必须由 Python 侧持有。
- `SimConfig` / `DiscreteConfig` 增加函数指针对 `Clone` / hash / signature 有一定影响，需要单独处理 backend refresh 签名。
- 与 Numba 的统计等价测试不能直接对比，因为 Numba 路径不支持任意 lambda；参考实现需要走纯 Python。
