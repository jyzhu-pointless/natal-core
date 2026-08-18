# P5 初步探索：Numba `@cfunc` 机器码桥接

> 状态：初步探索完成，已验证端到端可行性。
> 结论：**没有发现根本性障碍**，但完整接入 Rust lifecycle 仍需较多工程工作。

## 1. 背景

P5 的原目标是：用户用 Numba `@cfunc` 生成 C ABI 机器码，把函数指针交给 Rust 调用，从而让 Rust 后端也能执行 custom hook，而不是只能回退 Numba。

当前 Rust 后端只支持 CSR 声明式 hook；custom hook（`@hook(custom=True)` 或带 `(state, config, deme_id)` 参数的 njit 函数）存在时自动回退 Numba。

## 2. 探索内容

### 2.1 Numba `@cfunc` 基础能力

- `@cfunc` 能生成真正的 C ABI 函数，`cfunc.address` 可直接得到函数指针。
- `@cfunc` 的 nopython 模式**不接受 `pyobject` 参数**，因此不能直接把 Python 的 `PopulationState` / `PopulationConfig` 对象传进去。
- 但 `@cfunc` 接受：
  - `CPointer(float64)`、`CPointer(int64)`、`CPointer(boolean)` 等原生指针；
  - 标量 `int64`、`int32`、`float64`、`boolean`；
  - 配合 `carray(ptr, shape)` 在函数体内重建 Numba 数组视图。

### 2.2 可行桥接方案：生成 cfunc 适配器 + 扁平 config buffer

为了让现有用户 hook（`(state, config, deme_id)`）无需改写就能被 Rust 调用，探索采用**生成式适配器**：

1. 保持用户 hook 仍是普通 `@njit` 函数。
2. 为每个 `(hook, config_shape)` 组合生成一个 `@cfunc` 适配器。
3. 适配器 C ABI 固定为：

   ```c
   int64_t hook(
       double *individual_count,
       double *sperm_storage,
       int64_t tick,
       int64_t deme_id,
       double *config_f64,   /* 扁平化 float64 配置 */
       int64_t *config_i64   /* 扁平化 int64/bool/枚举配置 */
   );
   ```

4. 适配器内部用 `carray` + `reshape` 重建：
   - `PopulationState`（`individual_count`、`sperm_storage`）
   - `PopulationConfig`（所有标量 + 数组，从两个扁平 buffer 中按偏移切片）
   - 然后调用原始 `@njit` hook，并把返回的 `0/1` 原样返回。

### 2.3 已验证内容

- 用真实 `AgeStructuredPopulation` 的 42 字段 `PopulationConfig` 生成完整适配器，编译成功。
- 通过 `ctypes` 以 C ABI 调用，hook 能读取 config、修改 state、返回 stop 码。
- 编写了一个极简 Rust `cdylib`（`research/p5_cfunc_bridge/rust_caller`），接收 `usize` 函数指针并按同一 ABI 调用；从 Python 加载后调用 Numba cfunc，**Rust 端调用成功且 state 修改与 ctypes 一致**。

```text
[spike] cfunc adapter address = 0x104ebd8e4
[spike] ctypes result=0 ind[0,0,0]=99.0 expected=99.0
[spike] rust result=0 ind[0,0,0]=99.0 expected=99.0
```

## 3. 代码位置

- `research/p5_cfunc_bridge/spike_flat_abi.py` — 独立 spike，生成适配器并用 ctypes / Rust 调用。
- `research/p5_cfunc_bridge/rust_caller/` — 最小 Rust FFI caller，证明 Rust 可调用 Numba cfunc 指针。

运行方式：

```bash
# 先构建 Rust caller（可选，只验证 ctypes 不需要）
cd research/p5_cfunc_bridge/rust_caller
cargo build --release
cd ../../..

# 运行 spike
python research/p5_cfunc_bridge/spike_flat_abi.py
```

## 4. 风险与待解决问题

| 问题 | 说明 | 影响 |
|---|---|---|
| 配置 ABI 与 shape 绑定 | 适配器生成时把每个数组 shape 和偏移烘焙进代码；config 维度变化需重新生成/缓存 | 中 |
| 可选字段 | 当前 spike 把 `equilibrium_individual_distribution=None` 硬编码；真实运行中该字段可能为数组，需要处理 Optional 变体 | 低 |
| Discrete config | 目前只验证了 `PopulationConfig`；`DiscretePopulationConfig` 字段不同，需要第二套适配器生成器 | 中 |
| 混合优先级 | Rust 侧需要把 custom cfunc 与 CSR hook 按 priority 交错执行，类似 Python `compile_unified_event_hook` | 中 |
| 生命周期/缓存 | 生成的 cfunc 适配器需要像现有 hook codegen 一样缓存；函数指针生命周期需由 Python 侧持有 | 低 |
| 性能 | 每次调用都会重建 config NamedTuple 和若干数组视图；需要 benchmark 确认相比 Numba 回退是否仍有收益 | 待测 |
| GIL/线程 | 原生 cfunc 调用本身不依赖 Python 对象；但若未来 hook 使用 Numba 随机数或 NRT 资源，需确认 Rust 侧无 GIL 调用安全 | 待测 |

## 5. 下一步建议（如果继续做 P5）

1. 在 Python 侧实现 `CfuncHookAdapter` 工厂：
   - 输入 `CompiledHookDescriptor.njit_fn` 与 config/state 类型；
   - 输出可缓存的 `@cfunc` 适配器及其地址。
2. 扩展 Rust `HookProgram` / session：
   - 增加 `Vec<CustomHook>`（event_id、priority、fn_ptr、deme_selector）；
   - 在 `run_tick` / `run_batch` 的 first/early/late 阶段按优先级与 CSR 交错调用。
3. 先接 panmictic age-structured，再接 discrete / WF / spatial。
4. 加 benchmark：小 hook 频繁调用时，cfunc 桥接 vs Numba 回退的延迟差异。
5. 若性能收益不明显，则维持“custom hook 回退 Numba”策略，P5 仅作为可选高级路径。

## 6. 结论

P5 不是天方夜谭：Numba `@cfunc` + 生成式扁平 ABI 适配器可以桥接现有 njit custom hook，并且 Rust 能直接调用该机器码。  
但完整落地涉及适配器生成、缓存、Rust 调度、discrete/spatial 扩展和性能验证，属于**中等偏高工作量**。建议在出现真实用户案例或 benchmark 证明收益后再正式实现。
