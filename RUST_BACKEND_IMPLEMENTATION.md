# Rust 后端实现现状与使用说明（索引）

> 本页原先的实现说明已移除：它写成于 Numba 仍是默认后端的时期，其中的模块路径
> （`natal.engine.backends.*`）、后端选择开关（`backend=`、`enable_rust_backend`）、
> 会话类名、opcode 数量与性能数字均已与现状不符。旧正文仍可在 git 历史中查阅：
> `git show 9ab7683:RUST_BACKEND_IMPLEMENTATION.md`。

## 现状要点

- Rust 扩展 `natal._engine_rs`（PyO3 + maturin）是**唯一**执行后端；扩展缺失或与
  解释器不匹配时构造报错，不再回退 Numba 或纯 Python。
- 会话类：`EngineSession`、`DiscreteEngineSession`、
  `HeterogeneousSpatialEngineSession`。
- NumPy 数组跨 FFI 是**拷贝**语义（`set_state` 接收拥有型 `Vec<f64>`），不是零拷贝视图。
- 每 deme 的随机流种子由 `seed ^ deme_id` 导出（异或，不是相加）。
- 密度调控的自定义曲线槽位（`juvenile_growth_mode >= 5`）仍是**预留、未实现**，
  见 [DENSITY_CURVE_PLAN.md](DENSITY_CURVE_PLAN.md)。

## 维护中的文档

- 总体运行模型与生命周期：[docs/en/4_simulation_engine.md](docs/en/4_simulation_engine.md)
- 种群配置与状态：[docs/en/4_population_state_config.md](docs/en/4_population_state_config.md)
- Hook 系统：[docs/en/2_hooks.md](docs/en/2_hooks.md)
- 空间模拟与迁移：[docs/en/3_spatial_simulation.md](docs/en/3_spatial_simulation.md)
- API 参考：[docs/en/api/index.md](docs/en/api/index.md)

## 自行构建与门禁

- Rust 门禁（fmt / clippy / check / 单元测试）：`python scripts/check_rust.py`
- 构建 wheel：`scripts/build_rust_wheel.py`，CI 见 `.github/workflows/wheel-build.yml`
