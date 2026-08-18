# NATAL Core Rust 后端推进计划

> 分支：`feat/rust-engine-backend`
> 当前提交：`2f1b8a1 feat(rust): add Rust age-structured lifecycle backend`
> 状态：Rust 年龄结构 lifecycle 已完成首个可运行切片，并已接入真实 `AgeStructuredPopulation`。

---

## 1. 当前已完成

- PyO3 + maturin 原生扩展 `natal._engine_rs`。
- Rust `EngineSession`：
  - 年龄结构 reproduction / survival / aging。
- Rust `DiscreteEngineSession`：
  - 离散代 reproduction / survival / aging。
  - Wright-Fisher fused tick（mode 1/2/3）。
  - 批量 `run()` 与 observation history 记录。
  - 批量 `run()`：多个 tick 在 Rust 内循环。
  - CSR declarative hook（10 个 opcode、RPN 条件、deme selector、stop 语义）。
  - 内核内 raw history 与 observation mask 记录。
- Python 侧 `RustLifecycleBackend`。
- `AgeStructuredPopulation` 显式接入：
  - `enable_rust_backend(seed=...)` / `disable_rust_backend()`。
  - custom hook 存在时拒绝启用或自动回退 Numba。
- 测试：
  - 确定性路径与 Numba 逐元素精确一致。
  - stochastic 路径统计等价。
  - 真实 Population 的 raw / observation history 精确一致。
- 工具：
  - `scripts/check_rust.py`：cargo fmt + clippy + check 硬门禁，rust-analyzer 可选门禁。
  - demo 与大小两个 benchmark。

---

## 2. 后续方向总览

| 优先级 | 方向 | 价值 | 风险/成本 |
|---|---|---|---|
| P0 | 离散代 + Wright-Fisher lifecycle | 补全引擎覆盖面，让第二个 Population 模型也能走 Rust | 中 |
| P1 | `backend="auto"` 与 config 同步 | 让用户无需手动 enable，消除运行时 config 快照过期问题 | 中 |
| P2 | Rust 性能收尾 | 去掉 history 中间复制、优化小模型路径、建立 CI benchmark 基线 | 低 |
| P3 | 打包与 CI | `pip install .` 能自动构建 Rust，形成真实可发布形态 | 中 |
| P4 | 空间模型 | Rayon 并行 deme、异质 config bank、迁移内核 | 高 |
| P5 | custom hook 机器码桥接（研究） | `@cfunc` 函数指针供 Rust 调用 | 高，收益不明确 |
| P6 | GPU 后端抽象 | 为 wgpu/CUDA 预留接口 | 高，现阶段不建议投入 |

---

## 3. P0：离散代 + Wright-Fisher lifecycle（核心已完成，待收尾）

### 为什么先做这个

- 年龄结构已经证明整条链路可行。
- 离散代没有储精数组，内核更简单，适合快速复制同样的集成模式。
- Wright-Fisher 是融合 tick（只执行 first hook），可以和离散代一起完成。
- 完成后，核心的 panmictic 引擎就全部有 Rust 路径。

### 工作内容

1. Rust 侧新增 `DiscreteConfig`（已完成）：
   - 从 `DiscretePopulationConfig` 提取标量生存率、交配率、繁殖率等字段。
   - 复用现有 `offsprung_tensor`、fitness、compatibility 数组。
2. Rust 侧实现：
   - `discrete_reproduction`（`mate_discrete` + `fertilize_discrete`）。
   - `discrete_survival`（密度调节 + viability）。
   - `discrete_aging`（0 龄 → 1 龄）。
   - `run_discrete_tick`：first → reproduction → early → survival → late → aging。
   - `run_wf_tick`：first hook 后融合繁殖/存活/衰老，不执行 early/late。
3. PyO3：
   - `DiscreteEngineSession` 或给现有 `EngineSession` 增加 discrete 方法。
   - 批量 `run()` 与 observation history 复用同一套行布局（离散无 sperm）。
4. Python 接入：
   - `DiscreteGenerationPopulation.enable_rust_backend()`。
   - custom hook 同样自动回退 Numba。
5. 测试与验收：
   - 确定性模式与 Numba 逐元素一致。
   - stochastic 模式统计等价。
   - `test_discrete_population.py`、`test_wright_fisher.py` 的 Rust 对照用例。
   - 扩展 integration test：真实 `DiscreteGenerationPopulation` 状态与 history 完全一致。

### 验收标准

- `cargo fmt/clippy/check` 通过。
- `pytest` 全量通过。
- 新增 discrete / WF 对照测试全部通过。
- release benchmark 至少不慢于 Numba；小模型单 tick 应显著快于 Numba。

---

## 4. P1：`backend="auto"` 与 config 同步（已完成）

### 目标

把“手动 enable”升级为可配置、可自动回退的用户入口。

### 工作内容

1. 在 `Population.setup(..., backend="auto")` 增加参数（已完成）：
   - `"auto"`：有 Rust 扩展且无 custom hook → Rust，否则 Numba。
   - `"rust"`：强制 Rust；custom hook 存在时启动时报错。
   - `"numba"`：默认，保留旧路径。
   - `"python"`：强制纯 Python fallback（已接入）。
2. config 同步（已完成基础版）：
   - `pop.update()` 后在下一次 run 前根据 config/hook 全字段哈希自动重建 session。
   - 提供显式 `refresh_rust_backend()`。
3. 行为不变性：
   - 默认值保持 `"auto"` 之前的行为？**建议先保持显式 enable 为默认**，避免大范围行为变化；确认无回归后再切换默认。
4. 测试：
   - 每种 backend 组合下状态机、history、hook 回退测试。
   - `pop.update()` 后 Rust 路径参数确实生效。

### 验收标准

- 轴组合测试覆盖 `backend × stochastic × history mode`。
- `using_rust_backend` 与实际路径一致。
- 无 custom hook 时 auto 选择 Rust；有 custom hook 时 auto 选择 Numba。

---

## 5. P2：Rust 性能收尾（进行中）

当前 benchmark 已显示：

- 大模型：Rust 与 Numba 批量路径接近，单 tick 快 4–8 倍。
- 小模型：Rust 批量路径快 16 倍，单 tick 快约 200 倍（deterministic）。

仍可做：

1. 去掉 `Vec<Vec<f64>>` → `PyArray2` 的中间历史复制（已完成）。
2. `run_tick()` 单 tick 路径提供显式零拷贝入口 `run_tick_inplace()`（已完成；默认安全路径仍复制）。
3. 对 reproduction 的 `n_ztypes²` 层增加零格早停，进一步利用 stochastic 稀疏性。
4. 在 CI 中保存 release benchmark 基线，防止性能回退。

### 验收标准

- benchmark 有可复现数值。
- history 记录路径性能不退化。
- 不牺牲确定性逐元素一致性。

---

## 6. P3：打包与 CI（本地部分完成）

### 工作内容

1. 混合构建方案（已完成）：
   - 已切换到 maturin build backend；`pip install .` 现在会直接构建并安装 Rust 扩展。
   - 新增 `scripts/build_rust_wheel.py` 用于显式构建并校验 wheel。
2. CI workflow：
   - 已新增 `scripts/ci_full.py`，一条命令执行 Rust 门禁 + pytest + pyright + ruff + wheel 构建。
   - `.github/workflows` 仍未落盘（当前 `.gitignore` 忽略 `.github/*`，需用户允许取消忽略后再提交 workflow）。
3. 修复 rust-analyzer 可选门禁的环境依赖：
   - `rustup component add rust-src`。
   - 或移除 `.vscode/settings.json` 的 Homebrew 临时路径。

### 验收标准

- 干净环境按文档可一键构建。
- `pip install .` 或发布 wheel 包含 `natal._engine_rs`。
- 无 Rust 工具链环境仍可安装并自动回退 Numba（如果选择方案 B）。

---

## 7. P4：空间模型（第一个切片已完成）

用户已明确空间不急，但后续方案如下：

1. Rust `SpatialSession`（第一版已完成）：
   - `SpatialEngineSession` + `RustSpatialLifecycleBackend`。
   - deme 堆叠状态 buffer，按 deme 循环执行现有 lifecycle。
   - 每 deme 独立 seed = base + deme。
   - 已接入 Rayon 并行 per-deme lifecycle；迁移尚未接入。
2. 异质 config：
   - `config_bank` + `deme_config_ids`。
   - 更新时显式 copy-on-write。
3. 迁移：
   - adjacency 与 kernel 两种模式。
   - 迁移阶段与 lifecycle 阶段分离，保持同步。
4. 测试：
   - 与现有 spatial Numba 结果确定性精确一致。
   - stochastic 统计等价。
   - 大规模 deme benchmark，评估 Rayon 收益。

---

## 8. P5：初步探索完成（暂不正式实现）

- 目标：用户用 Numba `@cfunc` 生成 C ABI 机器码，把函数指针交给 Rust 调用。
- 已完成端到端可行性验证：
  - `@cfunc` 可暴露 C ABI 函数指针；
  - 生成式适配器可用扁平 `float64*` / `int64*` config buffer 重建 `PopulationState` / `PopulationConfig`；
  - Rust `cdylib` 能直接调用 Numba cfunc 指针并修改 state。
- 代码与详情见 `P5_EXPLORATION.md` 和 `research/p5_cfunc_bridge/`。
- 仍不正式实现：完整落地需要适配器生成/缓存、Rust 调度、discrete/spatial 扩展、性能验证，收益待 benchmark 确认。

---

## 9. P6：GPU 后端（长期，不启动）

- 当前 CPU 尚未完全覆盖，GPU 投入为时过早。
- 等空间大规模 deme benchmark 显示 CPU 瓶颈后再启动。
- 启动时先做抽象接口，不做具体 GPU kernel。

---

## 10. 当前不建议做的事

- 不要重写 Python 前端（Configurator / Species / patterns）。
- 不要移除 Numba 路径，它是回退与 golden reference。
- 不要逐位复现 NumPy legacy RNG。
- 不要为 GPU 预先写 kernel。

---

## 11. 下一步具体行动（一句话）

> 按 P0 开始实现离散代与 Wright-Fisher 的 Rust lifecycle，并以 `DiscreteGenerationPopulation` 的真实 integration test 作为验收。
