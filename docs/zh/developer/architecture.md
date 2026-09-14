# 架构与数据流

需要先建立完整认识时，参见[一个模型从声明到结果的完整旅程](model_journey.md)，其中使用同一个模型追踪各层输入与输出。

从一次构建到一次观测，最重要的是区分三个阶段：用户描述模型，Python 编译并发布数值布局，Rust 会话执行并记录。`frontend` 在 Python 包中主要表示模型前端；浏览器界面另有仓库根目录的 `frontend/`，不要把两者混为一层。

## 从入口走到结果

下面是职责流程，不是逐行调用栈。具体函数见[模型编译](model.md)和[会话执行](runtime.md)。

```text
Species / PopulationBuilder
    → ModelDefinition + ModelDraft
    → compile_definition → CompiledProducts（完整索引）
    → publish_products（最终运行时索引）
    → Hook / Observation / RecordingPlan
    → materialize → Blueprint + Params
    → Rust backend → native session
    → lifecycle kernels → native history / observation
    → Python 查询结果
```

[PopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_base.py) 把入口拆成 `_compile_products()`、`_publish_and_build()` 和 `_build_published()`。这使完整轴上的编译产物与最终运行模型之间有明确边界：Hook 选择器和观测布局依赖最终索引，不能提前绑定到压缩前的坐标。

## 谁持有什么

| 层 | 具体对象或模块 | 持有内容与约束 |
| --- | --- | --- |
| 声明 | `ModelDefinition` | 保留可重编译的声明；编译使用隔离工作副本 |
| 构建 | `ModelDraft`、`CompiledProducts` | 数组、索引、修饰器等候选产物；尚不是运行状态的权威来源 |
| 发布 | `IndexProjection`、已发布的 `IndexRegistry` | 固定运行时坐标，将所有相关轴一起投影 |
| 传输合同 | `Blueprint`、`Params` | Python 到 Rust 的字段、维度和数值表示 |
| 执行 | Rust `sessions/` | 当前数量、tick、执行状态、RNG，以及运行参数 |
| 输出 | Rust `output/` 与 Python 包装 | 历史行、投影、参数日志及查询元数据 |

[materialize()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/contracts/materialize.py) 将已构建草稿拆成合同对象，并为数组建立新的所有权。Rust 从合同创建自身持有的数据。不要据此推断每条更新路径都复制整个模型：`materialize_params()` 跳过 Blueprint，`contract_field_source()` 只投影被请求的字段，其连续数组转换也不保证总会复制。

## 读与写的方向不同

已有会话时，`pop.state` 和 `pop.config` 是 Python 查询快照。修改其中的数组不能作为修改引擎的机制。`RustDiscreteLifecycleBackend.state_snapshot()` 和 `config_snapshot_from_session()` 是查看快照如何重建的入口。

运行期写入经参数通道或更新器进入原生会话；回调内写入还受事务约束。Python 中保留的声明用于重建与解释，不能靠它推断原生会话的最新值。这也是参数读取要经过后端、而不能一律读取旧草稿的原因。

空间种群把执行所有权进一步集中到一个堆叠会话。deme 的 Python 对象是访问局部数据的入口，不能独立推进共享时间线，详见[空间执行](spatial.md)。

## 阅读与验证入口

- [rust_backend.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py)：合同转换、会话调用、快照与参数写入的适配层。
- [Rust model](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/model/mod.rs)：原生模型的数据组织；与 Python 合同对照阅读。
- [test_contracts_materialize.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_contracts_materialize.py)：字段映射、名称目录、custom 值、数组隔离与仅参数物化。
- [test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py)：原生字段的新值、回调通道与空间执行所有权。

改变一份数据的所有者时，要沿“构建→传输→执行→快照→恢复”检查所有路径。只检查正常运行是否得到相同结果，无法证明共享与恢复行为仍然正确。
