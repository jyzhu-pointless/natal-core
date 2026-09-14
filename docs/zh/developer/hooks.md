# Hook 与受控修改

Hook 既可以是可编译的声明式操作，也可以是 Python 回调。两者进入同一事件调度，但回调需要跨 Python/Rust 边界获取受控访问能力。

## 从声明到执行槽

[hooks/_compile.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/_compile.py) 的 `compile_hook_call()` 解析单个调用；`build_hook_program()` 组装执行程序。`HookLayoutContext` 提供布局上下文，选择器根据已发布的索引解析。编译时还处理 priority、身份去重和 deme 范围。

Rust [HookProgram](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/hooks/interpreter.rs) 按同一优先级顺序执行原生计划槽和 Python 回调槽。不能假定“先执行所有声明式 Hook，再执行 Python Hook”；混合注册的先后关系也是合同的一部分。

普通 tick 在 first、early、late 处调用 `execute_event()`；其后的 `EcoCtx.commit()` 让后续阶段读到已提交参数。手动事件、finish 与融合执行模式还有各自入口，不能从普通 tick 推断全部事件行为。

## 回调事务的有效期

[TickContext](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/tick_context.py) 与 [EventTransaction / HookRng](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/hooks/_transaction.py) 把回调访问连接到 [Rust transaction](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/hooks/transaction.rs)。

事务让回调在候选数据上修改，并在成功时提交。状态和参数按需要获取；只写标量的回调不应为此拉取完整个体数量数组。`HookRng` 使用当前受控随机流，不能在回调返回后继续使用；保留上下文或参数句柄也不能绕过有效期和事务路由。

应区分三个边界：一个回调的候选修改、同一事件里多个回调的提交、整个 tick 的阶段计算。后面的回调失败，不会撤销前面已经成功提交的回调；已经发生的生命周期阶段也不是自动恢复到 tick 起点。

例如同一事件的回调 A 成功修改参数，回调 B 随后失败，A 的提交应保留，B 的未提交修改不应泄漏。这个场景在 `test_prior_callback_commit_survives_later_failure` 中有明确断言。

## 标量更新与遗传重编译

[builder/_runtime.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_runtime.py) 中的 `RuntimeUpdater` 把不同更新送入相应写入路径。普通参数写入需要验证候选值；custom 更新先合并、规范化，再刷新原生槽，成功后才更新 Python 草稿和审计记录。

预设和遗传修饰器涉及派生映射。`compile_runtime_candidate()` 在隔离候选上重编译，`commit_genetic_update()` 负责提交；不能直接就地修改一张映射表而保留旧后代张量。运行布局已经发布，重编译还必须尊重当前轴身份。

`reconfigure_preset()` 与追加预设不同：它按其约定从中性 fitness 重放，并在回调事务中登记必要的外部对象恢复动作。编写扩展时要核对这一方法的实际语义，不能假定每种更新都保留先前手动 fitness。

## 验证与修改路线

[test_hook_transaction_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_hook_transaction_contract.py) 包含过期通道、递归运行、先前提交保留、按需数据获取、custom 数组隔离和原生参数更新原子性等场景。

新增 Hook 操作时，沿“Python 声明→编译槽→原生解释器→参数或状态提交→停止/失败→检查点”检查。至少明确：选择器解析时机、与回调的排序、可修改字段、后续阶段可见性，以及失败后保留的边界。参数名称和事件名称相同，并不足以证明两个入口行为一致。
