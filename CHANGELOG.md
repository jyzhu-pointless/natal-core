# Changelog

## 2026.3.26.d
- 修复了 algorithms 中未正确处理配子储存与替换的问题
- 为 algorithms 增加概率范围限制和异常报错
- 为 algorithms 支持完整的 mating rates
- 增加 numba 随机种子设置方法
- 优化 UI 界面

## 2026.3.26.c
- 重命名 `hook/` -> `hooks/`，并同步更新相关导入路径
- 重命名 `kernel/` -> `kernels/`，并同步更新相关导入路径

## 2026.3.26.b
- 移除 `simulation_kernels.py` 中未使用的 `run/run_tick` 旧入口，统一走 hook codegen 生成的 wrapper
- 将 kernel wrapper codegen 从 `hook/compiler.py` 拆分到 `natal/kernel/codegen.py`，并将模板外置到 `src/natal/kernel/templates/kernel_wrappers.py.tmpl`
- 将核心 `simulation_kernels` 实现迁移到 `natal/kernel/simulation_kernels.py`，保留 `natal/simulation_kernels.py` 兼容转发
- 包结构重命名：`natal/hook` -> `natal/hooks`，`natal/kernel` -> `natal/kernels`，并同步更新内部导入与模板引用

## 2026.3.26
- 修复 `HomingDrive` 中 maternal deposition 总是有效的 bug
- 增加 spatial model 的一些模块

## 2026.3.25
- 继续根据类型检查进行了一些修改，当前版本可用
- 修改 UI title 和 favicon

## 2026.3.24
- 根据类型检查进行了一些修改，当前版本可用

## 2026.3.23
- 创建 webui
- 部分概念重命名：`IndexCore` -> `IndexRegistry`，`recipe` -> `preset`
- 恢复 `AgeStructuredPopulation` 的生存率、初始状态构建逻辑
- 修复了 `gamete_allele_conversion.py` 中 `sex_filter` 无效的问题（现在可以识别字符串和数字表示的性别，且无效的表示会报错）
- 优化文档

## 2026.3.21
- 重构 hook，优化 hook 缓存

## 2026.3.20
- 移除 `jitclass`，`PopulationState`, `DiscretePopulationState` 和 `PopulationConfig` 全部改为用 `NamedTuple`
- 优化 `njit` 缓存
- 增加编译提示信息

## 2026.3.18.b
- Hook 执行内核拆分：将 CSR 执行逻辑抽为外部函数 `execute_csr_event_arrays`，统一以 `HookProgram` 作为数据载体
- 新增 `HookProgram` 与 `build_hook_program` 作为非 jitclass 路径的数据接口（第一阶段）
- 完全移除 `HookRegistry` 符号，核心内核统一走 `HookProgram + execute_csr_event_program`
- 优化 Declarative `when` 表达式的解析，支持 `and`、`or`、`not` 等逻辑运算符，增强表达能力
- 为 `Op.scale` / `Op.subtract` / `Op.sample` / `Op.set_count` 增加抽样逻辑

## 2026.3.18
- 完善 HomingModificationDrive 逻辑
- 修复了配子产生的默认逻辑，此前会产生各种 gamete label 的配子（且每一种归一化），现在只产生默认 label（"default" 或 index 0）
- 修复了 hooks 的 bug（响应式 hook 被注册但未被使用）
- 临时移除 `algorithms.fertilize_with_mating_genotype` 中对 `P` 的归一化，以支持致死效应的模拟（需后续检查正确性）

## 2026.3.14
- 优化种群初始化方式
- 加入高级转换规则和 recipe 支持
- 引入基因型 pattern 解析器
