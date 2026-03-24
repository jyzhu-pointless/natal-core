# Changelog

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
