# Changelog

## 2026.3.18
- 完善 HomingModificationDrive 逻辑
- 修复了配子产生的默认逻辑，此前会产生各种 gamete label 的配子（且每一种归一化），现在只产生默认 label（"default" 或 index 0）
- 修复了 hooks 的 bug（响应式 hook 被注册但未被使用）
- 临时移除 `algorithms.fertilize_with_mating_genotype` 中对 `P` 的归一化，以支持致死效应的模拟（需后续检查正确性）

## 2026.3.14
- 优化种群初始化方式
- 加入高级转换规则和 recipe 支持
- 引入基因型 pattern 解析器
