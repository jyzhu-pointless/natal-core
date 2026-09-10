# 执行引擎

`natal.backends` 包下只有一个适配层：

- `rust` —— 原生 `natal._engine_rs` 扩展的适配层（用 maturin 构建；`rust_backend_available()` 探测它）。它是唯一的执行引擎。

源码生成层已经不存在了：引擎是预编译的原生模块。确定性语义由测试套件对照手工推导的期望值与参考轨迹断言。
