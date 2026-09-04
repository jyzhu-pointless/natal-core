# 执行后端

`natal` 引擎在 `natal.backends` 下按后端族提供一套实现：

- `reference` —— 纯 Python 黄金实现（始终可用）。
- `rust` —— 原生 `natal._engine_rs` 扩展的适配层（用 maturin 构建；`rust_backend_available()` 探测它）。

源码生成层已经不存在了：参考内核是普通 NumPy 函数，Rust 引擎是预编译的原生模块。确定性语义由测试套件对照参考实现断言；空间模型的非 Rust 执行路径只有按 deme 的 Python 分发。
