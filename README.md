# ⚡ NATAL
**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles

[![NATAL](https://img.shields.io/badge/NATAL-0.1.0-purple.svg)](https://github.com/jyzhu-pointless/natal-core)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.2+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.64.0+-orange.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

![logo](./natal-brand.svg)

一个用于模拟种群遗传学动态的 Python 工具箱。可根据物种的生命周期建模，支持年龄结构种群、精子存储机制、遗传修饰器以及 Numba 加速的矩阵计算。

> **⚡️ NATAL 生态系统**：本项目包含两部分：
> - **🧬 NATAL Core** - 种群遗传学模拟引擎 **（本仓库）**
> - **📊 NATAL Inferencer** - 参数推断模块

---

**⭐ [15 分钟快速上手](docs/01_quickstart.md)**

## Development Notes

如果你修改了包级 public API，比如新增、删除或重命名某个模块里的 `__all__` 导出，运行下面的命令即可重新生成用于编辑器补全和语法高亮的 stub 文件：

```bash
python scripts/generate_init_pyi.py
```

这会更新 [src/natal/__init__.pyi](src/natal/__init__.pyi)，而运行时的懒加载逻辑仍然由 [src/natal/__init__.py](src/natal/__init__.py) 负责。
