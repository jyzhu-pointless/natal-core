# NATAL Core 文档

**N**umerical **A**ggregation **T**oolkit for **A**nalysis of **L**ifecycles

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Rust](https://img.shields.io/badge/engine-Rust-red.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** 是一个高性能的前向时间群体遗传学模拟引擎，支持可配置的物种生命周期。不同于逐个体模拟，它采用**数值聚合（numerical aggregation）**方法：将个体按年龄、性别、基因型等特征分组计算种群动态，通过概率分布赋予随机性，避免了逐个体遍历的开销。它支持年龄结构化和离散世代种群、精子储存、遗传预设、hook 干预，以及作为唯一执行引擎的 Rust 原生计算核心。尤其适用于**昆虫种群基因驱动（gene drive）建模**，灵活的架构也适用于更广泛的群体遗传学场景。

NATAL Core 是 NATAL 项目的一部分。完整项目还包括 **NATAL Inferencer**，这是一个基于 NATAL Core 的群体遗传学模型参数推断工具包。

## 主要特性

- 🪲 支持前向时间模拟，可灵活配置种群的生命周期（年龄结构化种群与离散世代种群）
- 🧬 可定义遗传结构，包括染色体、基因座和等位基因
- 🚀 数值聚合引擎，Rust 原生加速
- 🧩 内置多种基因驱动预设，特别是 homing drive 和 toxin-antidote drive
- 🪝 提供 Hook 系统，可在模拟过程中插入自定义干预逻辑
- 🔍 配备观察与过滤工具，便于后续分析
- 🗺️ 支持多 deme（亚种群）空间模拟

## 安装

### 1. 创建虚拟环境

强烈建议使用虚拟环境来管理依赖项。

请选择以下命令之一。**推荐使用 Python 3.12**，但任何 Python 版本 >= 3.10 应该都可以工作。

```bash
uv venv --python 3.12 .venv            # uv（推荐）
python -m venv .venv                   # venv（请确保使用 Python >= 3.10）
conda create -n natal-env python=3.12  # conda
```

在 Windows 上，你可以运行 `py -3.12 -m venv .venv` 来指定 Python 3.12 作为虚拟环境的解释器。

### 2. 激活虚拟环境

Linux / macOS：

```bash
source .venv/bin/activate    # uv / venv
conda activate natal-env     # conda
```

Windows：

```powershell
.venv\Scripts\activate       # uv / venv
conda activate natal-env     # conda
```

### 3. 安装 NATAL Core

```bash
uv pip install natal-core
# 或
pip install natal-core
```

## 一个最简示例

```python
import natal as nt
from natal import launch_vue

# 1. 定义物种的遗传架构
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {"loc1": ["WT", "Dr", "R2", "R1"]}
    },
    gamete_labels=["default", "cas9_deposited"]
)

# 2. 使用内置预设定义驱动
drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.9,
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    viability_scaling=1.0,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

# 3. 构建一个随机交配种群，并用 Op 声明释放事件
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="TestPop",
        stochastic=True
    )
    .initial_state(
        individual_count={
            "male": {"WT|WT": 50000}, "female": {"WT|WT": 50000}
        }
    )
    .reproduction(
        eggs_per_female=100
    )
    .competition(
        low_density_growth_rate=6.0,
        carrying_capacity=100000,
        juvenile_growth_mode="beverton_holt"
    )
    .presets(drive)
    .hooks(
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10"),
        event="first",
    )
    .build())

# 4. 启动交互式 WebUI 并运行模拟
launch(pop)
```

更多可即时使用的示例，请参阅 GitHub 仓库中的 [demos](https://github.com/jyzhu-pointless/natal-core/tree/main/demos) 目录。

## 文档目录

推荐先阅读第一部分，上手后以实际项目为驱动阅读第二部分，再根据实际需要选择性阅读第三、四部分。

### 第一部分：快速入门

> NATAL Core 的基本概念和使用方法，帮助你快速上手。

1. [快速入门：15 分钟上手 NATAL](1_quickstart.md)

### 第二部分：实用组件

> NATAL Core 的主要组件，日常使用的核心功能。

2. [遗传结构与实体](2_genetics.md)
3. [种群初始化](2_population_initialization.md)
4. [随机交配种群](2_population.md)
5. [遗传预设使用指南](2_genetic_presets.md)
6. [Hook 系统](2_hooks.md)
7. [模式匹配与可扩展配置](2_genotype_patterns.md)
8. [提取种群模拟数据](2_data_output.md)

### 第三部分：进阶指南

> 空间模拟、自定义配置等高级功能。

9. [空间模拟指南](3_spatial_simulation.md)
10. [运行时参数修改](3_runtime_modification.md)
11. [设计你自己的预设](3_custom_presets.md)
12. [Modifier 机制](3_modifiers.md)
13. [高级 Hook 教程](3_advanced_hooks.md)

### 第四部分：内部实现

> 不直接面向用户的底层实现机制，帮助你深入理解工作原理。


14. [IndexRegistry 索引机制](4_index_registry.md)
15. [PopulationState 与 ModelDraft](4_population_state_config.md)
16. [模拟内核深度解析](4_simulation_engine.md)
17. [Observation 历史记录实现解析](observation_impl.md)


## API 文档

- [完整 API 索引](api/index.md)

## 开发与发布检查

在仓库根目录的开发虚拟环境中运行检查，需要 Python 3.10 或更高版本，以及
Rust 工具链（包括 rustfmt 和 clippy）：

```bash
python -m pip install -e ".[dev]"
python scripts/ci_full.py
```

默认检查 Ruff、Pyright、生成的公开 stub、Python 测试、数值基线、Rust 检查与测试，
以及本次新构建的 wheel。可以选择检查阶段：

```bash
python scripts/ci_full.py --only lint types stubs
python scripts/ci_full.py --only tests
python scripts/ci_full.py --only baseline
python scripts/ci_full.py --only rust
python scripts/ci_full.py --only wheel
```

未知阶段，以及 `--only` 与旧 `--skip-*` 选项混用，都会显式报错。
任一必需阶段失败都会停止执行，并返回非零退出码。
stub 检查不会改写文件；有意变更导出后，使用
`python scripts/generate_init_pyi.py` 重新生成。

本地每次 wheel 构建都使用当前解释器，并在 `rust/target/wheels` 下创建独立输出目录。
`python scripts/build_rust_wheel.py --out PATH` 要求指定目录尚不存在。
本地构建和 CI 都使用 `python scripts/verify_wheel.py --wheel-dir PATH` 验证唯一的 wheel：
包名、版本、解释器与平台兼容性、元数据和原生扩展必须匹配。
验证器会在仓库之外的新临时虚拟环境中安装这个确切的 wheel，检查导入路径和版本，
然后运行已有的复杂遗传、空间种群和运行时更新端到端测试。
安装依赖需要访问包索引；验证不会复用可编辑安装，也不会读取仓库的 pytest 路径配置。

GitHub Actions 调用相同的检查阶段。完整 Python 测试在 Linux 上分别使用
Python 3.10、3.11、3.12、3.13 运行。
数值基线 job 使用 Ubuntu 24.04、Python 3.13 和 `uv.lock` 中的依赖，
只核对现有摘要，不更新基线。要复现其依赖环境，可在独立检出目录或环境中执行
`uv sync --locked --python 3.13`，然后执行
`uv run --no-sync --python 3.13 python scripts/ci_full.py --only baseline`。
本地基线检查使用当前环境，因此调查摘要差异时需要记录环境版本。
更新摘要必须说明科学计算变化的原因，不能用于绕过失败检查。

可复用的 wheel 工作流构建并验证 Python 3.10–3.13 与 Linux x86_64/ARM64、
macOS Intel/ARM64、Windows x86_64 的全部 20 种组合。
只有所有必需 job（包括 wheel 检查）成功，`ci-success` 汇总检查才成功。
新工作流在 GitHub 上运行后，应将它设为 `main` 的必需状态检查；
修改 YAML 本身不会配置分支保护。

`wheels` 发布工作流复用完整 CI，对所选提交执行检查，然后上传同一批已经验证的
wheel 产物，上传时不会重新构建。
`v*` tag 必须与 `pyproject.toml` 和 `natal.__version__` 的版本一致
（按标准版本规范化后比较）。手动运行默认 `dry_run: true`，选择 tag 时也一样。
只有版本 tag 的 push，或显式设置 `dry_run: false` 的手动 tag 运行，才会发布；
分支运行始终不会发布。
PyPI 的可信发布需要配置为使用 `wheels.yml` 工作流及其 `pypi` 环境。
本地通过不代表已验证 GitHub 托管 runner 或 PyPI 权限。

## 链接

- GitHub 仓库：https://github.com/jyzhu-pointless/natal-core
- PyPI 包：https://pypi.org/project/natal-core/

## 许可证

本项目采用 MIT 许可证。
