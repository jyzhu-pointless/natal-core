# 模型编译与索引发布

同一生物学类型在完整物种目录和压缩后的运行布局中可能具有不同整数索引。模型构建的关键任务，是让遗传表、初始数量、选择器和结果名称始终使用同一套坐标。

## 先理解 ZType 和 GType

[IndexRegistry](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/registry/index.py) 的 `index_to_ztype` 保存 `(genotype, slab_label)`，`index_to_gtype` 保存 `(haploid_genotype, glab_label)`。因此 ZType 不只是一个裸基因型：体细胞标签也参与身份；GType 同理包含配子标签。

用 Z 表示最终 ZType 数，G 表示最终 GType 数，A 表示年龄层数。常见数组如下；性别顺序为 female、male。

| 草稿字段 | shape | 轴含义 |
| --- | --- | --- |
| `initial_individual_count` | `(2, A, Z)` | 性别、年龄、个体类型 |
| `initial_sperm_storage` | `(A, Z, Z)` | 雌性年龄、雌性类型、存储精子的雄性类型 |
| `zygotes_to_gametes_map` | `(2, Z, G)` | 亲本性别、亲本类型、配子类型 |
| `gametes_to_zygotes_map` | `(G, G, Z)` | 雌性配子、雄性配子、后代类型 |
| `offspring_tensor` | `(Z, Z, Z)` | 雌性亲本、雄性亲本、后代类型 |
| `sexual_selection_fitness` | `(Z, Z)` | 雌性亲本、雄性亲本 |

这里列出的是草稿表示；离散世代会话不保存跨 tick 精子库。不要因为草稿有精子字段，就假定每种运行模型都有相同状态。

## 完整轴上的编译

[compile_definition()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/definition_compiler.py) 要求未发布且完整的物种注册表。它用 `CompileHost` 建立隔离工作副本，从 fitness 基线重新开始，按 priority 排列预设，并在记录的位置应用显式 fitness 步骤；随后追加手动修饰器，从孟德尔基线重建遗传映射。

这一过程的重要约束是每个收集到的修饰器应用一次。失败时不会发布候选结果，并恢复预设的物种绑定。不能把当前已修饰的映射直接当作下一轮基线，否则重编译可能重复施加同一转换。

[build_config_maps()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/assembly.py) 负责默认值、维度验证和完整轴草稿组装。后代张量延迟到最终运行轴上推导，避免先构造完整的 Z³ 张量再丢弃大部分元素。

## 发布是一起换坐标

[publish_products()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/publication.py) 执行以下工作：

1. 选择显式 `IndexProjection`、压缩规划或恒等投影。
2. 验证投影源的数量、类型身份和顺序；尺寸相同不代表坐标相同。
3. 创建新的运行注册表，并通过 `_project_config()` 一起投影初始状态、fitness 和遗传映射。
4. 在最终轴上推导后代张量，或复用已验证的空间遗传模板。
5. 通过 `_validate_runtime_layout()` 检查 shape 和名称，再发布结果。

原编译产物保持未发布，可继续用于隔离构建。`IndexProjection.z_full_to_runtime` 和 `g_full_to_runtime` 用 `-1` 表示删除的类型；不能把这个值直接作为 NumPy 索引使用，因为 NumPy 的 `-1` 指向最后一个元素。

## 压缩保留的不是只有当前非零个体

`plan_projection()` 的种子包括显式声明类型、初始非零个体，以及初始精子存储的雌雄两条类型轴，然后求遗传可达闭包。没有种子时保留完整轴。Builder 还会把能提取的 Hook 类型引用加入显式保留集合。

例如完整目录顺序是 A、B、C，运行时保留 A、C，那么运行索引 1 表示 C。只压缩数量而不压缩后代张量，会把 C 的数量送进 B 的遗传路径；结果可能维度合法却生物学意义错误。

`_build_published()` 因此只在发布后编译 Hook 描述符并构造输出布局。任意 Python 回调的动态行为不能仅靠静态类型引用收集完全推断，显式保留声明仍然有意义。

## 验证入口与修改约束

- [test_publication_contracts.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication_contracts.py) 与 [test_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication.py)：发布、投影和布局合同。
- [test_offspring_single_derivation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_single_derivation.py)：后代张量的推导时机。
- [test_spatial_publication.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_spatial_publication.py)：多个 deme 的共同运行布局。

增加带 Z 或 G 轴的字段时，同时检查投影、shape 验证、名称、物化和运行期重编译。新增字段若遗漏投影，通常不能靠“构建成功”发现。
