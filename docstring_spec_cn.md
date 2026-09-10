# Google 风格 Docstring 规范

> [English version](./docstring_spec.md)。中英文同步；含义冲突时以英文版为准。

本规范基于 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)，定义文档和类型标注格式；验证及修复政策统一由 [quality_checks_spec.md](./quality_checks_spec.md) 管理。

## 必需文档

- 为模块、类、公开函数和方法编写文档，包括公开构造方法和属性 getter。
- 简单私有辅助函数的名称、签名和实现足以说明用途，且没有非显而易见的约束或副作用时，可以省略 docstring。复杂私有逻辑及其前提仍须记录，不添加仅复述代码的文字。
- 属性 setter 仅在存在 getter 未说明的行为或约束时补充文档。
- docstring 使用美式英语，包括摘要、描述和示例注释。
- 使用三重双引号，首行为简短摘要，与较长说明之间空一行。
- 当单位、形状、所有权、状态变化和失败行为影响正确使用时，明确说明。

## 字段与格式

按适用性使用以下字段名，不自创替代名称。

| 字段 | 用途 |
|---|---|
| `Args:` | 描述隐式 `self`/`cls` 以外的参数；存在 `*args` 或 `**kwargs` 时一并说明 |
| `Returns:` | 描述返回值；返回 `None` 时省略 |
| `Yields:` | 描述生成值，替代 `Returns:` |
| `Raises:` | 描述调用者需要处理的异常及触发条件 |
| `Attributes:` | 描述公开的类或模块属性 |
| `Examples:` | 用法示例；自包含时优先使用 doctest |
| `Todo:` | 适用时记录后续事项；Sphinx 渲染使用 `sphinx.ext.todo` |
| `Note:` | 额外说明，也允许 `.. note::`、`.. admonition::` 等 reST 指令 |

不得自创 `Parameters:`、`Argument:`、`Return:`、`Author:` 或 `Version:` 字段。元数据放在模块注释或 Sphinx 配置中。

字段内容统一缩进，通常为四个空格；续行对齐，各节适当分隔。允许 reStructuredText，需要时使用 `::` 加缩进代码块。

## 类型标注

函数和方法的参数及返回值须有注解，包括 `__init__ -> None` 和生成器返回类型。隐式 `self`、`cls` 无须重复标注。公开属性通过带注解的声明，或直接赋值给该属性的带注解构造参数，明确其类型。

明显的局部类型允许推断。含义不明确的值、无法推断元素类型的空容器，或类型表达重要约束的值，应补充注解。

新增或修改代码使用 PEP 484 注解。Docstring 中的类型描述可作为补充，但不能替代 strict 类型检查所需的注解。注解已清楚表达类型时，文字不必重复。宽泛类型、cast 和抑制规则按质量规范执行。

## 示例

以下示例可独立运行，展示构造方法、属性、返回值和异常文档。

```python
"""Utilities for proportional scaling."""


class Scale:
    """Scale a value by a positive reference.

    Attributes:
        reference: Positive divisor used for scaling.
    """

    def __init__(self, reference: float) -> None:
        """Initialize the scale.

        Args:
            reference: Positive divisor.

        Raises:
            ValueError: If reference is not positive.
        """
        if not reference > 0:
            raise ValueError("reference must be positive")
        self.reference: float = reference

    def apply(self, value: float) -> float:
        """Return the value relative to the reference.

        Args:
            value: Value to scale.

        Returns:
            The ratio of value to reference.
        """
        return value / self.reference
```

以下简单私有辅助函数不需要重复实现含义的 docstring：

```python
def _is_empty(values: list[int]) -> bool:
    return len(values) == 0
```

## Sphinx 可见性与检查

成员是否显示在 Sphinx 文档中，与源码是否需要文档是不同问题。需要在生成文档中显示这些带文档成员时，在 Sphinx 配置中使用 `napoleon_include_special_with_doc = True` 或 `napoleon_include_private_with_doc = True`。

交付前确认必需 API 和非显而易见的私有逻辑已记录、类型符合标注要求、字段及缩进一致、示例符合实际行为。示例和检查的执行范围按质量规范选择。
