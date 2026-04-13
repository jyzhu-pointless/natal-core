# Google 风格 Docstring 规范（中文说明）

本文档基于 [Google Python 风格指南](http://google.github.io/styleguide/pyguide.html) 制定，规定了 docstring 的编写规范。
**核心要求**：
1. **字段名固定** – 只能使用标准章节标题，不得自创。
2. **所有变量必须标注类型** – 通过 PEP 484 类型注解或在 docstring 中明确类型。
3. **docstring 内容使用英文** – 摘要、参数描述、返回值描述等均用英文书写。

---

## 1. 总体原则

- 每个模块、类、函数（包括 `__init__`）都必须编写 docstring。
- 使用三重双引号 `"""` 包裹。
- 第一行为简短摘要，与后续内容之间空一行。
- 支持 reStructuredText 格式，可用于生成文档（如 Sphinx）。
- 优先使用 PEP 484 类型注解，若已使用注解，docstring 中可省略类型，仅保留描述。

---

## 2. 固定字段名（不得自创）

docstring 中只允许使用以下标准章节标题。如需补充额外信息，可使用 `Note:` 等 reST 指令（不属于自定义字段）。

| 字段名 | 适用对象 | 说明 |
|--------|----------|------|
| `Args:` | 函数、方法 | 描述参数，每行格式：`name (type): description`。若使用类型注解，可省略 `(type)`。 |
| `Returns:` | 函数、方法 | 描述返回值，可包含类型和说明。若返回 `None` 可省略。 |
| `Yields:` | 生成器函数 | 描述生成的值，用法同 `Returns:`。 |
| `Raises:` | 函数、方法 | 列出可能抛出的异常，每行格式：`ExceptionType: description`。 |
| `Attributes:` | 类、模块 | 描述公共属性（类属性或实例属性），格式同 `Args:`。 |
| `Examples:` | 任意 | 提供使用示例，通常采用 doctest 格式。 |
| `Todo:` | 任意 | 记录待办事项，需启用 `sphinx.ext.todo` 扩展。 |
| `Note:` | 任意 | 额外说明（reST 指令，非自定义字段）。 |

### 禁止行为
- 不得使用 `Parameters:`、`Argument:`、`Return:` 等变体。
- 不得自创如 `Author:`、`Version:` 等字段，此类元数据应放在模块级注释或 Sphinx 配置中。

---

## 3. 类型标注要求

**所有变量（参数、返回值、属性）都必须有明确的类型**。
两种方式任选其一，推荐使用类型注解。

### 方式一：使用 PEP 484 类型注解（推荐）
在函数签名或变量声明处标注类型，docstring 中仅保留描述，不重复类型。

```python
def process_items(items: list[str], limit: int = 10) -> bool:
    """Process the list of items.

    Args:
        items: List of strings to be processed.
        limit: Maximum number of items to handle.

    Returns:
        True if processing succeeded, False otherwise.
    """
```

### 方式二：在 docstring 中标注类型
若未使用注解，则必须在 docstring 的字段中明确类型。

```python
def process_items(items, limit=10):
    """Process the list of items.

    Args:
        items (list[str]): List of strings to be processed.
        limit (int, optional): Maximum number of items to handle. Defaults to 10.

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
```

**属性类型标注示例**：
- 类属性：在类 docstring 的 `Attributes:` 中说明，或在属性声明处使用行内注释 `#: type: description`，或使用紧跟属性的 docstring。
- 实例属性：推荐在 `__init__` 的 docstring 的 `Attributes:` 中集中说明，或使用类型注解直接标注。

```python
class Container:
    """A simple container.

    Attributes:
        name (str): The name of the container.
    """

    name: str = "default"   # type annotation

    def __init__(self):
        self.items: list[str] = []   # annotated
        """list[str]: The stored items."""
```

---

## 4. docstring 内容必须使用英文

所有 docstring 内部的文本（摘要、参数描述、返回值描述、示例注释等）**必须使用英文**。
这保证了文档的一致性，并便于国际读者理解。

```python
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle.

    Args:
        radius: Radius of the circle.

    Returns:
        Area of the circle.
    """
    return 3.14159 * radius * radius
```

---

## 5. 各层级 docstring 详细规范

### 5.1 模块 docstring
位于文件开头，描述模块功能。可包含模块级变量的 `Attributes:` 节。

```python
"""Utilities for string manipulation.

This module provides helper functions for working with strings.

Attributes:
    DEFAULT_SEPARATOR (str): The default separator used by join_with_default.
"""

DEFAULT_SEPARATOR: str = ", "
```

### 5.2 类 docstring
位于 `class` 语句下一行。描述类的作用，并列出公共属性（`Attributes:`）。若 `__init__` 可能抛出异常，可在类 docstring 中添加 `Raises:`。

```python
class Circle:
    """Represent a circle.

    Attributes:
        radius (float): The radius of the circle.
    """

    def __init__(self, radius: float):
        """Initialize a circle.

        Args:
            radius: The radius.
        """
        self.radius = radius
```

### 5.3 函数/方法 docstring
必须包含：
- 简短摘要
- `Args:`（若有参数）
- `Returns:` 或 `Yields:`（若有返回值/生成值）
- `Raises:`（若可能抛出异常）

```python
def divide(dividend: float, divisor: float) -> float:
    """Divide two numbers.

    Args:
        dividend: The number to be divided.
        divisor: The number to divide by.

    Returns:
        The quotient.

    Raises:
        ZeroDivisionError: If divisor is zero.
    """
    if divisor == 0:
        raise ZeroDivisionError("Divisor cannot be zero")
    return dividend / divisor
```

### 5.4 生成器函数
使用 `Yields:` 替代 `Returns:`。

```python
def count_up_to(limit: int):
    """Generate numbers from 0 up to limit-1.

    Args:
        limit: The exclusive upper bound.

    Yields:
        The next integer.
    """
    for i in range(limit):
        yield i
```

### 5.5 属性（property）
在 getter 方法上编写 docstring，描述属性类型和作用。setter 不需要重复。

```python
@property
def name(self) -> str:
    """str: The user's full name."""
    return self._name
```

---

## 6. 格式细节

- 章节标题独占一行，后跟冒号，后续内容缩进（通常 4 个空格）。
- 多行描述应对齐到首行缩进位置。
- 不同章节之间可空一行。
- 若使用 `*args` 和 `**kwargs`，应在 `Args:` 中明确列出。
- 代码块用 `::` 引入，并缩进。

---

## 7. 特殊成员与私有成员

- 特殊成员（双下划线开头结尾）默认不包含在文档中。若希望包含带有 docstring 的特殊成员，可在 Sphinx 的 `conf.py` 中设置 `napoleon_include_special_with_doc = True`。
- 私有成员（单下划线开头）默认不包含。若希望包含带有 docstring 的私有成员，设置 `napoleon_include_private_with_doc = True`。

---

## 8. 检查清单

- [ ] 每个模块、类、函数都有 docstring。
- [ ] 仅使用标准章节标题。
- [ ] 所有参数、返回值、属性都有明确类型（注解或 docstring 中的类型）。
- [ ] docstring 内容全部使用英文。
- [ ] 章节格式正确，缩进一致。
- [ ] 示例采用 doctest 格式（如适用）。
- [ ] 特殊/私有成员按项目文档配置决定是否包含。

---

## 附录：完整示例

```python
"""Utilities for string manipulation.

This module provides helper functions for working with strings.

Attributes:
    DEFAULT_SEPARATOR (str): The default separator used by `join_with_default`.
"""

DEFAULT_SEPARATOR: str = ", "

def join_with_default(words: list[str], separator: str = DEFAULT_SEPARATOR) -> str:
    """Join a list of words using a separator.

    Args:
        words: List of strings to join.
        separator: String used to separate the words. Defaults to DEFAULT_SEPARATOR.

    Returns:
        The concatenated string.

    Examples:
        >>> join_with_default(["a", "b", "c"])
        'a, b, c'
        >>> join_with_default(["a", "b"], "|")
        'a|b'
    """
    return separator.join(words)


class Sentence:
    """A simple sentence container.

    Attributes:
        words (list[str]): The words that form the sentence.
    """

    def __init__(self, words: list[str]):
        """Initialize a sentence.

        Args:
            words: List of words.
        """
        self.words = words

    def to_string(self, separator: str = " ") -> str:
        """Convert the sentence to a string.

        Args:
            separator: Separator between words. Defaults to a space.

        Returns:
            The formatted sentence.
        """
        return separator.join(self.words)
```

遵循本规范可确保代码文档的一致性、可读性以及与 Sphinx 等工具的兼容性。
