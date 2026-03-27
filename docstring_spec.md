# Google Style Docstring Specification

This specification is based on the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html). It defines the required format for docstrings with two additional mandatory rules:  
1. **Fixed section names** – only the standard section headers may be used.  
2. **All variables must be typed** – either via PEP 484 annotations or explicitly in the docstring.  
3. **Docstring content must be written in English** – including summaries, descriptions, and examples.

---

## 1. General Principles

- Every module, class, and function (including `__init__`) must have a docstring.
- Use triple double quotes `"""`.
- The first line is a brief summary. Leave a blank line before the detailed description.
- reStructuredText formatting is allowed (e.g., for Sphinx).
- Type information should be given via PEP 484 annotations whenever possible. If annotations are used, the docstring can omit the type and only describe the purpose.

---

## 2. Fixed Section Names (Do Not Create Custom Sections)

Only the following standard section headers are allowed. If extra information is needed, use reST directives like `Note:` or `.. admonition::` (which are not custom sections).

| Section        | Applies To              | Description |
|----------------|-------------------------|-------------|
| `Args:`        | Functions, methods      | Describe each parameter. Format: `name (type): description`. If type annotations are used, the `(type)` can be omitted. |
| `Returns:`     | Functions, methods      | Describe the return value. May include type and description. Omit if `None` is returned. |
| `Yields:`      | Generator functions     | Describe the yielded values. Same format as `Returns:`. |
| `Raises:`      | Functions, methods      | List exceptions that may be raised. One per line: `ExceptionType: description`. |
| `Attributes:`  | Classes, modules        | Describe public attributes (class or instance). Format similar to `Args:`. |
| `Examples:`    | Any                     | Provide usage examples, preferably in doctest format. |
| `Todo:`        | Any                     | Record to-do items. Requires the `sphinx.ext.todo` extension. |
| `Note:`        | Any                     | Additional remarks (reST directive, not a custom section). |

### Prohibited
- Do not use variations like `Parameters:`, `Argument:`, `Return:`.
- Do not invent sections like `Author:`, `Version:`, etc. Such metadata should go in module‑level comments or Sphinx configuration.

---

## 3. Type Annotations Requirement

**Every variable (parameters, return values, attributes) must have an explicit type.**  
Two acceptable methods:

### Method 1: Use PEP 484 Type Annotations (Recommended)
Annotate in the signature or variable declaration. The docstring then only describes the purpose, not the type.

```python
def process_data(data: list[int], threshold: float = 0.5) -> bool:
    """Process the input data.

    Args:
        data: List of integers to process.
        threshold: Minimum value to consider.

    Returns:
        True if processing succeeded, False otherwise.
    """
```

### Method 2: Specify Types in the Docstring
If type annotations are not used, the type must be explicitly stated in the docstring section.

```python
def process_data(data, threshold=0.5):
    """Process the input data.

    Args:
        data (list[int]): List of integers to process.
        threshold (float, optional): Minimum value to consider. Defaults to 0.5.

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
```

**Attribute typing examples:**
- Class attributes: document in the `Attributes:` section of the class docstring, or use inline `#: type: description`, or a docstring directly after the attribute.
- Instance attributes: document in the `Attributes:` section of `__init__` (or in the class docstring) and/or use type annotations.

```python
class MyClass:
    class_attr: str = "default"   # type annotation

    def __init__(self):
        self.instance_attr: list[str] = []   # annotated
        """list[str]: This attribute stores processed items."""
```

---

## 4. Docstring Content Must Be in English

All text inside docstrings – including the summary, parameter descriptions, return value descriptions, and example comments – **must be written in English**.  
This ensures consistency and makes the documentation accessible to a wider audience.

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

## 5. Section‑by‑Section Rules

### 5.1 Module Docstring
Placed at the top of the file. Describe the module purpose and optionally list module‑level variables in an `Attributes:` section.

```python
"""Module for geometric calculations.

This module provides basic geometric formulas.

Attributes:
    PI (float): The mathematical constant pi.
"""

PI: float = 3.14159
```

### 5.2 Class Docstring
Placed immediately after the `class` statement. Describe the class and list public attributes in an `Attributes:` section. If the constructor raises exceptions, list them under `Raises:`.

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

### 5.3 Function / Method Docstring
Must include:
- A short summary.
- `Args:` if the function has parameters.
- `Returns:` or `Yields:` if the function returns (or yields) a value.
- `Raises:` if it may raise exceptions.

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

### 5.4 Generator Functions
Use `Yields:` instead of `Returns:`.

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

### 5.5 Property Docstring
Document the property in its getter method. The setter does not need a docstring.

```python
@property
def name(self) -> str:
    """str: The user's full name."""
    return self._name
```

---

## 6. Formatting Details

- Section headers are on their own line, followed by a colon, and then indented content (typically 4 spaces).
- Multi‑line descriptions should be indented to align with the first line.
- Leave a blank line between sections if needed.
- Use `*args` and `**kwargs` explicitly in `Args:`.
- Code blocks are introduced with `::` and indented.

---

## 7. Special and Private Members

- Special members (double underscore) are excluded from documentation by default. To include those with docstrings, set `napoleon_include_special_with_doc = True` in Sphinx’s `conf.py`.
- Private members (single underscore) are excluded by default. To include those with docstrings, set `napoleon_include_private_with_doc = True`.

---

## 8. Checklist

- [ ] Every module, class, and function has a docstring.
- [ ] Only standard section names are used.
- [ ] All parameters, return values, and attributes have explicit types (annotations or docstring‑based).
- [ ] Docstring content is written in English.
- [ ] Sections are formatted correctly with consistent indentation.
- [ ] Examples are in doctest format (where applicable).
- [ ] Special/private members are handled according to project documentation settings.

---

## Appendix: Full Example

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

Following this specification ensures that docstrings are consistent, well‑typed, and ready for automatic documentation generation with tools like Sphinx.
