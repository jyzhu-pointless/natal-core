# Google Style Docstring Specification

> [Chinese version](./docstring_spec_cn.md). Keep both versions synchronized; English takes precedence.

Based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). This file defines documentation and annotation format; [quality_checks_spec.md](./quality_checks_spec.md) owns validation and repair policy.

## Required Documentation

- Document modules, classes, and public functions/methods, including public constructors and property getters.
- A simple private helper may omit a docstring when its name, signature, and body make its purpose clear and there are no non-obvious constraints or side effects. Document complex private logic and its assumptions. Do not add text that only repeats the code.
- Property setters need documentation only for behavior or constraints not covered by the getter.
- Write docstrings in English with American spelling, including summaries, descriptions, and example comments.
- Use triple double quotes and a short first-line summary. Separate longer explanations with a blank line.
- Explain relevant units, shapes, ownership, state changes, and failure behavior when these affect correct use.

## Sections and Formatting

Use the following section names where applicable; do not invent alternative names.

| Section | Use |
|---|---|
| `Args:` | Describe parameters other than implicit `self`/`cls`; include `*args` and `**kwargs` when present |
| `Returns:` | Describe a returned value; omit for `None` |
| `Yields:` | Describe generated values instead of using `Returns:` |
| `Raises:` | Describe exceptions callers need to handle and their conditions |
| `Attributes:` | Describe public class or module attributes |
| `Examples:` | Provide usage examples, preferably doctest when self-contained |
| `Todo:` | Record follow-up work when appropriate; Sphinx rendering uses `sphinx.ext.todo` |
| `Note:` | Additional remarks; reST directives such as `.. note::` or `.. admonition::` are also allowed |

Do not use `Parameters:`, `Argument:`, `Return:`, `Author:`, or `Version:` as custom sections. Put metadata in module comments or Sphinx configuration.

Indent section contents consistently, normally four spaces. Align continuation lines and separate sections for readability. reStructuredText is allowed; use `::` with an indented block where needed.

## Type Annotations

Annotate function and method parameters and return values, including `__init__ -> None` and generator return types. Implicit `self` and `cls` need no redundant annotation. Give public attributes explicit types through annotated declarations or constructor parameters assigned directly to those attributes.

Obvious local types may be inferred. Annotate ambiguous values, empty containers whose element type cannot be inferred, or values where the type communicates an important constraint.

Use PEP 484 annotations in new or changed code. Docstring type descriptions may supplement annotations but do not replace annotations required by strict type checking. Do not repeat a type in prose when the annotation already makes it clear. Broad types, casts, and suppressions follow the quality specification.

## Examples

The following example is self-contained and demonstrates constructor, attribute, return, and exception documentation.

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

The following simple private helper needs no redundant docstring:

```python
def _is_empty(values: list[int]) -> bool:
    return len(values) == 0
```

## Sphinx Visibility and Checklist

Sphinx visibility is separate from whether a member needs source documentation. Use `napoleon_include_special_with_doc = True` or `napoleon_include_private_with_doc = True` in Sphinx configuration when those documented members should appear in generated documentation.

Before delivery, check that required APIs and non-obvious private logic are documented, types meet the annotation rules, sections and indentation are consistent, and examples match actual behavior. Follow the quality specification for deciding which examples and checks to run.
