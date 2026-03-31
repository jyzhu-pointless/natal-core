#!/usr/bin/env python3
"""
Normalize docstrings to English format

This script scans Python files and converts Chinese docstrings to English.
It focuses on standardizing the format while preserving technical content.
"""

import ast
import re
from pathlib import Path
from typing import Tuple

# Chinese to English translation mapping for common terms
CHINESE_TO_ENGLISH = {
    # Common parameter descriptions
    "参数": "Args",
    "返回": "Returns",
    "示例": "Example",
    "描述": "Description",
    "默认": "Default",
    "类型": "Type",

    # Common technical terms
    "函数": "Function",
    "方法": "Method",
    "类": "Class",
    "模块": "Module",
    "属性": "Attribute",
    "变量": "Variable",

    # Data types
    "字符串": "string",
    "整数": "integer",
    "浮点数": "float",
    "布尔值": "boolean",
    "列表": "list",
    "字典": "dictionary",
    "数组": "array",

    # Common phrases
    "如果": "If",
    "否则": "Otherwise",
    "当": "When",
    "则": "Then",
    "例如": "For example",
    "注意": "Note",
    "警告": "Warning",

    # Mathematical terms
    "概率": "probability",
    "分布": "distribution",
    "值": "value",
    "计算": "compute",
    "采样": "sample",
}

def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def translate_chinese_terms(text: str) -> str:
    """Translate common Chinese terms to English"""
    for chinese, english in CHINESE_TO_ENGLISH.items():
        text = text.replace(chinese, english)
    return text

def normalize_docstring(docstring: str) -> str:
    """Normalize a docstring to English format"""
    if not docstring or not contains_chinese(docstring):
        return docstring

    # Basic translation of common terms
    normalized = translate_chinese_terms(docstring)

    # Standardize format markers
    normalized = re.sub(r'(参数|参数说明)\s*:', 'Args:', normalized)
    normalized = re.sub(r'(返回|返回值)\s*:', 'Returns:', normalized)
    normalized = re.sub(r'(示例|例子)\s*:', 'Examples:', normalized)

    return normalized

def process_file(file_path: Path) -> Tuple[bool, str]:
    """Process a Python file and normalize docstrings"""

    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, f"Syntax error in {file_path}"

    modified = False
    new_content = content

    # Process module-level docstring
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            docstring = tree.body[0].value.value
            if contains_chinese(docstring):
                normalized = normalize_docstring(docstring)
                if normalized != docstring:
                    # Replace the docstring in the original content
                    original_docstring = f'"""{docstring}"""'
                    new_docstring = f'"""{normalized}"""'
                    new_content = new_content.replace(original_docstring, new_docstring, 1)
                    modified = True

    # Process class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring and contains_chinese(docstring):
                normalized = normalize_docstring(docstring)
                if normalized != docstring:
                    # Find and replace the docstring in original content
                    # This is a simplified approach - for production use, we'd need more precise location tracking
                    original_docstring = f'"""{docstring}"""'
                    new_docstring = f'"""{normalized}"""'
                    new_content = new_content.replace(original_docstring, new_docstring)
                    modified = True

    if modified:
        # Write the modified content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, f"Updated docstrings in {file_path}"

    return False, f"No Chinese docstrings found in {file_path}"

def main():
    """Main function"""

    natal_dir = Path(__file__).parent.parent / 'src' / 'natal'

    print("Scanning for Chinese docstrings...")

    # Scan all Python files
    python_files = list(natal_dir.glob('*.py'))

    files_with_chinese = []

    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue

        print(f"Checking: {py_file.name}")

        # First, just detect Chinese content
        with open(py_file, encoding='utf-8') as f:
            content = f.read()

        if contains_chinese(content):
            files_with_chinese.append(py_file)
            print(f"  Found Chinese content in {py_file.name}")

    if not files_with_chinese:
        print("No files with Chinese content found.")
        return

    print(f"\nFound {len(files_with_chinese)} files with Chinese content:")
    for file_path in files_with_chinese:
        print(f"  - {file_path.name}")

    # Ask for confirmation before modifying
    response = input("\nDo you want to normalize these docstrings to English? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return

    # Process files
    print("\nNormalizing docstrings...")
    for file_path in files_with_chinese:
        modified, message = process_file(file_path)
        print(f"  {message}")

    print("\nDocstring normalization complete!")

if __name__ == "__main__":
    main()
