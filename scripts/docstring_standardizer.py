#!/usr/bin/env python3
"""
Docstring Standardizer

This script standardizes all docstrings to English format with consistent structure.
It ensures all public APIs have proper English documentation.
"""

import ast
import re
from pathlib import Path
from typing import Tuple, Optional

# Comprehensive Chinese to English translation mapping
CHINESE_TO_ENGLISH = {
    # Docstring sections
    "参数": "Args",
    "返回": "Returns", 
    "示例": "Example",
    "描述": "Description",
    "默认": "Default",
    "类型": "Type",
    "说明": "Description",
    "功能": "Functionality",
    "用途": "Purpose",
    
    # Common technical terms
    "函数": "Function",
    "方法": "Method", 
    "类": "Class",
    "模块": "Module",
    "属性": "Attribute",
    "变量": "Variable",
    "对象": "Object",
    "实例": "Instance",
    "接口": "Interface",
    
    # Data types and structures
    "字符串": "string",
    "整数": "integer", 
    "浮点数": "float",
    "布尔值": "boolean",
    "列表": "list",
    "字典": "dictionary",
    "数组": "array",
    "元组": "tuple",
    "集合": "set",
    "映射": "mapping",
    
    # Mathematical and statistical terms
    "概率": "probability",
    "分布": "distribution", 
    "参数": "parameter",
    "值": "value",
    "计算": "compute",
    "采样": "sample",
    "均值": "mean",
    "方差": "variance",
    "矩阵": "matrix",
    "向量": "vector",
    "标量": "scalar",
    
    # Genetic terms
    "基因": "gene",
    "等位基因": "allele", 
    "基因型": "genotype",
    "染色体": "chromosome",
    "位点": "locus",
    "重组": "recombination",
    "遗传": "genetic",
    "种群": "population",
    "个体": "individual",
    "配子": "gamete",
    "合子": "zygote",
    
    # Common phrases and connectors
    "如果": "If",
    "否则": "Otherwise",
    "当": "When",
    "则": "Then",
    "例如": "For example",
    "注意": "Note",
    "警告": "Warning",
    "重要": "Important",
    "必须": "Must",
    "可以": "Can",
    "应该": "Should",
    "可能": "May",
    
    # Direction and position
    "父本": "paternal",
    "母本": "maternal", 
    "父系": "paternal",
    "母系": "maternal",
    "雄性": "male",
    "雌性": "female",
    "常染色体": "autosome",
    "性染色体": "sex chromosome",
}

def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def translate_chinese_text(text: str) -> str:
    """Translate Chinese text to English"""
    # First translate common terms
    for chinese, english in CHINESE_TO_ENGLISH.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(chinese) + r'\b', english, text)
    
    # Translate common patterns
    patterns = {
        r'用(.+?)分布连续化(.+?)分布': r'Use \1 distribution to continuousize \2 distribution',
        r'矩匹配：(.+)': r'Moments matching: \1',
        r'均值和方差都是(.+)': r'Mean and variance are both \1',
        r'采样的比例乘以(.+)，得到(.+)': r'Multiply the sampled proportion by \1 to get \2',
        r'当(.+)时': r'When \1',
        r'无法通过(.+)进行有效的(.+)': r'Cannot perform effective \2 through \1',
        r'导致严重的(.+)': r'Leads to severe \1',
        r'故回退到(.+)': r'Therefore fall back to \1',
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def standardize_docstring_format(docstring: str) -> str:
    """Standardize docstring format to English conventions"""
    
    if not docstring:
        return docstring
    
    # Split into lines
    lines = docstring.split('\n')
    standardized_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            standardized_lines.append(line)
            continue
        
        # Standardize section headers
        line = re.sub(r'^参数\s*:$', 'Args:', line)
        line = re.sub(r'^返回\s*:$', 'Returns:', line)
        line = re.sub(r'^示例\s*:$', 'Example:', line)
        line = re.sub(r'^注意\s*:$', 'Note:', line)
        line = re.sub(r'^警告\s*:$', 'Warning:', line)
        
        # Translate Chinese content
        if contains_chinese(line):
            line = translate_chinese_text(line)
        
        standardized_lines.append(line)
    
    return '\n'.join(standardized_lines)

def extract_docstring_info(node) -> Optional[Tuple[int, int, str]]:
    """Extract docstring location and content from AST node"""
    docstring = ast.get_docstring(node)
    if not docstring:
        return None
    
    # Find the line numbers where the docstring starts and ends
    # This is a simplified approach - for precise location we'd need more complex parsing
    return (node.lineno, node.lineno + docstring.count('\n'), docstring)

def process_file(file_path: Path) -> Tuple[bool, str]:
    """Process a Python file and standardize all docstrings"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, f"Syntax error in {file_path}: {e}"
    
    # Track modifications
    modifications = []
    
    # Process module-level docstring
    if (tree.body and 
        isinstance(tree.body[0], ast.Expr) and 
        isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
        
        if isinstance(tree.body[0].value, ast.Constant):
            docstring_value = tree.body[0].value.value
        else:  # ast.Str (Python < 3.8)
            docstring_value = tree.body[0].value.s
            
        if isinstance(docstring_value, str):
            standardized = standardize_docstring_format(docstring_value)
            if standardized != docstring_value:
                modifications.append((f'"""{docstring_value}"""', f'"""{standardized}"""'))
    
    # Process class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                standardized = standardize_docstring_format(docstring)
                if standardized != docstring:
                    modifications.append((f'"""{docstring}"""', f'"""{standardized}"""'))
    
    if modifications:
        # Apply all modifications
        new_content = content
        for old, new in modifications:
            new_content = new_content.replace(old, new)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, f"Standardized {len(modifications)} docstrings in {file_path.name}"
    
    return False, f"No docstrings needed standardization in {file_path.name}"

def main():
    """Main function"""
    
    natal_dir = Path(__file__).parent.parent / 'src' / 'natal'
    
    print("Scanning for docstrings to standardize...")
    
    # Scan all Python files
    python_files = list(natal_dir.glob('*.py'))
    
    files_to_process = []
    
    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue
            
        # Check if file contains Chinese
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if contains_chinese(content):
            files_to_process.append(py_file)
            print(f"  Found Chinese content in {py_file.name}")
    
    if not files_to_process:
        print("All docstrings are already in English format.")
        return
    
    print(f"\nFound {len(files_to_process)} files to standardize:")
    for file_path in files_to_process:
        print(f"  - {file_path.name}")
    
    # Process files
    print("\nStandardizing docstrings...")
    for file_path in files_to_process:
        modified, message = process_file(file_path)
        print(f"  {message}")
    
    print("\nDocstring standardization complete!")

if __name__ == "__main__":
    main()