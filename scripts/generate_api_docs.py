#!/usr/bin/env python3
"""
从代码docstring生成API参考文档

这个脚本会扫描natal包中的所有模块，提取类、方法和函数的docstring，
生成格式化的Markdown格式的API参考文档。
"""

import ast
import sys
from pathlib import Path
from typing import Any, Dict

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def extract_docstring_from_source(file_path: Path) -> Dict[str, Any]:
    """从源代码文件提取docstring信息"""

    with open(file_path, encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    module_info = {
        'module_name': file_path.stem,
        'file_path': file_path,
        'classes': [],
        'functions': []
    }

    # 提取模块级docstring
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
        module_info['module_docstring'] = tree.body[0].value.s

    for node in tree.body:
        # 类定义
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'methods': []
            }

            # 提取类的方法
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        'name': item.name,
                        'docstring': ast.get_docstring(item),
                        'args': [arg.arg for arg in item.args.args if arg.arg != 'self']
                    }
                    class_info['methods'].append(method_info)

            module_info['classes'].append(class_info)

        # 模块级函数
        elif isinstance(node, ast.FunctionDef):
            function_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'args': [arg.arg for arg in node.args.args]
            }
            module_info['functions'].append(function_info)

    return module_info

def generate_markdown_docs(module_info: Dict[str, Any]) -> str:
    """生成Markdown格式的API文档"""

    content = f"# {module_info['module_name']} 模块API参考\n\n"

    # 模块描述
    if 'module_docstring' in module_info:
        content += f"## 模块概述\n\n{module_info['module_docstring']}\n\n"

    # 类文档
    if module_info['classes']:
        content += "## 类参考\n\n"

        for class_info in module_info['classes']:
            content += f"### {class_info['name']}\n\n"

            if class_info['docstring']:
                content += f"{class_info['docstring']}\n\n"

            # 方法列表
            if class_info['methods']:
                content += "#### 方法\n\n"

                for method in class_info['methods']:
                    content += f"**{method['name']}**\n\n"

                    if method['docstring']:
                        content += f"{method['docstring']}\n\n"

                    if method['args']:
                        content += f"参数: {', '.join(method['args'])}\n\n"

                    content += "---\n\n"

    # 模块级函数
    if module_info['functions']:
        content += "## 函数参考\n\n"

        for func in module_info['functions']:
            content += f"### {func['name']}\n\n"

            if func['docstring']:
                content += f"{func['docstring']}\n\n"

            if func['args']:
                content += f"参数: {', '.join(func['args'])}\n\n"

    return content

def main():
    """主函数"""

    natal_dir = Path(__file__).parent.parent / 'src' / 'natal'
    output_dir = Path(__file__).parent.parent / 'docs' / 'api'

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始扫描natal包中的模块...")

    # 扫描所有Python文件
    python_files = list(natal_dir.glob('*.py'))

    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue

        print(f"处理文件: {py_file.name}")

        module_info = extract_docstring_from_source(py_file)

        if module_info:
            markdown_content = generate_markdown_docs(module_info)

            # 写入文件
            output_file = output_dir / f"{py_file.stem}_api.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"生成文档: {output_file}")

    print("API文档生成完成！")

if __name__ == "__main__":
    main()
