#!/usr/bin/env python3
"""
增强版API文档生成脚本

基于inspect模块动态导入模块，提取更详细的参数和类型信息。
"""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def extract_module_info(module_name: str) -> Dict[str, Any]:
    """动态导入模块并提取详细信息"""

    try:
        module = importlib.import_module(f"natal.{module_name}")
    except ImportError as e:
        print(f"无法导入模块 {module_name}: {e}")
        return {}

    module_info = {
        'module_name': module_name,
        'docstring': inspect.getdoc(module),
        'classes': [],
        'functions': []
    }

    # 提取类信息
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == f"natal.{module_name}":
            class_info = extract_class_info(obj)
            module_info['classes'].append(class_info)

    # 提取模块级函数
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == f"natal.{module_name}":
            function_info = extract_function_info(obj)
            module_info['functions'].append(function_info)

    return module_info

def extract_class_info(cls) -> Dict[str, Any]:
    """提取类的详细信息"""

    class_info = {
        'name': cls.__name__,
        'docstring': inspect.getdoc(cls),
        'methods': [],
        'properties': []
    }

    # 提取方法
    for name, method in inspect.getmembers(cls, inspect.ismethod):
        if not name.startswith('_'):
            method_info = extract_function_info(method)
            class_info['methods'].append(method_info)

    # 提取属性
    for name, prop in inspect.getmembers(cls, lambda x: isinstance(x, property)):
        if not name.startswith('_'):
            prop_info = {
                'name': name,
                'docstring': inspect.getdoc(prop)
            }
            class_info['properties'].append(prop_info)

    return class_info

def extract_function_info(func) -> Dict[str, Any]:
    """提取函数的详细信息"""

    # 获取签名信息
    try:
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            param_info = {
                'name': param_name,
                'annotation': str(param.annotation) if param.annotation != param.empty else None,
                'default': param.default if param.default != param.empty else None
            }
            parameters.append(param_info)

        return_info = {
            'annotation': str(sig.return_annotation) if sig.return_annotation != sig.empty else None
        }

    except (ValueError, TypeError):
        parameters = []
        return_info = {}

    return {
        'name': func.__name__,
        'docstring': inspect.getdoc(func),
        'parameters': parameters,
        'return_info': return_info
    }

def generate_enhanced_markdown(module_info: Dict[str, Any]) -> str:
    """生成增强版的Markdown文档"""

    content = f"# {module_info['module_name']} 模块API参考\n\n"

    # 模块概述
    if module_info['docstring']:
        content += f"## 模块概述\n\n{module_info['docstring']}\n\n"

    # 类文档
    if module_info['classes']:
        content += "## 类参考\n\n"

        for class_info in module_info['classes']:
            content += f"### {class_info['name']}\n\n"

            if class_info['docstring']:
                # 清理docstring格式
                docstring = class_info['docstring'].replace('    ', '  ')
                content += f"{docstring}\n\n"

            # 属性
            if class_info['properties']:
                content += "#### 属性\n\n"
                for prop in class_info['properties']:
                    content += f"- **{prop['name']}**"
                    if prop['docstring']:
                        content += f": {prop['docstring']}"
                    content += "\n"
                content += "\n"

            # 方法
            if class_info['methods']:
                content += "#### 方法\n\n"

                for method in class_info['methods']:
                    content += f"**{method['name']}**\n\n"

                    if method['docstring']:
                        content += f"{method['docstring']}\n\n"

                    # 参数信息
                    if method['parameters']:
                        content += "**参数:**\n\n"
                        for param in method['parameters']:
                            content += f"- `{param['name']}`"
                            if param['annotation']:
                                content += f" ({param['annotation']})"
                            if param['default'] is not None:
                                content += f" = {param['default']}"
                            content += "\n"
                        content += "\n"

                    # 返回值信息
                    if method['return_info'].get('annotation'):
                        content += f"**返回:** `{method['return_info']['annotation']}`\n\n"

                    content += "---\n\n"

    # 模块级函数
    if module_info['functions']:
        content += "## 函数参考\n\n"

        for func in module_info['functions']:
            content += f"### {func['name']}\n\n"

            if func['docstring']:
                content += f"{func['docstring']}\n\n"

            # 参数信息
            if func['parameters']:
                content += "**参数:**\n\n"
                for param in func['parameters']:
                    content += f"- `{param['name']}`"
                    if param['annotation']:
                        content += f" ({param['annotation']})"
                    if param['default'] is not None:
                        content += f" = {param['default']}"
                    content += "\n"
                content += "\n"

            # 返回值信息
            if func['return_info'].get('annotation'):
                content += f"**返回:** `{func['return_info']['annotation']}`\n\n"

    return content

def main():
    """主函数"""

    natal_dir = Path(__file__).parent.parent / 'src' / 'natal'
    output_dir = Path(__file__).parent.parent / 'docs' / 'enhanced_api'

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始动态扫描natal包中的模块...")

    # 扫描所有Python文件
    python_files = list(natal_dir.glob('*.py'))

    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue

        module_name = py_file.stem
        print(f"处理模块: {module_name}")

        module_info = extract_module_info(module_name)

        if module_info:
            markdown_content = generate_enhanced_markdown(module_info)

            # 写入文件
            output_file = output_dir / f"{module_name}_api.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"生成增强文档: {output_file}")

    print("增强版API文档生成完成！")

if __name__ == "__main__":
    main()
