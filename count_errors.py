#!/usr/bin/env python3
# type: ignore
"""
统计 Pyright 严格模式报告中的错误类型数量。

用法: python count_errors.py pyright_report.txt
"""

import re
import sys
from collections import Counter
from pathlib import Path


def count_pyright_errors(report_path: str) -> Counter:
    """
    解析 Pyright 报告文件，返回每种错误类型的计数。

    Args:
        report_path: Pyright 报告文件路径。

    Returns:
        Counter 对象，键为错误类型（如 "reportUnknownMemberType"），值为出现次数。
    """
    counter = Counter()
    pattern = re.compile(r"\(report(\w+)\)")

    with open(report_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                error_type = match.group(1)
                counter[error_type] += 1

    return counter


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python count_errors.py pyright_report.txt")
        sys.exit(1)

    report_path = sys.argv[1]
    if not Path(report_path).exists():
        print(f"错误: 文件 {report_path} 不存在")
        sys.exit(1)

    counter = count_pyright_errors(report_path)

    print("错误类型统计:")
    print("-" * 40)
    for error_type, count in counter.most_common():
        print(f"{error_type}: {count}")
    print("-" * 40)
    print(f"总计: {sum(counter.values())} 个错误")


if __name__ == "__main__":
    main()
