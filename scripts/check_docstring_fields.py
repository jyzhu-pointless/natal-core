"""Docstring Field Checker - Validates docstring section headers against Google style.

This script checks that docstrings in the src directory use only allowed section headers
as specified in docstring_spec.md.

Allowed sections (per docstring_spec.md):
    - Args:       Functions, methods
    - Returns:    Functions, methods
    - Yields:     Generator functions
    - Raises:     Functions, methods
    - Attributes: Classes, modules
    - Examples:   Any
    - Todo:       Any
    - Note:       Any (reST directive, not a custom section)

Prohibited sections:
    - Parameters:, Argument:, Return:, Author:, Version:, etc.
    - Any custom sections not listed above
"""

import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ALLOWED_SECTIONS = {
    "Args:", "Returns:", "Yields:", "Raises:", "Attributes:", "Examples:", "Todo:", "Note:"
}


PROHIBITED_SECTION_PATTERNS = [
    r"^Parameters:",
    r"^Parameter:",
    r"^Argument:",
    r"^Arguments:",
    r"^Author:",
    r"^Authors:",
    r"^Version:",
    r"^Copyright:",
    r"^Responsibilities:",
    r"^Warning:",
    r"^Warnings:",
    r"^Usage:",
    r"^Description:",
    r"^Summary:",
    r"^See Also:",
    r"^Reference:",
    r"^References:",
    r"^Data:",
    r"^Design Notes:",
    r"^Note::",  # Double colon is reStructuredText style
    r"^Example:",  # Must be "Examples:" (plural) per spec
]


@dataclass
class DocstringIssue:
    file_path: str
    line_number: int
    docstring_type: str
    section_name: str
    message: str
    severity: str = "error"


def get_docstring_type(node: ast.AST) -> str:
    """Determine the type of docstring based on the AST node."""
    if isinstance(node, ast.Module):
        return "module"
    elif isinstance(node, ast.ClassDef):
        return "class"
    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        if node.name.startswith("_") and not node.name.startswith("__"):
            return "private_method"
        elif node.name.startswith("__") and node.name.endswith("__"):
            return "dunder_method"
        else:
            return "function"
    return "unknown"


def extract_sections(docstring: str) -> list[tuple[int, str]]:
    """Extract section headers from a Google-style docstring.

    Returns list of (indent_level, section_name) tuples.
    Only captures lines that appear to be actual section headers,
    not content like exception descriptions or attribute definitions.
    """
    if not docstring:
        return []

    sections = []
    lines = docstring.split('\n')

    for _i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Calculate indent level
        indent = len(line) - len(line.lstrip())

        # Section headers in Google style docstrings:
        # - Start with capital letter
        # - End with colon
        # - Are relatively short (not long descriptions)
        # - Typically at certain indent levels (0, 4, or 8 spaces for docstrings)

        # Must end with colon and be relatively short to be a section header
        if not stripped.endswith(':'):
            continue

        # Section headers should be relatively short (not paragraphs)
        if len(stripped) > 30:
            continue

        # Check if it matches a known section pattern
        # Match "SectionName:" at the start of the line
        match = re.match(r'^([A-Z][a-zA-Z]+):$', stripped)
        if match:
            section_name = match.group(1) + ':'
            sections.append((indent, section_name))

    return sections


def check_section_header(section: str) -> Optional[str]:
    """Check if a section header is allowed or prohibited."""
    for pattern in PROHIBITED_SECTION_PATTERNS:
        if re.match(pattern, section, re.IGNORECASE):
            return f"Prohibited section '{section}' - use allowed sections only"

    if section not in ALLOWED_SECTIONS:
        # Check if it looks like a potential custom section
        if re.match(r'^[A-Z][a-zA-Z]+:', section):
            return f"Non-standard section '{section}' - verify it matches allowed sections"

    return None


def analyze_file(file_path: Path) -> list[DocstringIssue]:
    """Analyze a single Python file for docstring issues."""
    issues = []

    try:
        with open(file_path, encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        issues.append(DocstringIssue(
            file_path=str(file_path),
            line_number=0,
            docstring_type="file",
            section_name="",
            message=f"Failed to read file: {e}",
            severity="error"
        ))
        return issues

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(DocstringIssue(
            file_path=str(file_path),
            line_number=e.lineno or 0,
            docstring_type="file",
            section_name="",
            message=f"Syntax error: {e}",
            severity="error"
        ))
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        docstring_type = get_docstring_type(node)

        # Extract sections from docstring
        sections = extract_sections(docstring)

        # Get line number of the docstring
        if hasattr(node, 'body') and node.body:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                line_number = node.body[0].lineno
            else:
                line_number = node.lineno
        else:
            line_number = node.lineno

        # Check each section
        for _indent, section in sections:
            error = check_section_header(section)
            if error:
                issues.append(DocstringIssue(
                    file_path=str(file_path),
                    line_number=line_number,
                    docstring_type=docstring_type,
                    section_name=section,
                    message=error,
                    severity="error"
                ))

    return issues


def check_directory(src_dir: Path, verbose: bool = False) -> list[DocstringIssue]:
    """Check all Python files in a directory."""
    all_issues = []

    for root, dirs, files in os.walk(src_dir):
        # Skip test directories, virtual environments, etc.
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'venv', '.venv', 'node_modules')]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if verbose:
                    print(f"Checking: {file_path}")

                issues = analyze_file(file_path)
                all_issues.extend(issues)

    return all_issues


def print_report(issues: list[DocstringIssue]) -> None:
    """Print a formatted report of issues found."""
    if not issues:
        print("\n✅ No docstring issues found!")
        return

    print(f"\n❌ Found {len(issues)} issue(s):\n")

    # Group by file
    by_file: dict[str, list[DocstringIssue]] = {}
    for issue in issues:
        if issue.file_path not in by_file:
            by_file[issue.file_path] = []
        by_file[issue.file_path].append(issue)

    for file_path, file_issues in sorted(by_file.items()):
        print(f"📁 {file_path}")
        for issue in file_issues:
            print(f"   Line {issue.line_number} [{issue.docstring_type}]: {issue.section_name}")
            print(f"   → {issue.message}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check docstring section headers for Google style compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="src",
        help="Directory to check (default: src)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--allowed-sections",
        action="store_true",
        help="Show allowed sections and exit"
    )

    args = parser.parse_args()

    if args.allowed_sections:
        print("Allowed sections (Google style):")
        for section in sorted(ALLOWED_SECTIONS):
            print(f"  - {section}")
        return

    src_dir = Path(args.path)
    if not src_dir.exists():
        print(f"Error: Directory '{src_dir}' does not exist")
        sys.exit(1)

    print(f"Checking docstrings in: {src_dir}")

    issues = check_directory(src_dir, verbose=args.verbose)
    print_report(issues)

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
