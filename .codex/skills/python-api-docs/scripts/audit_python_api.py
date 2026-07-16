#!/usr/bin/env python3
"""Statically audit public Python APIs for docs and type annotations.

The audit intentionally uses only the standard library and never imports the
target modules. It is a conservative inventory for documentation work, not a
replacement for repository-specific API decisions.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "site",
    "venv",
}


@dataclass(frozen=True)
class Finding:
    """Describe one statically detected API documentation issue."""

    path: str
    line: int
    symbol: str
    kind: str
    code: str
    message: str


@dataclass
class Summary:
    """Accumulate audit counts."""

    files: int = 0
    parse_errors: int = 0
    modules: int = 0
    classes: int = 0
    functions: int = 0
    methods: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Python files or directories to audit (default: current directory).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="Exclude a path component; may be repeated.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include files below test/tests directories.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "findings", "errors"),
        default="never",
        help="Select which results produce a non-zero exit status.",
    )
    return parser.parse_args(argv)


def iter_python_files(
    roots: Iterable[Path], excludes: set[str], include_tests: bool
) -> Iterable[Path]:
    """Yield unique Python files below the requested roots."""

    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = root.rglob("*.py")
        else:
            continue

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            parts = set(candidate.parts)
            if parts & excludes:
                continue
            if not include_tests and parts & {"test", "tests"}:
                continue
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def literal_exports(tree: ast.Module) -> set[str] | None:
    """Return literal names assigned to ``__all__``, when statically known."""

    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if value is None:
            continue
        try:
            exports = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if isinstance(exports, (list, tuple, set)) and all(
            isinstance(item, str) for item in exports
        ):
            return set(exports)
        return None
    return None


def is_public(name: str, exports: set[str] | None = None) -> bool:
    """Return whether a name is part of the statically visible public surface."""

    if exports is not None:
        return name in exports
    return not name.startswith("_")


def qualified_name(parents: Sequence[str], name: str) -> str:
    """Join a symbol with its lexical parents."""

    return ".".join((*parents, name))


def add_missing_docstring(
    findings: list[Finding], path: Path, node: ast.AST, symbol: str, kind: str
) -> None:
    """Add a missing-docstring finding when needed."""

    if ast.get_docstring(node, clean=False) is None:
        findings.append(
            Finding(
                path=str(path),
                line=getattr(node, "lineno", 1),
                symbol=symbol,
                kind=kind,
                code="DOC001",
                message=f"Public {kind} is missing a docstring.",
            )
        )


def function_annotation_findings(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: str,
    kind: str,
) -> list[Finding]:
    """Return missing-annotation findings for one function or method."""

    findings: list[Finding] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    if kind == "method" and positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]

    parameters = [*positional, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        parameters.append(node.args.vararg)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg)

    for parameter in parameters:
        if parameter.annotation is None:
            findings.append(
                Finding(
                    path=str(path),
                    line=parameter.lineno,
                    symbol=symbol,
                    kind=kind,
                    code="ANN001",
                    message=f"Parameter '{parameter.arg}' is missing a type annotation.",
                )
            )

    if node.returns is None:
        findings.append(
            Finding(
                path=str(path),
                line=node.lineno,
                symbol=symbol,
                kind=kind,
                code="ANN002",
                message="Return value is missing a type annotation.",
            )
        )
    return findings


def audit_class(
    path: Path,
    node: ast.ClassDef,
    parents: Sequence[str],
    findings: list[Finding],
    summary: Summary,
) -> None:
    """Audit a public class and its directly declared public methods."""

    symbol = qualified_name(parents, node.name)
    summary.classes += 1
    add_missing_docstring(findings, path, node, symbol, "class")

    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_public(child.name):
            continue
        method_symbol = f"{symbol}.{child.name}"
        summary.methods += 1
        add_missing_docstring(findings, path, child, method_symbol, "method")
        findings.extend(
            function_annotation_findings(path, child, method_symbol, "method")
        )


def audit_file(path: Path, findings: list[Finding], summary: Summary) -> None:
    """Audit one Python source file without importing it."""

    summary.files += 1
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        summary.parse_errors += 1
        findings.append(
            Finding(
                path=str(path),
                line=getattr(error, "lineno", 1) or 1,
                symbol="<module>",
                kind="module",
                code="PARSE001",
                message=f"Could not parse file: {error}",
            )
        )
        return

    exports = literal_exports(tree)
    module_name = path.stem
    if path.name == "__init__.py":
        module_name = path.parent.name
    summary.modules += 1
    if ast.get_docstring(tree, clean=False) is None:
        findings.append(
            Finding(
                path=str(path),
                line=1,
                symbol=module_name,
                kind="module",
                code="DOC002",
                message="Module is missing a docstring; review whether one adds useful context.",
            )
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_public(node.name, exports):
            audit_class(path, node, [module_name], findings, summary)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(
            node.name, exports
        ):
            symbol = qualified_name([module_name], node.name)
            summary.functions += 1
            add_missing_docstring(findings, path, node, symbol, "function")
            findings.extend(
                function_annotation_findings(path, node, symbol, "function")
            )


def render_text(findings: Sequence[Finding], summary: Summary) -> None:
    """Print a human-readable report."""

    for finding in findings:
        print(
            f"{finding.path}:{finding.line}: {finding.code} "
            f"{finding.symbol}: {finding.message}"
        )
    print(
        "\nSummary: "
        f"{summary.files} files, {summary.modules} modules, "
        f"{summary.classes} classes, {summary.functions} functions, "
        f"{summary.methods} methods, {len(findings)} findings, "
        f"{summary.parse_errors} parse errors"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API audit and return a process exit status."""

    args = parse_args(argv)
    excludes = DEFAULT_EXCLUDES | set(args.exclude)
    roots = [Path(path) for path in args.paths]
    findings: list[Finding] = []
    summary = Summary()

    files = sorted(
        iter_python_files(roots, excludes, args.include_tests),
        key=lambda item: str(item).lower(),
    )
    for path in files:
        audit_file(path, findings, summary)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "summary": asdict(summary) | {"findings": len(findings)},
                    "findings": [asdict(finding) for finding in findings],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        render_text(findings, summary)

    if args.fail_on == "findings" and findings:
        return 1
    if args.fail_on == "errors" and summary.parse_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
