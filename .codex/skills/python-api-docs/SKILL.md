---
name: python-api-docs
description: Prepare and polish open-source Python repositories whose documentation is built with MkDocs and mkdocstrings. Use when defining or reviewing a Python public API, adding or correcting type annotations and docstrings, improving explanatory code comments, creating runnable examples or demos, generating API-reference pages, configuring MkDocs/mkdocstrings, or validating that code, examples, and strict documentation builds stay synchronized before an open-source release.
---

# Python API Docs

Build documentation around a deliberate, stable public API. Treat docstrings, type annotations, examples, and generated API pages as one tested interface rather than separate writing tasks.

## Select the work mode

Choose the smallest mode that satisfies the request:

- **Audit**: inspect the API, docstrings, comments, demos, and documentation build; report findings without editing.
- **API pass**: clarify exports, signatures, annotations, docstrings, and comments without introducing a documentation site.
- **Demo pass**: create or repair runnable, user-oriented examples and connect them to the docs.
- **Docs setup**: create or update MkDocs + mkdocstrings around an already coherent API.
- **Release-ready pass**: perform all passes and verify code, demos, and the strict site build.

Do not broaden an audit or documentation-only request into behavior-changing refactoring.

## 1. Inspect before editing

1. Read repository instructions and inspect the working tree. Preserve unrelated and pre-existing changes.
2. Identify the package layout, supported Python versions, dependency manager, optional dependencies, test commands, and CI conventions.
3. Locate current exports, docstrings, type annotations, docs, notebooks, scripts, examples, and README quick starts.
4. Determine the established docstring style. Preserve a consistent Google, NumPy, or Sphinx style; use Google style only when no convention exists.
5. Identify import-time side effects and optional heavy dependencies that could break mkdocstrings collection.
6. Establish the intended audience and the smallest useful documentation scope. Distinguish library users from repository contributors and research reproducers.

Do not guess scientific meaning, tensor layout, units, coordinate systems, device behavior, or numerical constraints. Infer them from implementations and tests where possible; flag unresolved semantics instead of inventing them.

## 2. Define the public API

Treat an object as public when at least one of these holds:

- It is exported through a package `__init__.py` or `__all__`.
- Users are instructed to import or call it in the README, docs, or demos.
- It is a documented CLI entry point or stable configuration object.
- Tests exercise it as a supported user-facing interface.

Before documenting, stabilize the public surface:

1. Prefer short, intentional exports over documenting every non-underscored helper.
2. Add leading underscores to implementation-only helpers only when compatibility permits.
3. Avoid silently changing public names, defaults, argument order, return shapes, or exception behavior.
4. Keep compatibility shims documented as deprecated when removal is out of scope.
5. Use `__all__` when it materially clarifies wildcard exports or the supported surface; do not add it mechanically to every module.

Run the bundled static audit early to create an inventory:

```powershell
python .codex/skills/python-api-docs/scripts/audit_python_api.py <package-or-module>
```

Use `--format json` for machine-readable output and `--fail-on findings` in CI. Treat the report as a lead, not ground truth: dynamic exports, decorators, protocols, generated methods, and inherited documentation require human judgment.

Read [references/api-style.md](references/api-style.md) before modifying public API documentation or annotations.

## 3. Document code at the right layer

Apply these rules in order:

1. Put machine-checkable facts in signatures and type annotations.
2. Put the user contract in docstrings: purpose, input semantics, output semantics, exceptions, side effects, and a minimal example when useful.
3. Put rationale, numerical subtleties, invariants, and non-obvious tradeoffs in nearby comments.
4. Put task-oriented workflows and multi-object examples in the documentation site or demo files.

Avoid comments that restate the next line. Explain why a branch, tolerance, copy, synchronization, clipping operation, coordinate transform, or unusual algorithm exists.

For array/tensor APIs, document shape symbols, dtype expectations, device placement, units, coordinate conventions, mutability, and batch semantics when applicable. Keep types in annotations and semantic constraints in prose.

Document classes at the class level and merge constructor details consistently. Do not duplicate identical parameter descriptions across the class and `__init__` unless the selected renderer requires it.

## 4. Build demos as executable documentation

Create demos that answer realistic user goals, not internal implementation tours. Prefer a progression:

1. A minimal quick start that completes quickly.
2. One or more focused examples for common tasks.
3. An advanced or research reproduction example only when justified.

Make every demo runnable from a documented clean environment. Remove absolute paths, hidden local files, undeclared dependencies, interactive-only assumptions, and unexplained large downloads. Use deterministic seeds where feasible and keep outputs modest.

Link demos from both the README quick start and the documentation site when appropriate. Test copy-pasted commands exactly as shown.

Read [references/demo-guidelines.md](references/demo-guidelines.md) before creating or substantially revising demos.

## 5. Configure MkDocs and mkdocstrings

Prefer a small structure that can grow:

```text
mkdocs.yml
docs/
  index.md
  getting-started.md
  api/
  examples/
```

Adapt dependency declarations to the repository's existing manager. Keep documentation dependencies separate from runtime dependencies where the project supports optional or development groups.

Generate API pages from explicit public modules or objects. Avoid dumping an entire internal package tree into navigation. Keep hand-written conceptual and task-oriented pages separate from generated reference pages.

Use mkdocstrings' Python handler with a source path matching the repository layout. Configure the selected docstring style globally. Enable source links and signature annotations unless project needs justify otherwise.

Read [references/mkdocs-mkdocstrings.md](references/mkdocs-mkdocstrings.md) before adding or changing MkDocs configuration.

## 6. Verify the result

Use the repository's own environment and commands. At minimum:

1. Import the documented package or objects without unexpected side effects.
2. Run focused tests for edited APIs.
3. Run every new or changed quick-start/demo command.
4. Run the static API audit and review each finding.
5. Build with `mkdocs build --strict`.
6. Check generated navigation, headings, signatures, cross-references, code blocks, and images.
7. Confirm snippets use public imports and match current defaults and outputs.
8. Confirm the README, demos, and API reference do not contradict one another.

Do not claim success when optional dependencies or external assets prevented relevant verification. Report the exact unverified command and reason.

## Quality bar

- Keep behavior unchanged unless the user authorizes API changes.
- Prefer a smaller documented public surface over exhaustive internal documentation.
- Require every public API docstring to add semantic information beyond the signature.
- Keep examples short, realistic, deterministic where possible, and directly runnable.
- Make strict documentation builds reproducible in CI.
- Avoid generated prose that merely paraphrases function names.
- Cite external technical claims in narrative docs when they are not self-evident from the codebase.
- Finish with a concise summary of API decisions, documentation structure, demos added, validation performed, and unresolved gaps.
