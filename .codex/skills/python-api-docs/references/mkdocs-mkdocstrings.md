# MkDocs and mkdocstrings setup

Use this reference when adding or changing the documentation site. Check the installed versions and current official documentation before relying on version-sensitive options.

Consult these primary references when verification is needed:

- MkDocs configuration: <https://www.mkdocs.org/user-guide/configuration/>
- mkdocstrings Python handler: <https://mkdocstrings.github.io/python/usage/>
- Material for MkDocs setup: <https://squidfunk.github.io/mkdocs-material/getting-started/>

## Dependencies

Add documentation tools to the repository's existing development or documentation dependency group. The minimal set is normally:

```text
mkdocs-material
mkdocstrings[python]
```

Add optional extensions only when a page uses them. Keep documentation dependencies out of the runtime install when the package manager supports optional dependency groups.

Pin or constrain versions according to the repository's reproducibility policy. Do not invent a new dependency-management system solely for docs.

## Package layouts

Set `handlers.python.paths` relative to `mkdocs.yml`:

- `src/package/...` layout: use `paths: [src]`.
- `package/...` at repository root: use `paths: [.]`.
- Configuration stored under `docs/`: adjust paths relative to that location.

Prefer static collection through mkdocstrings/Griffe. Do not mutate `sys.path` in documentation pages. Make documented modules import-safe even if runtime dependencies are optional or heavy.

## Baseline `mkdocs.yml`

Adapt names, repository URLs, source path, language, and navigation:

```yaml
site_name: Project name
site_description: One-sentence project description

theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: google
            docstring_section_style: list
            filters:
              - "!^_"
            heading_level: 2
            members_order: source
            merge_init_into_class: true
            separate_signature: true
            show_root_heading: true
            show_signature_annotations: true
            show_source: true
            signature_crossrefs: true

nav:
  - Home: index.md
  - Getting started: getting-started.md
  - Examples: examples/index.md
  - API reference:
      - Overview: api/index.md
      - Core: api/core.md
```

Change `docstring_style` to `numpy` or `sphinx` when that is the repository convention. Verify every option against the installed mkdocstrings Python handler; remove unsupported options rather than silently accepting warnings.

Use tables instead of lists for short parameter descriptions if the site renders them well. For scientific APIs with long shape and unit descriptions, `docstring_section_style: list` is often more readable.

## API pages

Use the dotted import path expected by users:

```markdown
# Core API

::: package.core.Simulation

::: package.core.run_simulation
```

Prefer explicit objects for a curated stable API. For a cohesive public module, document the module with intentional options:

```markdown
::: package.geometry
    options:
      members:
        - Body
        - load_mesh
        - transform_points
```

Avoid rendering private helpers, imported implementation details, and large inherited surfaces by accident. Do not rely solely on the default non-underscore filter when the module has many incidental public names.

## Site organization

Keep four information types distinct:

- `index.md`: project purpose, status, and the shortest successful example.
- `getting-started.md`: installation, prerequisites, first workflow, and troubleshooting.
- `examples/`: task-oriented guides and demo outputs.
- `api/`: generated signatures and object contracts.

Add architecture, contributor, or theory pages only when the project needs them. Do not bury the first working example under internal design documentation.

## Cross-references

Link Python objects using the cross-reference syntax supported by the installed mkdocstrings/autorefs versions. Add external object inventories only for dependencies that appear in the public API and provide a stable inventory.

Check unresolved references under a strict build. Do not silence broad classes of warnings just to make CI green.

## Import and collection failures

If a documented module depends on optional packages or hardware:

1. Avoid executing heavyweight setup at module import time.
2. Move optional imports into the operation that requires them when appropriate.
3. Guard type-only imports without degrading rendered annotations.
4. Install the relevant documentation extra in CI when the API genuinely depends on it.
5. Prefer a small documented facade over mocking numerous dependencies.

Do not add fake modules globally unless no cleaner design exists; mocks can hide real API and import defects.

## Validation

Run from the repository root:

```text
mkdocs build --strict
mkdocs serve
```

Use `serve` for visual inspection, not as a production server. Check at least:

- Navigation order and broken links.
- Heading hierarchy and table of contents.
- Signatures, annotations, defaults, and source links.
- Docstring sections and code formatting.
- Cross-references between examples and API objects.
- Mobile-width rendering for long signatures and parameter descriptions.
- Image paths, captions, and alt text.

## CI shape

Adapt to the existing CI provider:

1. Install the package and documentation dependency group.
2. Run any docstring or link checks used by the repository.
3. Run demo smoke tests that do not require unavailable hardware.
4. Run `mkdocs build --strict`.
5. Upload or deploy only from the intended branch and event.

Keep build verification separate from deployment. Do not publish a documentation site unless the user explicitly requests deployment.
