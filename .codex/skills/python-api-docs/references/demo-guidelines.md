# Demo and example guidelines

Use this reference when creating or revising runnable examples, notebooks, screenshots, animations, or README snippets.

## Choose the artifact

- Use a short README snippet for installation plus the first successful result.
- Use a `.py` file for repeatable, testable, and command-line-friendly demos.
- Use a notebook for exploration, visualization, or a narrative where intermediate state matters.
- Use a documentation page for explanation that combines multiple public objects.
- Use a gallery only when each example answers a distinct user question.

Prefer scripts over notebooks for the canonical runnable path. If both exist, share library code instead of duplicating logic.

## Structure each demo

Make each demo answer five questions in order:

1. What will the user accomplish?
2. What prerequisites or assets are required?
3. What is the minimal setup?
4. Which public API calls perform the task?
5. What output should the user expect?

For scripts, prefer a small `main()` and a standard guard:

```python
def main() -> None:
    """Run the minimal simulation demo."""
    ...


if __name__ == "__main__":
    main()
```

Add CLI arguments only when they make the example meaningfully reusable. Keep documented defaults fast and portable.

## Reproducibility

- Set random seeds across every random-number library actually used.
- State when hardware, precision, parallelism, or solver nondeterminism remains.
- Resolve paths relative to the repository, installed package, or explicit CLI arguments.
- Declare every dependency and optional extra needed by the demo.
- Use small checked-in assets with compatible licenses, or provide a deterministic retrieval command and checksum.
- Avoid requiring credentials, private storage, workstation-specific paths, or pre-existing output directories.
- Create outputs under an explicit user-provided or documented directory.
- Do not overwrite valuable results silently.

## Scientific and visual demos

State:

- The physical/numerical scenario and relevant assumptions.
- Units or nondimensionalization.
- Initial and boundary conditions when they affect interpretation.
- Array or tensor shapes at user-facing boundaries.
- Expected qualitative behavior and any useful quantitative check.
- Runtime and hardware expectations when substantial.

For images, plots, or animations:

- Generate them from a tracked script or notebook.
- Use readable labels, units, legends, and color scales.
- Avoid screenshots with local paths, usernames, editor chrome, or irrelevant windows.
- Keep generated binary output out of version control when it is large or reproducible and not needed by the docs.
- Add concise alt text and captions in the documentation.

## Testing demos

Test the exact documented command from a clean environment when practical. Separate fast smoke-test parameters from high-quality production parameters:

```text
python examples/run_case.py --steps 5       # CI smoke test
python examples/run_case.py --steps 1000    # documented full result
```

Do not present placeholder output as measured output. Use assertions or cheap sanity checks for shapes, finiteness, file creation, and basic invariants.

For notebooks:

- Restart the kernel and run all cells in order.
- Clear accidental debug output and secrets.
- Keep outputs only when they help readers and remain reasonably small.
- Ensure the narrative does not rely on hidden execution state.

## Documentation integration

- Link the minimal example from `docs/index.md` or `docs/getting-started.md`.
- Create an examples index organized by user goal.
- Link every example to the API objects it uses.
- Link API reference pages back to a focused example when one exists.
- Keep README commands synchronized with the canonical docs instead of maintaining divergent variants.

## Demo review checklist

- Uses public imports only.
- Runs with documented dependencies.
- Has no absolute or private paths.
- Has a clear expected result.
- Completes in reasonable time with default or smoke-test arguments.
- Avoids unnecessary framework or training boilerplate.
- Explains inputs and outputs without exposing internal implementation details.
- Produces deterministic or bounded-variance results where feasible.
- Has a compatible license for every bundled asset.
- Is exercised by CI or a documented manual validation command.
