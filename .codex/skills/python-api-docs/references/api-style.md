# Python public API and docstring style

Use this reference when defining public objects or writing type annotations, docstrings, and code comments.

## Public API checklist

For every supported module, class, function, method, property, exception, and configuration object, verify:

- The name expresses a user concept rather than an implementation detail.
- The import path is intentional and demonstrated consistently.
- The signature exposes stable concepts and avoids unnecessary positional booleans.
- Parameters and returns have useful type annotations.
- Mutable defaults are avoided.
- `None`, sentinel values, accepted protocols, and union branches have clear semantics.
- Shapes, dtypes, devices, units, ranges, coordinate systems, and mutability are stated when relevant.
- Side effects, state changes, caching, I/O, randomness, and thread/process constraints are stated when relevant.
- Raised exceptions describe actionable caller errors rather than every internal exception.
- Deprecations point to a replacement and, when known, a removal boundary.
- The object has at least one realistic usage path in docs or demos when it is central to the package.

## Type annotations

Use annotations as the canonical representation of types. Do not repeat types in Google-style parameter descriptions unless additional semantic detail is necessary.

Prefer:

- Precise built-in generics such as `list[str]`, `dict[str, float]`, and `tuple[int, ...]` on supported Python versions.
- Abstract input protocols such as `Sequence`, `Mapping`, `Iterable`, or path-like types when the implementation accepts them.
- Concrete return types when callers rely on concrete behavior.
- `Protocol`, `TypedDict`, dataclasses, or small configuration classes when they clarify a stable structured contract.
- Type aliases for repeated domain shapes or identifiers.

Avoid:

- `Any` used only to silence analysis.
- Extremely complex annotations that obscure the user contract.
- Types that promise broader inputs than the implementation accepts.
- Annotations imported from optional heavy dependencies in a way that breaks runtime imports.

When runtime import cost or cycles matter, use the repository's supported postponed-annotation strategy and `TYPE_CHECKING` carefully. Verify that mkdocstrings still resolves useful signatures.

## Google-style docstrings

Use the imperative mood for actions and a noun phrase for data containers when natural. Start with a one-line summary that states the user-visible result. Add details only when they change correct use.

```python
def advect_density(
    density: "ArrayLike",
    velocity: "ArrayLike",
    *,
    time_step: float,
) -> "NDArray[np.float32]":
    """Advance a density field by one simulation step.

    Args:
        density: Scalar field with shape ``(ny, nx)``. Values are copied and
            the input is not modified.
        velocity: Velocity field with shape ``(ny, nx, 2)`` in lattice units
            per step. The final axis stores ``(x, y)`` components.
        time_step: Positive integration step in lattice time units.

    Returns:
        Advected density with shape ``(ny, nx)`` and dtype ``float32``.

    Raises:
        ValueError: If the spatial shapes do not match or ``time_step`` is not
            positive.

    Examples:
        >>> result = advect_density(rho, velocity, time_step=0.1)
        >>> result.shape
        (128, 256)
    """
```

Include only applicable sections:

- `Args`: Explain semantic meaning, valid ranges, shapes, units, and special values.
- `Returns`: Explain meaning and structure, not merely the annotated type.
- `Yields`: Document iterator or generator semantics.
- `Raises`: List stable, caller-relevant failure modes.
- `Examples`: Provide a small copy-pastable interaction.
- `Notes`: Explain algorithms, numerical properties, or performance constraints.
- `Warnings`: State risks that can cause wrong results, data loss, or unsafe use.
- `See Also`: Connect related public objects without replacing explanation.
- `References`: Cite papers or standards supporting scientific algorithms.

Do not add empty sections. Do not document `self` or `cls`. Avoid promising exact exception text unless callers are expected to depend on it.

## Classes and constructors

Put the object's purpose, lifecycle, and constructor parameters in the class docstring when MkDocs is configured with `merge_init_into_class: true`.

```python
class Simulation:
    """Run a two-dimensional lattice-Boltzmann simulation.

    Args:
        grid_shape: Grid size as ``(ny, nx)``.
        viscosity: Positive kinematic viscosity in lattice units.

    Attributes:
        step_count: Number of completed simulation steps.
    """
```

Document public attributes that users may read or set. Do not expose incidental cached fields just because they exist on the instance.

## Modules and packages

Add a module docstring when it communicates role, boundaries, or important usage. Do not add boilerplate such as "This module contains functions." A useful module docstring can state:

- The concept owned by the module.
- The primary public entry points.
- Important conventions shared across its API.
- A short example when the module is a main entry point.

Use package `__init__.py` to present intentional convenience imports. Avoid re-exporting every internal symbol.

## Comments

Write comments for non-obvious decisions and invariants:

```python
# Copy before applying the boundary condition because callers reuse the
# pre-collision field when computing the diagnostic loss.
updated = field.copy()
```

Prefer naming or extracting a helper over narrating ordinary code. Remove stale commented-out code and misleading TODOs during the documentation pass only when their intent is safely understood.

For scientific and numerical code, explain:

- Why a tolerance or stabilization term is used.
- Which convention an index or coordinate transform follows.
- Why synchronization, copying, detaching, or casting is necessary.
- Which paper, equation, or invariant motivates a non-obvious operation.
- Whether a shortcut trades accuracy for performance.

Do not use comments to hide unclear API design.

## Examples in docstrings

Keep docstring examples small enough to run as doctests when feasible. Use ellipses or approximate checks for unstable representations. Avoid expensive simulations, GUI windows, network calls, large assets, and hardware-only operations in docstrings; place those in standalone demos.

Prefer examples that demonstrate the supported import path:

```python
from package import Simulation
```

Avoid reaching into private modules in user-facing examples.
