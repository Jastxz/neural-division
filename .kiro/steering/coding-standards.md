---
inclusion: always
---

# Coding Standards — DivisionNeuronal (Julia)

## Julia
- Follow Julia style guide: functions lowercase with underscores, types PascalCase
- Use multiple dispatch — define methods on concrete types
- Explicit type annotations on struct fields
- Avoid global mutable state — pass state as arguments
- Use `@inbounds` and `@simd` only with verified bounds
- Allocations: minimize in hot loops — preallocate buffers

## Performance
- Type stability is critical — verify with `@code_warntype`
- Use `StaticArrays` or preallocated arrays for fixed-size data
- GPU kernels via KernelAbstractions when available

## Testing
- `test/` uses Julia's `Test` stdlib + PropCheck for property tests
- Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Property tests should cover edge cases in numerical operations

## Package Management
- `Project.toml` defines deps and compat bounds
- `Manifest.toml` locks exact versions — commit it
