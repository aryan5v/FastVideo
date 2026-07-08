---
date: 2026-07-08
experiment: Day-1 mx.compile A/B (docs/design/apple_silicon_program_plan.md, step 1)
category: infrastructure
severity: important
---

# A NumPy scalar times a traced mx.array silently breaks mx.compile

## What Happened
`mx.compile` on the FastWan MLX DiT forward produced no speedup: it either raised
`Attempting to eval an array during function transformations like compile or vmap
is not allowed` (caught by the wrapper in `fastwan.py`, silent eager fallback) or
**segfaulted the process (exit 139)**. The roadmap counted on `mx.compile` as a
"next runtime win," so this was a blocker.

## Root Cause
`gelu_tanh` computed `np.sqrt(2.0 / np.pi) * x`, where `x` is a **traced** array
during compile. `np.sqrt(...)` is a `np.float64`, so `np.float64.__mul__` runs
first and tries to build a NumPy array from the traced `mx.array` — which forces
an eval, illegal under `mx.compile`. The same illegal eval is what crashed the
Metal backend (the SIGSEGV was a downstream symptom, not a separate bug).

`timestep_embedding` had the same `np.log(...) * mx.array` anti-pattern but did
not fail, because its operand (`mx.arange(...)`) is a compile-time constant that
NumPy *can* materialize; only a numpy-scalar × *traced* array breaks.

## Fix / Workaround
Use Python `float` constants, not NumPy scalars, in any expression multiplying a
traced array. Fixed both sites in `fastvideo/mlx_runtime/fastwan.py`
(`_GELU_TANH_COEF = math.sqrt(2/pi)`; `math.log` in `timestep_embedding`). With
the fix, compile traces cleanly (no fallback), is **bit-identical to eager**, and
gives **1.41× (fp16) / 1.43× (int8)** steady-step speedup. Guarded by
`fastvideo/tests/mlx/test_mlx_compile_parity.py`.

## Prevention
- In MLX code that may be `mx.compile`d, never let a `np.<scalar>` multiply/add a
  traced array. Cast to Python `float` (or use `math`), or wrap the constant in
  `mx.array`. Prefer `math` over `np` for scalar constants in hot forward paths.
- `mx.compile` failures can be a **native segfault**, not a Python exception — a
  `try/except` around the compiled call cannot catch it. Test compile paths with
  an explicit parity test rather than relying on runtime fallback.
