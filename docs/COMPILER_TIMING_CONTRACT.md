# DNA-Lang Compiler Correctness & Timing Contract (`S1-02`)

**Repository**: `dnalang-core`  
**Canonical Pipeline**: `dnalang.parse(src)` $\rightarrow$ `dnalang.check(org)` $\rightarrow$ `dnalang.lower(org)` $\rightarrow$ `Circuit.validate()` $\rightarrow$ `Circuit.resolve_timing(dt, policy)` $\rightarrow$ `Circuit.to_qasm3(...)` / `Circuit.to_qiskit(...)`

---

## 1. Duration Units & Exact Normalization Policy

| Unit Literal | Exact Factor in Seconds (`fractions.Fraction`) | Valid Example | Invalid Example & Diagnostic |
| :--- | :--- | :--- | :--- |
| `ns` | `Fraction(1, 1_000_000_000)` | `wait(1000ns) q;` | `wait(10ps) q;` $\rightarrow$ `LexError: unknown unit 'ps'` |
| `us` | `Fraction(1, 1_000_000)` | `wait(1us) q;` | `wait(-5us) q;` $\rightarrow$ `line:col: wait duration must be non-negative` |
| `ms` | `Fraction(1, 1_000)` | `wait(0.001ms) q;` | `wait(1.5) q;` $\rightarrow$ `line:col: wait requires a duration expression` |
| `s` | `Fraction(1, 1)` | `wait(0.000001s) q;` | `rz(10ns) q;` $\rightarrow$ `line:col: gate 'rz' expects angle parameter, got duration` |

- **Exact Equivalence Invariant**: `1000ns`, `1us`, `0.001ms`, `0.000001s`, and `1e-6s` all evaluate to `Fraction(1, 1_000_000)` seconds and serialize to the identical canonical string `"0.000001"` in `Circuit.canonical()`.
- **Constant Folding Dimensional Algebra**:
  - Allowed: `duration ± duration -> duration`, `duration * scalar -> duration`, `scalar * duration -> duration`, `duration / scalar -> duration`, `duration / duration -> scalar`, and neutral literal `0` (`-5ns` evaluates to `-5e-9 s` `duration` and fails the non-negative check with `line:col`).
  - Forbidden: `duration ± scalar` (non-zero), `duration * duration`, `scalar / duration`, and passing a `duration` to `pi()`/`phi()`/`sqrt()`.

---

## 2. Target Clock (`dt`) Conversion & Quantization Policy (`S1-04`)

- **Missing or Non-Positive Clock**: Calling `Circuit.resolve_timing(dt=None)` or `Circuit.to_qasm3(dt=None, require_dt=True)` or passing `dt <= 0` raises `TimingError`.
- **`timing_policy="exact"` (Default)**:
  - Computes exact rational ratio `duration_exact_s / dt_exact`.
  - If `ratio.denominator != 1` (duration is not an integer multiple of `dt`), raises `TimingError` reporting the exact requested duration, `dt`, and surrounding integer tick bounds.
- **`timing_policy="quantize"` (Explicit Opt-In)**:
  - Rounds `duration_exact_s / dt_exact` deterministically to integer `ticks` and returns a `TimingResolutionReport` containing a `QuantizedDelayRecord` per delay (`requested_duration_s`, `effective_duration_s`, `ticks`, `adjustment_s`).

---

## 3. Qubit Binding, Aliasing & Measurement Mapping (`S1-02` / `S1-03`)

- **Port Declarations**: Every `gene` port list must contain unique port identifiers; every statement operand must reference a declared port.
- **Prohibited Aliasing**:
  - Multi-qubit gates (`cx`/`bond`, `cz`, `swap`) and multi-target `wait`/`echo`/`barrier` statements reject duplicate port operands.
  - Genome instances (`g(...) @ [q0, q1]`) reject negative physical qubit indices and reject binding the same physical qubit index to two ports.
  - Logical `Circuit.validate()` rejects out-of-bounds qubits (`q < 0` or `q >= n_qubits`) and duplicate qubits in `Op.qubits`.
- **Measurement Mapping (`cleave port -> cbit`)**:
  - Source check rejects `cbit` colliding with a declared quantum port (`cleave q -> q;`) and rejects duplicate `cbit` destinations within the same gene (`cleave a -> c; cleave b -> c;`).
  - Logical `Circuit.validate()` rejects `cbit is None`, non-integer `cbit`, `cbit < 0`, or `cbit >= n_cbits`.
  - All export entry points (`canonical()`, `to_qasm3()`, `to_qiskit()`, `resolve_timing()`) enforce `Circuit.validate()`.

---

## 4. Stage Scope Boundary

- This contract covers `parse` $\rightarrow$ `sema.check` $\rightarrow$ `lower` $\rightarrow$ `Circuit.validate` $\rightarrow$ `Circuit.resolve_timing` $\rightarrow$ `to_qasm3` and untranspiled `to_qiskit` construction.
- Downstream vendor hardware transpilation / instruction scheduling (e.g., Qiskit `generate_preset_pass_manager` on remote `ibm_*` targets requiring `IBM_QUANTUM_TOKEN`) is outside this local compiler boundary and is not claimed by logical or QASM3 export checks.
