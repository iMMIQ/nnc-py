# X86 Codegen IR Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor x86 code generation around a lowered codegen IR so scheduled O3 and serial x86 lowering are separated from final C emitters while preserving current behavior.

**Architecture:** Keep `X86Backend.generate()` as a thin orchestrator. Introduce x86-specific lowered codegen IR types plus two lowering entry points, one for serial x86 and one for scheduled O3 x86. Move artifact formatting into focused emitter modules that consume lowered IR instead of raw `CompileContext`.

**Tech Stack:** Python 3.10+, pytest, existing `nnc_py` IR/pass pipeline, current x86 codegen runtime and emitter helpers

---

## File Structure

### Create

- `src/nnc_py/codegen/x86_ir.py`
- `src/nnc_py/codegen/x86_lowering/__init__.py`
- `src/nnc_py/codegen/x86_lowering/common.py`
- `src/nnc_py/codegen/x86_lowering/serial.py`
- `src/nnc_py/codegen/x86_lowering/scheduled.py`
- `src/nnc_py/codegen/x86_emitters/__init__.py`
- `src/nnc_py/codegen/x86_emitters/header.py`
- `src/nnc_py/codegen/x86_emitters/model_source.py`
- `src/nnc_py/codegen/x86_emitters/tensors.py`
- `src/nnc_py/codegen/x86_emitters/constants_loader.py`
- `src/nnc_py/codegen/x86_emitters/build_files.py`
- `tests/test_x86_codegen_refactor_contract.py`

### Modify

- `src/nnc_py/codegen/x86_backend.py`
- `src/nnc_py/codegen/__init__.py`
- `tests/test_codegen_pipeline_schedule.py`
- `tests/test_pipeline_scheduler_e2e.py`
- `tests/test_cli.py`

### Responsibilities

- `x86_ir.py`: Narrow codegen-facing data structures shared by lowerers and emitters.
- `x86_lowering/common.py`: Shared extraction and helper shaping that should not stay in `X86Backend`.
- `x86_lowering/serial.py`: Build lowered codegen IR for normal x86 compile output.
- `x86_lowering/scheduled.py`: Build lowered codegen IR for scheduled O3 compile output, including schedule summary and parallel-runtime metadata.
- `x86_emitters/*.py`: One module per output artifact family. No schedule decisions here.
- `x86_backend.py`: Coordinate symbol assignment, choose lowerer, call emitters, return `CodeGenResult`.
- `tests/test_x86_codegen_refactor_contract.py`: New characterization tests for orchestration seam and stable generated-output markers.

## References

- Spec: `docs/superpowers/specs/2026-03-27-x86-codegen-ir-design.md`
- Existing heavy module: `src/nnc_py/codegen/x86_backend.py`
- Existing scheduled codegen coverage: `tests/test_codegen_pipeline_schedule.py`
- Existing end-to-end scheduled compile coverage: `tests/test_pipeline_scheduler_e2e.py`

### Task 1: Pin Current Behavior With Characterization Tests

**Files:**
- Create: `tests/test_x86_codegen_refactor_contract.py`
- Modify: `tests/test_codegen_pipeline_schedule.py`
- Modify: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing serial characterization test**

Add a test in `tests/test_x86_codegen_refactor_contract.py` that compiles a tiny x86 graph and asserts:
- `model.c`, `model.h`, `tensors.c`, and `test_runner.c` are emitted
- the public entry point name is preserved
- the generated source still contains a stable non-snapshot marker such as `nnc_run` or an explicit wrapper when `entry_point` is set

- [ ] **Step 2: Run the new serial test to verify it fails for the right reason**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_x86_backend_serial_contract -v`

Expected: FAIL because the new test file or test function does not exist yet.

- [ ] **Step 3: Write the failing scheduled characterization test**

Add a second test in `tests/test_x86_codegen_refactor_contract.py` that builds a scheduled-O3 context and asserts stable markers only:
- `schedule_metadata=present`
- `parallel_runtime=enabled` or `parallel_runtime=disabled`
- scheduled output still generates `model.c`

Reuse the smallest existing scheduled fixtures or helpers from `tests/test_codegen_pipeline_schedule.py` instead of inventing a new graph builder.

- [ ] **Step 4: Run the scheduled characterization test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_x86_backend_scheduled_contract -v`

Expected: FAIL because the test function does not exist yet.

- [ ] **Step 5: Implement the characterization tests and keep assertions stable**

Write the tests with fragment assertions, not full generated-file snapshots. Prefer existing helper constructors from nearby test modules where possible.

- [ ] **Step 6: Run the characterization slice until green**

Run: `pytest tests/test_x86_codegen_refactor_contract.py tests/test_codegen_pipeline_schedule.py -q`

Expected: PASS with no new warnings from the added tests.

- [ ] **Step 7: Commit the test-only baseline**

```bash
git add tests/test_x86_codegen_refactor_contract.py tests/test_codegen_pipeline_schedule.py tests/test_pipeline_scheduler_e2e.py
git commit -m "test: pin x86 codegen refactor contract"
```

### Task 2: Introduce the Lowered X86 Codegen IR

**Files:**
- Create: `src/nnc_py/codegen/x86_ir.py`
- Modify: `tests/test_x86_codegen_refactor_contract.py`

- [ ] **Step 1: Write the failing lowered-IR unit test**

Add a focused test in `tests/test_x86_codegen_refactor_contract.py` for the first IR type, for example a dataclass or small container that must hold:
- entry-point metadata
- artifact sections
- pipeline summary lines
- whether the lowered package represents serial or scheduled codegen

Keep it narrow: instantiate the type and assert defaults/fields.

- [ ] **Step 2: Run the IR test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_x86_codegen_package_defaults -v`

Expected: FAIL with `ImportError` or `AttributeError` because `x86_ir.py` does not exist yet.

- [ ] **Step 3: Write the minimal IR types**

Create `src/nnc_py/codegen/x86_ir.py` with small dataclasses only. Start with the minimum needed by later tasks, such as:

```python
@dataclass
class X86CodegenPackage:
    mode: str
    entry_point: str
    files: dict[str, Any] = field(default_factory=dict)
    pipeline_summary_lines: list[str] = field(default_factory=list)
```

Do not over-design. Add new fields later only when a failing test demands them.

- [ ] **Step 4: Run the new IR test until it passes**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_x86_codegen_package_defaults -v`

Expected: PASS.

- [ ] **Step 5: Commit the IR seam**

```bash
git add src/nnc_py/codegen/x86_ir.py tests/test_x86_codegen_refactor_contract.py
git commit -m "refactor: add x86 lowered codegen ir"
```

### Task 3: Move Scheduled-O3 Lowering Behind a Dedicated Entry Point

**Files:**
- Create: `src/nnc_py/codegen/x86_lowering/__init__.py`
- Create: `src/nnc_py/codegen/x86_lowering/common.py`
- Create: `src/nnc_py/codegen/x86_lowering/scheduled.py`
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `tests/test_codegen_pipeline_schedule.py`
- Modify: `tests/test_x86_codegen_refactor_contract.py`

- [ ] **Step 1: Write the failing scheduled-lowering seam test**

Add a test that calls a new function such as `lower_scheduled_x86_codegen(...)` and asserts it returns lowered IR containing:
- scheduled mode
- non-empty schedule summary lines
- scheduled runtime metadata when available

- [ ] **Step 2: Run the seam test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_lower_scheduled_x86_codegen_builds_pipeline_metadata -v`

Expected: FAIL because the new lowering module/function does not exist yet.

- [ ] **Step 3: Create the shared lowering helpers**

Add `x86_lowering/common.py` for shared extraction that should leave `X86Backend`, such as:
- selecting scheduled vs legacy memory-plan inputs
- comment sanitization helpers that are semantic, not formatting-only
- common symbol/metadata normalization consumed by both lowerers

Do not move emitter-specific string assembly into this module.

- [ ] **Step 4: Create the scheduled lowerer with a compatibility-first adapter**

Move the schedule-specific shaping logic from `X86Backend` into `x86_lowering/scheduled.py`. Start by wrapping existing helpers and returning lowered IR, even if some helper methods are still temporarily called through `X86Backend`.

This step is allowed to use adapters as long as the output behavior is unchanged.

- [ ] **Step 5: Route one scheduled path through the new lowerer**

Modify `x86_backend.py` so the scheduled path calls the new lowering entry point before generating `model.c`.

- [ ] **Step 6: Run the focused scheduled tests**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_lower_scheduled_x86_codegen_builds_pipeline_metadata tests/test_codegen_pipeline_schedule.py tests/test_pipeline_scheduler_e2e.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the scheduled lowering extraction**

```bash
git add src/nnc_py/codegen/x86_lowering/__init__.py src/nnc_py/codegen/x86_lowering/common.py src/nnc_py/codegen/x86_lowering/scheduled.py src/nnc_py/codegen/x86_backend.py tests/test_codegen_pipeline_schedule.py tests/test_x86_codegen_refactor_contract.py tests/test_pipeline_scheduler_e2e.py
git commit -m "refactor: extract scheduled x86 lowering"
```

### Task 4: Move Serial X86 Lowering Behind a Dedicated Entry Point

**Files:**
- Create: `src/nnc_py/codegen/x86_lowering/serial.py`
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `tests/test_x86_codegen_refactor_contract.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing serial-lowering seam test**

Add a test for `lower_serial_x86_codegen(...)` that asserts:
- serial mode
- expected artifact metadata scaffolding
- no scheduled-only runtime metadata is attached

- [ ] **Step 2: Run the seam test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_lower_serial_x86_codegen_omits_scheduled_runtime_metadata -v`

Expected: FAIL because `serial.py` or the function does not exist yet.

- [ ] **Step 3: Implement the serial lowerer using the minimum common helpers**

Create `serial.py` and move serial shaping logic out of `X86Backend`. Keep the logic close to existing behavior; do not try to unify every branch prematurely.

- [ ] **Step 4: Route normal x86 generation through the serial lowerer**

Update `x86_backend.py` so both compile paths now go through dedicated lowerers and return lowered IR before emission.

- [ ] **Step 5: Verify CLI-facing serial behavior**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_lower_serial_x86_codegen_omits_scheduled_runtime_metadata tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the serial lowering extraction**

```bash
git add src/nnc_py/codegen/x86_lowering/serial.py src/nnc_py/codegen/x86_backend.py tests/test_x86_codegen_refactor_contract.py tests/test_cli.py
git commit -m "refactor: extract serial x86 lowering"
```

### Task 5: Extract Artifact Emitters

**Files:**
- Create: `src/nnc_py/codegen/x86_emitters/__init__.py`
- Create: `src/nnc_py/codegen/x86_emitters/header.py`
- Create: `src/nnc_py/codegen/x86_emitters/model_source.py`
- Create: `src/nnc_py/codegen/x86_emitters/tensors.py`
- Create: `src/nnc_py/codegen/x86_emitters/constants_loader.py`
- Create: `src/nnc_py/codegen/x86_emitters/build_files.py`
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/codegen/__init__.py`
- Modify: `tests/test_x86_codegen_refactor_contract.py`

- [ ] **Step 1: Write the failing emitter contract test**

Add a test that constructs a minimal lowered package and asserts a dedicated emitter returns text for one artifact, for example:
- header emitter returns a string containing the entry-point declaration
- model source emitter includes schedule summary lines when supplied

- [ ] **Step 2: Run the emitter contract test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_header_emitter_uses_lowered_package_entry_point -v`

Expected: FAIL because the emitter module does not exist yet.

- [ ] **Step 3: Extract the simplest emitter first**

Implement `header.py` and move only the header-specific formatting there. Keep signatures narrow, for example:

```python
def emit_header(pkg: X86CodegenPackage) -> str:
    ...
```

- [ ] **Step 4: Extract source and artifact emitters incrementally**

Move source, tensors, constants loader, Makefile, and test-runner formatting into emitter modules. Preserve current formatting behavior unless a cleanup is required to separate semantic decisions from string formatting.

- [ ] **Step 5: Update `X86Backend.generate()` to call emitters only**

The backend should now:
- choose lowering path
- obtain lowered package
- call emitter functions
- assemble `CodeGenResult`

- [ ] **Step 6: Run codegen-focused tests**

Run: `pytest tests/test_x86_codegen_refactor_contract.py tests/test_codegen_pipeline_schedule.py tests/test_scheduled_tile_codegen.py tests/test_x86_runtime.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the emitter extraction**

```bash
git add src/nnc_py/codegen/x86_emitters src/nnc_py/codegen/x86_backend.py src/nnc_py/codegen/__init__.py tests/test_x86_codegen_refactor_contract.py
git commit -m "refactor: extract x86 codegen emitters"
```

### Task 6: Remove Temporary Adapters and Tighten Interfaces

**Files:**
- Modify: `src/nnc_py/codegen/x86_backend.py`
- Modify: `src/nnc_py/codegen/x86_lowering/common.py`
- Modify: `src/nnc_py/codegen/x86_lowering/serial.py`
- Modify: `src/nnc_py/codegen/x86_lowering/scheduled.py`
- Modify: `src/nnc_py/codegen/x86_emitters/model_source.py`
- Modify: `tests/test_x86_codegen_refactor_contract.py`

- [ ] **Step 1: Write the failing no-backsliding test**

Add a test that verifies the emitter path consumes lowered IR directly and does not require raw `CompileContext` for schedule markers or entry-point naming. This can be expressed as a unit-level emitter test using only lowered package data.

- [ ] **Step 2: Run the no-backsliding test to verify it fails**

Run: `pytest tests/test_x86_codegen_refactor_contract.py::test_model_source_emitter_uses_lowered_pipeline_summary_without_context -v`

Expected: FAIL because one emitter still depends on backend/context-only helpers.

- [ ] **Step 3: Delete compatibility shims that only existed for transition**

Remove temporary adapter methods from `X86Backend` once lowerers and emitters can operate on direct inputs. Keep only backend orchestration helpers that still belong at backend level.

- [ ] **Step 4: Tighten imports and public exports**

Update `src/nnc_py/codegen/__init__.py` if needed so the package exposes only stable entry points. Do not expose internal lowerers unless tests or internal imports require it.

- [ ] **Step 5: Run the focused seam tests again**

Run: `pytest tests/test_x86_codegen_refactor_contract.py tests/test_cli.py tests/test_pipeline_scheduler_e2e.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the cleanup**

```bash
git add src/nnc_py/codegen/x86_backend.py src/nnc_py/codegen/x86_lowering src/nnc_py/codegen/x86_emitters src/nnc_py/codegen/__init__.py tests/test_x86_codegen_refactor_contract.py tests/test_cli.py tests/test_pipeline_scheduler_e2e.py
git commit -m "refactor: slim x86 backend orchestration"
```

### Task 7: Full Verification and Final Cleanup

**Files:**
- Modify: any touched files from prior tasks if verification exposes regressions

- [ ] **Step 1: Run the full targeted verification matrix**

Run:

```bash
pytest tests/test_x86_codegen_refactor_contract.py -q
pytest tests/test_codegen_pipeline_schedule.py tests/test_pipeline_scheduler_e2e.py tests/test_pipeline_pass_integration.py -q
pytest tests/test_scheduled_tile_codegen.py tests/test_x86_runtime.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run: `pytest -q`

Expected: PASS. Existing environment-dependent skips are acceptable; new failures are not.

- [ ] **Step 3: Inspect the size reduction in `x86_backend.py`**

Run:

```bash
wc -l src/nnc_py/codegen/x86_backend.py
find src/nnc_py/codegen/x86_lowering -maxdepth 1 -type f | sort
find src/nnc_py/codegen/x86_emitters -maxdepth 1 -type f | sort
```

Expected: `x86_backend.py` is materially smaller and the new modules exist.

- [ ] **Step 4: Commit the verified refactor**

```bash
git add src/nnc_py/codegen tests
git commit -m "refactor: split x86 codegen around lowered ir"
```

## Local Review Checklist

- Lowerers own compile-path decisions; emitters do not.
- Emitters accept lowered package data, not raw `CompileContext`.
- Scheduled O3 metadata is preserved in generated output.
- Serial x86 output still works when schedule metadata is absent.
- `X86Backend.generate()` reads like orchestration, not a giant code dump.
- Tests pin behavior with stable markers instead of overfitting to full-file formatting.
