# External Joint Solver Submodule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the current joint baseline solver into a standalone `nnc-joint-solver` GitHub repository, mount it into `nnc-py` as a git submodule, and make the main compiler require the submodule-backed CLI solver instead of any in-repo fallback.

**Architecture:** Keep `nnc-py` responsible for building the external joint problem, validating returned solutions, materializing them into internal schedule IR, and importing solver-authored SRAM placement. Move the minimum closed solver loop into the new submodule: joint contract types, validation, baseline solver, and CLI entrypoint. In `nnc-py`, delete the in-repo baseline path and require a CLI command derived from the submodule location so solver capability is exercised through the external repository boundary.

**Tech Stack:** Python 3.11+, dataclasses, pytest, git submodules, GitHub SSH remotes, current `nnc_py` joint schedule pipeline.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `joint_solver/` | Create | Local checkout target for the new `nnc-joint-solver` git submodule |
| `joint_solver/pyproject.toml` | Create | Package metadata and CLI entrypoint for the external solver |
| `joint_solver/bin/nnc-joint-solver` | Create | Checkout-local executable wrapper that runs directly from the submodule without installation |
| `joint_solver/README.md` | Create | Submodule usage, contract scope, CLI contract, and local development instructions |
| `joint_solver/src/nnc_joint_solver/ir/joint_tiling_schedule.py` | Create | Standalone copy of the external joint problem/solution contract used by the solver repo |
| `joint_solver/src/nnc_joint_solver/validation.py` | Create | Standalone problem/solution validator for the solver repo |
| `joint_solver/src/nnc_joint_solver/solver.py` | Create | Baseline solver extracted from `nnc-py` |
| `joint_solver/src/nnc_joint_solver/cli.py` | Create | JSON stdin/stdout CLI used by `nnc-py` |
| `joint_solver/src/nnc_joint_solver/__init__.py` | Create | Package exports for solver repo |
| `joint_solver/tests/test_solver.py` | Create | Solver repo tests for baseline solving and CLI payloads |
| `.gitmodules` | Create/Modify | Register `joint_solver` as a git submodule pointing at `git@github.com:iMMIQ/nnc-joint-solver.git` |
| `src/nnc_py/joint_schedule/solver.py` | Modify | Remove in-repo baseline solver and keep only external CLI transport adapter |
| `src/nnc_py/passes/joint_tiling_schedule.py` | Modify | Require the submodule-backed solver command and fail fast when the submodule/CLI is unavailable |
| `src/nnc_py/compiler.py` | Modify | Keep compiler failure surfaces readable when the required external solver cannot be resolved |
| `tests/fake_joint_solver.py` | Modify | Keep transport-error fixtures self-contained without importing deleted in-repo baseline logic |
| `tests/test_joint_tiling_schedule_solver.py` | Modify | Reframe tests around CLI transport only; remove in-repo baseline assertions |
| `tests/test_joint_tiling_schedule_materialize.py` | Modify | Replace direct in-repo baseline usage with submodule CLI-backed fixtures or explicit handcrafted solutions |
| `tests/test_pipeline_pass_integration.py` | Modify | Verify the joint path resolves the submodule command and rejects missing solver configuration |
| `tests/test_pipeline_scheduler_e2e.py` | Modify | Point joint e2e coverage at the submodule CLI |
| `tests/joint_solver_helpers.py` | Create | Shared helper for resolving the checked-out submodule CLI in tests with actionable failure messages |
| `README.md` | Modify | Document the required `git submodule update --init --recursive` bootstrap for joint-solver development and tests |
| `docs/superpowers/plans/2026-04-01-external-joint-solver-submodule.md` | Create | Implementation plan for this extraction and submodule integration |

---

### Task 1: Create The GitHub Repo And Add The Real Submodule Checkout

**Files:**
- Create/Modify: `.gitmodules`
- Create/Modify: `joint_solver/` submodule checkout
- Modify: `README.md`

- [ ] **Step 1: Create the empty GitHub repository first**

Run:

```bash
gh repo create iMMIQ/nnc-joint-solver --private
```

Expected: GitHub repo exists at `git@github.com:iMMIQ/nnc-joint-solver.git`.

- [ ] **Step 2: Add it to `nnc-py` as a real submodule checkout**

Run:

```bash
git submodule add git@github.com:iMMIQ/nnc-joint-solver.git joint_solver
```

Expected: `.gitmodules` is created and `joint_solver/` is a real submodule checkout.

- [ ] **Step 3: Document bootstrap requirements immediately**

Update `README.md` to state that any clone intending to use the joint solver path must run:

```bash
git submodule update --init --recursive
```

and that the joint path in source checkouts expects the checked-out submodule CLI to be present. If installed-package execution without the source checkout is not supported for the joint path, say that explicitly.

- [ ] **Step 4: Verify submodule state**

Run:

```bash
git submodule status
git config -f .gitmodules --get-regexp 'submodule\\..*'
git -C joint_solver remote -v
```

Expected: `joint_solver` points at `git@github.com:iMMIQ/nnc-joint-solver.git`.

- [ ] **Step 5: Commit**

```bash
git add .gitmodules README.md joint_solver
git commit -m "chore: add external joint solver submodule"
```

### Task 2: Create The Standalone Solver Repository Skeleton

**Files:**
- Create: `joint_solver/.gitignore`
- Create: `joint_solver/pyproject.toml`
- Create: `joint_solver/bin/nnc-joint-solver`
- Create: `joint_solver/README.md`
- Modify: `joint_solver/src/nnc_joint_solver/__init__.py`
- Create: `joint_solver/src/nnc_joint_solver/cli.py`
- Modify: `joint_solver/tests/test_solver.py`

- [ ] **Step 1: Write the failing packaging and CLI test**

```python
from nnc_joint_solver.cli import main


def test_cli_main_returns_failure_for_invalid_payload(monkeypatch):
    monkeypatch.setattr("sys.stdin.read", lambda: "{}")
    assert main([]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: FAIL because the submodule package and CLI do not exist yet.

- [ ] **Step 3: Add the minimal standalone package skeleton**

Create:

- `pyproject.toml` with a console script like `nnc-joint-solver = nnc_joint_solver.cli:run`
- `bin/nnc-joint-solver` as a checked-in executable wrapper that runs with the current Python interpreter and the submodule's `src/` path
- `README.md` documenting stdin/stdout JSON behavior and repository role
- package `__init__.py`
- CLI wrapper that reads stdin and writes stdout

- [ ] **Step 4: Run the test to verify the skeleton loads**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: PASS for the packaging/import smoke case or fail later on missing solver logic, which is acceptable before Task 3.

- [ ] **Step 4: Run the tests to verify round-trip and validation behavior**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: PASS for the contract/validation tests.

- [ ] **Step 5: Commit**

```bash
git -C joint_solver add .
git -C joint_solver commit -m "feat: scaffold external joint solver package"
git -C joint_solver push origin HEAD
```

### Task 3: Copy The Joint Contract And Validation Into The Solver Repo

**Files:**
- Create: `joint_solver/src/nnc_joint_solver/ir/joint_tiling_schedule.py`
- Create: `joint_solver/src/nnc_joint_solver/validation.py`
- Modify: `joint_solver/src/nnc_joint_solver/__init__.py`
- Modify: `joint_solver/tests/test_solver.py`

- [ ] **Step 1: Write the failing contract round-trip and validation tests**

```python
from nnc_joint_solver.ir.joint_tiling_schedule import JointProblem, JointSolution
from nnc_joint_solver.validation import validate_joint_problem


def test_joint_problem_round_trips_from_json(minimal_problem_payload):
    problem = JointProblem.from_json(minimal_problem_payload)
    assert problem.to_json()["schema_version"] == "joint_tiling_schedule_problem_v1"
    assert problem.sram_items
    assert problem.default_alignment_bytes == 16


def test_invalid_problem_returns_structured_failure(minimal_problem_payload):
    payload = dict(minimal_problem_payload)
    payload["recipes"] = []
    problem = JointProblem.from_json(payload)
    failure = validate_joint_problem(problem)
    assert failure is not None


def test_joint_solution_round_trips_required_sram_fields(solution_payload):
    solution = JointSolution.from_json(solution_payload)
    assert solution.residency_windows[0].residency_id
    assert solution.generated_sram_items[0].owner_residency_id
    assert solution.sram_allocations[0].offset >= 0


def test_upgraded_v1_contract_rejects_missing_required_sram_fields(problem_payload, solution_payload):
    bad_problem_payload = dict(problem_payload)
    bad_problem_payload.pop("sram_items")
    bad_solution_payload = dict(solution_payload)
    bad_solution_payload.pop("sram_allocations")
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: FAIL because the copied contract and validator do not exist yet.

- [ ] **Step 3: Copy the minimum closed contract and validation logic**

Copy from `nnc-py` into the solver repo:

- external IR types needed by the solver and CLI
- `JointFailure`, `JointProblem`, `JointSolution`
- `validate_joint_problem()` and `validate_joint_solution()`

Keep names and schema IDs identical to `nnc-py`. Do not introduce `V2` naming or extra wrappers.
The copied contract must include the already-upgraded required fields:

- `JointProblem.sram_items`
- `JointProblem.default_alignment_bytes`
- `JointResidencyWindow.residency_id`
- `JointSolution.generated_sram_items`
- `JointSolution.sram_allocations`

Add explicit tests that fail fast on missing:

- `sram_items`
- `default_alignment_bytes`
- `residency_id`
- `generated_sram_items`
- `sram_allocations`

and on invalid residency/allocation linkage.

- [ ] **Step 4: Run the tests to verify round-trip and validation behavior**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: PASS for the contract/validation tests.

 - [ ] **Step 5: Commit**

```bash
git -C joint_solver add src/nnc_joint_solver tests/test_solver.py
git -C joint_solver commit -m "feat: vendor joint contract and validation"
git -C joint_solver push origin HEAD
```

### Task 4: Extract The Baseline Solver Into The Solver Repo

**Files:**
- Create: `joint_solver/src/nnc_joint_solver/solver.py`
- Modify: `joint_solver/src/nnc_joint_solver/cli.py`
- Modify: `joint_solver/tests/test_solver.py`

- [ ] **Step 1: Write the failing baseline solver tests**

```python
from nnc_joint_solver.solver import BaselineJointScheduleSolver


def test_baseline_solver_returns_solution_for_allocatable_problem(allocatable_problem):
    result = BaselineJointScheduleSolver().solve(allocatable_problem)
    assert result.schema_version == "joint_tiling_schedule_solution_v1"
    assert {window.residency_id for window in result.residency_windows}
    assert result.generated_sram_items
    assert result.sram_allocations


def test_cli_emits_solution_json_for_valid_problem(allocatable_problem_json):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: FAIL because the solver implementation is not present yet.

- [ ] **Step 3: Copy and adapt the current baseline solver**

Move the closed solver logic currently in `src/nnc_py/joint_schedule/solver.py` into the new repository:

- `BaselineJointScheduleSolver`
- helper functions for topo order, residency windows, SRAM item generation, offset packing, and structured baseline failures

Adapt imports to point at `nnc_joint_solver` local modules.
Do not regress the upgraded contract: the extracted solver must emit solver-authored residency IDs, generated resident items, and offset-bearing allocations exactly as the current upgraded `nnc-py` contract expects.

- [ ] **Step 4: Wire the CLI to use the extracted baseline solver**

`cli.py` should:

- parse a `JointProblem` from stdin JSON
- run `BaselineJointScheduleSolver`
- print `JointSolution` or `JointFailure` JSON to stdout
- return non-zero only on transport/protocol failure, not on structured infeasible/error payloads

- [ ] **Step 5: Run solver repo tests**

Run:

```bash
pytest joint_solver/tests/test_solver.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C joint_solver add src/nnc_joint_solver tests/test_solver.py
git -C joint_solver commit -m "feat: add baseline external joint solver"
git -C joint_solver push origin HEAD
```

### Task 5: Delete The In-Repo Baseline Path From `nnc-py`

**Files:**
- Modify: `src/nnc_py/joint_schedule/solver.py`
- Modify: `src/nnc_py/passes/joint_tiling_schedule.py`
- Modify: `src/nnc_py/compiler.py`
- Modify: `tests/test_joint_tiling_schedule_solver.py`

- [ ] **Step 1: Write the failing main-repo tests**

```python
from nnc_py.passes.joint_tiling_schedule import _build_solver


def test_joint_solver_requires_external_command(ctx):
    ctx.metadata["joint_tiling_schedule_solver_command"] = None
    with pytest.raises(RuntimeError):
        _build_solver(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_joint_tiling_schedule_solver.py -v
```

Expected: FAIL because `nnc-py` still falls back to the in-repo baseline solver.

- [ ] **Step 3: Remove the in-repo baseline implementation**

In `src/nnc_py/joint_schedule/solver.py`:

- delete `BaselineJointScheduleSolver`
- keep `JointScheduleSolver`, `CliJointScheduleSolver`, and transport helpers
- remove any baseline-only helpers no longer used

In `src/nnc_py/passes/joint_tiling_schedule.py`:

- replace fallback behavior with required external-command resolution
- resolve the default command from the submodule path under the repo root
- use the checked-in executable path `joint_solver/bin/nnc-joint-solver` or an equivalent checkout-local script path that works without package installation
- raise a clear runtime error if the submodule checkout or CLI is unavailable

In `src/nnc_py/compiler.py`:

- keep sanitized error messages readable for missing solver command / missing submodule failures

- [ ] **Step 4: Reframe tests around CLI-only behavior**

Update `tests/test_joint_tiling_schedule_solver.py` so it validates:

- successful CLI solution payload handling
- structured CLI failure payload handling
- crash/timeout/malformed payload handling
- missing-solver-command or missing-submodule rejection

- [ ] **Step 5: Run the tests to verify the baseline path is gone**

Run:

```bash
pytest tests/test_joint_tiling_schedule_solver.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/joint_schedule/solver.py src/nnc_py/passes/joint_tiling_schedule.py src/nnc_py/compiler.py tests/test_joint_tiling_schedule_solver.py
git commit -m "refactor: require external joint solver cli"
```

### Task 6: Point Integration And Materialization Tests At The External Solver Repo

**Files:**
- Modify: `tests/fake_joint_solver.py`
- Create: `tests/joint_solver_helpers.py`
- Modify: `tests/test_joint_tiling_schedule_materialize.py`
- Modify: `tests/test_pipeline_pass_integration.py`
- Modify: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing integration tests**

```python
def test_joint_pipeline_uses_submodule_solver_command(...):
    ...


def test_materialize_fixture_no_longer_imports_baseline_directly(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_joint_tiling_schedule_materialize.py tests/test_pipeline_pass_integration.py tests/test_pipeline_scheduler_e2e.py -v --run-integration
```

Expected: FAIL because the tests still import or rely on the deleted in-repo baseline solver.

- [ ] **Step 3: Update fixtures and helpers**

Change tests to use one of these two patterns:

- invoke the submodule CLI directly for end-to-end solver path verification
- build explicit `JointSolution` fixtures inline when testing materialization in isolation

Do not import solver implementation symbols from the submodule into `nnc-py` tests. Outside the submodule's own `joint_solver/tests`, the boundary is CLI-only or explicit hand-authored solution fixtures.
Use `tests/joint_solver_helpers.py` to centralize submodule CLI resolution and emit an actionable failure if the checkout is missing.

- [ ] **Step 4: Run the integration tests**

Run:

```bash
pytest tests/test_joint_tiling_schedule_materialize.py tests/test_pipeline_pass_integration.py tests/test_pipeline_scheduler_e2e.py -v --run-integration
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/fake_joint_solver.py tests/test_joint_tiling_schedule_materialize.py tests/test_pipeline_pass_integration.py tests/test_pipeline_scheduler_e2e.py
git commit -m "test: exercise external joint solver submodule path"
```

### Task 7: Run Focused Regression And Finalize

**Files:**
- Modify: any touched files above if regressions require follow-up

- [ ] **Step 1: Run the focused solver and joint-path regression**

Run:

```bash
git submodule update --init --recursive
pytest joint_solver/tests/test_solver.py tests/test_joint_tiling_schedule_solver.py tests/test_joint_tiling_schedule_validation.py tests/test_joint_tiling_schedule_materialize.py tests/test_pipeline_pass_integration.py tests/test_pipeline_scheduler_e2e.py tests/test_joint_schedule_memory_import.py -v --run-integration
```

Expected: PASS

- [ ] **Step 2: Run a broader scheduled/joint regression if the focused suite passes**

Run:

```bash
pytest tests/test_joint_tiling_schedule_ir.py tests/test_pipeline_schedule_ir.py tests/test_joint_tiling_schedule_problem_builder.py tests/test_codegen_pipeline_schedule.py tests/test_pipeline_scheduler.py tests/test_scheduled_memory_planning.py -v --run-integration
```

Expected: PASS

- [ ] **Step 3: Inspect git state in both repositories**

Run:

```bash
git status --short --branch
git -C joint_solver status --short --branch
git log --oneline -n 8
git -C joint_solver log --oneline -n 8
```

Expected: clean working trees with the intended commit history.

- [ ] **Step 4: Commit any final fixups**

```bash
git add <final-files>
git commit -m "fix: finalize external joint solver extraction"
```
