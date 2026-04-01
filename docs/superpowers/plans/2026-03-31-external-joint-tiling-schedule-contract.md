# External Joint Tiling Schedule Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned external joint tiling/schedule contract, a compiler-owned problem builder + validator + materializer, and a gated solver adapter that can hand the optimization problem to an external team without exposing compiler-engineering internals.

**Architecture:** Introduce a new external-facing IR above the current `PipelineScheduleProblem` / `PipelineScheduleResult` layer. Build `Region` / `Recipe` / `Value` / `Action` problems from existing tiled-lowering metadata, solve them through a pluggable adapter, validate the returned solution, then materialize the result back into the current scheduled pipeline path so `ScheduledMemoryPlanningPass` and codegen can keep working.

**Tech Stack:** Python 3.11+, dataclasses, existing `nnc_py` pass pipeline, pytest, current scheduled O3 path, JSON stdin/stdout adapter for external solvers.

**Spec:** `docs/superpowers/specs/2026-03-31-external-joint-tiling-schedule-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nnc_py/ir/joint_tiling_schedule.py` | Create | External problem / solution / failure dataclasses, enums, JSON helpers, metadata accessors |
| `src/nnc_py/ir/context.py` | Modify | Typed accessors for joint problem / solution / failure metadata |
| `src/nnc_py/joint_schedule/__init__.py` | Create | Package exports |
| `src/nnc_py/joint_schedule/regions.py` | Create | Build `Region` interface objects from graph + tiled execution metadata |
| `src/nnc_py/joint_schedule/recipes.py` | Create | Enumerate recipes, values, actions, optional spill/reload candidates, dependency edges, boundary constraints |
| `src/nnc_py/joint_schedule/solver.py` | Create | Solver interface, external CLI adapter, baseline internal solver |
| `src/nnc_py/joint_schedule/validation.py` | Create | Problem validation + solution validation + failure categorization |
| `src/nnc_py/joint_schedule/materialize.py` | Create | Lower validated joint solution into current `PipelineScheduleProblem` / `PipelineScheduleResult` |
| `src/nnc_py/passes/joint_tiling_schedule.py` | Create | Passes for problem build, solve, and materialization |
| `src/nnc_py/passes/base.py` | Modify | Register gated joint-contract O3 path |
| `src/nnc_py/passes/__init__.py` | Modify | Export the new passes |
| `src/nnc_py/compiler.py` | Modify | Metadata flag / solver-command wiring / failure handling for the new path |
| `tests/test_joint_tiling_schedule_ir.py` | Create | External IR contract tests |
| `tests/test_joint_tiling_schedule_problem_builder.py` | Create | Region / recipe / value / edge construction tests |
| `tests/test_joint_tiling_schedule_solver.py` | Create | Baseline solver + CLI adapter contract tests |
| `tests/test_joint_tiling_schedule_validation.py` | Create | Validator success / failure category tests |
| `tests/test_joint_tiling_schedule_materialize.py` | Create | Materialization back into internal schedule IR |
| `tests/fake_joint_solver.py` | Create | Tiny stdout-driven fake solver for CLI adapter and e2e tests |
| `tests/test_pipeline_pass_integration.py` | Modify | Scheduled O3 pass ordering + gated path coverage |
| `tests/test_pipeline_scheduler_e2e.py` | Modify | End-to-end compile behavior through the new joint-contract path |

---

### Task 1: Add External Joint Contract IR And Context Accessors

**Files:**
- Create: `src/nnc_py/ir/joint_tiling_schedule.py`
- Modify: `src/nnc_py/ir/context.py`
- Test: `tests/test_joint_tiling_schedule_ir.py`

- [ ] **Step 1: Write the failing IR tests**

```python
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.joint_tiling_schedule import (
    JointAction,
    JointBoundaryConstraint,
    JointDependencyEdge,
    JointProblem,
    JointRecipe,
    JointRegion,
    JointResource,
    JointSelectedRecipe,
    JointSolution,
    JointValue,
    JointValueTier,
    get_joint_tiling_schedule_problem,
    set_joint_tiling_schedule_problem,
)


def test_joint_problem_keeps_required_top_level_arrays():
    problem = JointProblem(
        schema_version="joint_tiling_schedule_problem_v1",
        regions=(JointRegion(region_id="r0", kind="single_op", input_value_ids=(), output_value_ids=("v0",)),),
        recipes=(JointRecipe(...),),
        values=(JointValue(...),),
        actions=(JointAction(...),),
        boundary_constraints=(),
        dependency_edges=(),
        resources=(JointResource(resource_kind="DMA", slot_count=1),),
        sram_capacity_bytes=1024,
        objective="min_makespan",
    )
    assert problem.objective == "min_makespan"


def test_context_round_trips_joint_problem():
    ctx = CompileContext(Graph("g"), target="x86", optimization_level=3)
    problem = JointProblem(...)
    set_joint_tiling_schedule_problem(ctx, problem)
    assert get_joint_tiling_schedule_problem(ctx) is problem


def test_joint_problem_round_trips_json_and_ignores_unknown_fields():
    payload = JointProblem(...).to_json()
    payload["unknown_extension"] = {"future": True}
    restored = JointProblem.from_json(payload)
    assert restored.schema_version == "joint_tiling_schedule_problem_v1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_tiling_schedule_ir.py -v`
Expected: FAIL with import errors because the external IR module and context accessors do not exist yet.

- [ ] **Step 3: Implement the external IR module**

Create `src/nnc_py/ir/joint_tiling_schedule.py` with:

- enums and literals for:
  - `JointValueTier`
  - action kind strings
  - dependency edge kind strings
  - failure categories
- immutable dataclasses for:
  - `JointRegion`
  - `JointRecipe`
  - `JointValue`
  - `JointAction`
  - `JointBoundaryConstraint`
  - `JointDependencyEdge`
  - `JointResource`
  - `JointProblem`
  - `JointSelectedRecipe`
  - `JointScheduledAction`
  - `JointResidencyWindow`
  - `JointSolution`
  - `JointFailure`
- JSON-ready `to_json()` helpers and strict constructor validation for:
  - schema version literal
  - required field presence
  - enum values
  - ID uniqueness
  - nullability rules from the spec
- `from_json()` loaders for:
  - `JointProblem`
  - `JointSolution`
  - `JointFailure`
- recognized-v1 unknown-field ignore behavior during `from_json()`

Use the same pattern as `pipeline_schedule.py`: typed metadata keys, frozen payloads, and explicit `get_*` / `set_*` helpers.

- [ ] **Step 4: Add typed context accessors**

Update `src/nnc_py/ir/context.py` to expose:

- `joint_tiling_schedule_problem`
- `joint_tiling_schedule_solution`
- `joint_tiling_schedule_failure`

and matching `get_*` helper methods.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_joint_tiling_schedule_ir.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_joint_tiling_schedule_ir.py src/nnc_py/ir/joint_tiling_schedule.py src/nnc_py/ir/context.py
git commit -m "feat: add external joint tiling schedule IR"
```

### Task 2: Build Regions, Recipes, And Full External Problems

**Files:**
- Create: `src/nnc_py/joint_schedule/__init__.py`
- Create: `src/nnc_py/joint_schedule/regions.py`
- Create: `src/nnc_py/joint_schedule/recipes.py`
- Test: `tests/test_joint_tiling_schedule_problem_builder.py`

- [ ] **Step 1: Write the failing problem-builder tests**

```python
from nnc_py.joint_schedule.regions import build_joint_regions
from nnc_py.joint_schedule.recipes import build_joint_problem


def test_region_builder_emits_region_interfaces_from_execution_plans(ctx_with_tiled_group):
    regions = build_joint_regions(ctx_with_tiled_group)
    assert [region.region_id for region in regions] == ["conv0_group"]
    assert regions[0].output_value_ids


def test_problem_builder_emits_boundary_constraints_and_actions(ctx_with_two_regions):
    problem = build_joint_problem(ctx_with_two_regions)
    assert problem.regions
    assert problem.recipes
    assert problem.values
    assert problem.actions
    assert problem.boundary_constraints
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_tiling_schedule_problem_builder.py -v`
Expected: FAIL because the new builder package does not exist yet.

- [ ] **Step 3: Implement `build_joint_regions()`**

Create `src/nnc_py/joint_schedule/regions.py` with a builder that:

- walks `ctx.node_execution_plans`
- collapses tiled producer chains already surfaced by `TiledLoweringPass`
- emits one `JointRegion` per:
  - single tileable op
  - fused/tiled group already represented in current metadata
- populates:
  - `region_id`
  - `kind`
  - `member_nodes`
  - `input_value_ids`
  - `output_value_ids`

Do not invent new fusion logic here. Reuse existing tiled-group metadata from the current path.

- [ ] **Step 4: Implement `build_joint_problem()`**

Create `src/nnc_py/joint_schedule/recipes.py` with a top-level builder that:

- consumes `CompileContext`
- calls `build_joint_regions()`
- emits:
  - `JointRecipe` candidates
  - `JointValue` objects
  - `JointAction` objects
  - predeclared optional spill / reload action instances
  - `JointDependencyEdge` objects
  - `JointBoundaryConstraint` objects
  - `JointResource` defaults
- reuses current execution-plan / schedule-step estimation logic instead of inventing new hardware assumptions

For the first implementation, keep recipe enumeration intentionally small:

- one “baseline” recipe per simple region
- one extra recipe only where current tiled metadata already provides a real alternative

Make optional transfer generation explicit here:

- replace the current implicit role of `ScheduledMemoryExpansionPass`
- predeclare every optional spill / reload candidate required by the v1 spec
- attach legality and dependency structure before solving

This satisfies the contract without exploding scope.

- [ ] **Step 5: Encode the consistency invariants**

Inside the builder, enforce the spec’s three dataflow views:

- region interface values
- value producer / consumer actions
- action reads / writes

If the builder cannot make these agree, raise a typed invalid-problem error rather than silently emitting partial metadata.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_joint_tiling_schedule_problem_builder.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_joint_tiling_schedule_problem_builder.py src/nnc_py/joint_schedule/__init__.py src/nnc_py/joint_schedule/regions.py src/nnc_py/joint_schedule/recipes.py
git commit -m "feat: build external joint tiling schedule problems"
```

### Task 3: Add Solver Interface, CLI Adapter, And Failure-Payload Handling

**Files:**
- Create: `src/nnc_py/joint_schedule/solver.py`
- Create: `tests/fake_joint_solver.py`
- Test: `tests/test_joint_tiling_schedule_solver.py`

- [ ] **Step 1: Write the failing solver tests**

```python
from nnc_py.joint_schedule.solver import (
    CliJointScheduleSolver,
)


def test_cli_solver_parses_failure_payload(tmp_path, minimal_joint_problem):
    solver = CliJointScheduleSolver(["python3", "tests/fake_joint_solver.py", "infeasible"])
    result = solver.solve(minimal_joint_problem)
    assert result.schema_version == "joint_tiling_schedule_failure_v1"
    assert result.status == "infeasible"


def test_cli_solver_parses_timeout_payload(tmp_path, minimal_joint_problem):
    solver = CliJointScheduleSolver(["python3", "tests/fake_joint_solver.py", "timeout"])
    result = solver.solve(minimal_joint_problem)
    assert result.schema_version == "joint_tiling_schedule_failure_v1"
    assert result.status == "timeout"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_tiling_schedule_solver.py -v`
Expected: FAIL because the solver module does not exist.

- [ ] **Step 3: Add the solver interface**

Create `src/nnc_py/joint_schedule/solver.py` with:

```python
class JointScheduleSolver(ABC):
    @abstractmethod
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        raise NotImplementedError
```

Also add:

- `CliJointScheduleSolver`

- [ ] **Step 4: Implement the CLI adapter**

The CLI adapter should:

- serialize `JointProblem.to_json()`
- send it to stdin as one JSON blob
- parse either:
  - `joint_tiling_schedule_solution_v1`
  - `joint_tiling_schedule_failure_v1`
- treat non-zero process exit as transport / solver crash
- attach stderr to diagnostics when possible

Reuse the same robustness style already used in `src/nnc_py/cost_model/cli.py`.

- [ ] **Step 5: Add the fake solver helper**

Create `tests/fake_joint_solver.py` that:

- reads one JSON payload from stdin
- emits either:
  - a minimal valid `joint_tiling_schedule_solution_v1`
  - a minimal valid `joint_tiling_schedule_failure_v1`
- supports at least these argv modes:
  - `solution`
  - `infeasible`
  - `timeout`
  - `error`

- [ ] **Step 6: Add explicit failure-payload mapping tests**

Extend `tests/test_joint_tiling_schedule_solver.py` to verify:

- `status="infeasible"` maps to a `JointFailure`
- `status="timeout"` maps to a `JointFailure`
- `status="error"` maps to a `JointFailure`
- non-zero exit still counts as transport failure, not a valid failure payload

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_joint_tiling_schedule_solver.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/fake_joint_solver.py tests/test_joint_tiling_schedule_solver.py src/nnc_py/joint_schedule/solver.py
git commit -m "feat: add joint tiling schedule solver adapters"
```

### Task 4: Add Problem Validation And Solution Validation

**Files:**
- Create: `src/nnc_py/joint_schedule/validation.py`
- Test: `tests/test_joint_tiling_schedule_validation.py`

- [ ] **Step 1: Write the failing validator tests**

```python
from nnc_py.joint_schedule.validation import (
    validate_joint_problem,
    validate_joint_solution,
)


def test_validator_rejects_missing_boundary_constraint(problem_missing_boundary):
    failure = validate_joint_problem(problem_missing_boundary)
    assert failure.status == "invalid_problem"


def test_validator_rejects_overlapping_same_resource_actions(problem, bad_solution):
    failure = validate_joint_solution(problem, bad_solution)
    assert failure.error_category == "resource_overlap"


def test_validator_rejects_incompatible_recipe_boundary(problem, bad_solution):
    failure = validate_joint_solution(problem, bad_solution)
    assert failure.error_category == "incompatible_recipe_boundary"


def test_validator_rejects_missing_final_output(problem, bad_solution):
    failure = validate_joint_solution(problem, bad_solution)
    assert failure.error_category == "incomplete_solution"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_tiling_schedule_validation.py -v`
Expected: FAIL because the validator module does not exist.

- [ ] **Step 3: Implement `validate_joint_problem()`**

The problem validator must check:

- required schema literals
- ID uniqueness
- action activation invariants
- region adjacency / boundary completeness
- producer / consumer / reads / writes consistency
- legal tier combinations
- exactly one required resource entry for each v1 resource kind

- [ ] **Step 4: Implement `validate_joint_solution()`**

The solution validator must check:

- exactly one selected recipe per region
- active-action derivation
- dependency satisfaction using `end(src) <= start(dst)`
- same-resource non-overlap
- legal transfer activation
- normalized residency windows
- selected adjacent recipes satisfy boundary constraints
- final required outputs are produced and reachable
- `must_keep` continuous residency
- `allows_multiple_sram_windows` restrictions
- required initial SRAM windows that start at `0`
- required final SRAM windows that end at `objective_value`
- full-interval residency for compute reads
- spill / reload gap justification
- SRAM capacity from:
  - resident `Value.size_bytes`
  - active `Action.temp_bytes`
- objective value equals makespan

Return `JointFailure` objects with the exact v1 `error_category` enum when validation fails.

- [ ] **Step 5: Add failure-category coverage**

Add one focused failing test per standardized `error_category` enum that the validator can emit from compiler-side checks:

- `invalid_solution`
- `incomplete_solution`
- `dependency_violation`
- `resource_overlap`
- `sram_capacity_exceeded`
- `illegal_transfer`
- `incompatible_recipe_boundary`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_joint_tiling_schedule_validation.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_joint_tiling_schedule_validation.py src/nnc_py/joint_schedule/validation.py
git commit -m "feat: validate external joint schedule problems and solutions"
```

### Task 5: Materialize Joint Solutions And Add The Baseline Internal Solver

**Files:**
- Create: `src/nnc_py/joint_schedule/materialize.py`
- Modify: `src/nnc_py/joint_schedule/solver.py`
- Test: `tests/test_joint_tiling_schedule_materialize.py`
- Test: `tests/test_joint_tiling_schedule_solver.py`

- [ ] **Step 1: Write the failing materialization tests**

```python
from nnc_py.joint_schedule.materialize import materialize_joint_solution
from nnc_py.ir.pipeline_schedule import PipelineScheduleProblem, PipelineScheduleResult


def test_materialize_joint_solution_returns_internal_schedule_pair(valid_joint_problem, valid_joint_solution):
    problem, result = materialize_joint_solution(valid_joint_problem, valid_joint_solution)
    assert isinstance(problem, PipelineScheduleProblem)
    assert isinstance(result, PipelineScheduleResult)
    assert result.feasible is True


def test_baseline_solver_returns_solution_shape(valid_joint_problem):
    result = BaselineJointScheduleSolver().solve(valid_joint_problem)
    assert result.schema_version == "joint_tiling_schedule_solution_v1"
    assert result.selected_recipes
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_joint_tiling_schedule_materialize.py -v`
Expected: FAIL because the materializer does not exist.

- [ ] **Step 3: Implement the materializer**

Create `src/nnc_py/joint_schedule/materialize.py` with helpers that:

- expand active joint actions into internal `ScheduleStep` / `TransferStep`
- derive internal `ScheduledValue` records
- derive `ResidencyWindow`s
- build:
  - `PipelineScheduleProblem`
  - `PipelineScheduleResult`

Keep the internal structures compatible with:

- `ScheduledMemoryPlanningPass`
- current x86 scheduled codegen

Do not rewrite `ScheduledMemoryPlanningPass` in this task. Adapt the new output to what it already consumes.

- [ ] **Step 4: Implement the baseline internal solver**

Now that problem validation and materialization rules exist, add
`BaselineJointScheduleSolver` in `src/nnc_py/joint_schedule/solver.py`.

Keep it deliberately simple and deterministic:

- choose the first recipe per region
- schedule mandatory actions in topological order
- schedule no optional spill/reload actions
- derive only the minimum valid residency windows needed by the spec

This solver exists only as the internal regression baseline required by the
spec migration plan.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_joint_tiling_schedule_materialize.py -v`
Expected: PASS

- [ ] **Step 6: Regression check existing scheduled memory planning and baseline solver**

Run: `pytest tests/test_joint_tiling_schedule_solver.py tests/test_scheduled_memory_planning.py tests/test_pipeline_scheduler.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_joint_tiling_schedule_materialize.py tests/test_joint_tiling_schedule_solver.py src/nnc_py/joint_schedule/materialize.py src/nnc_py/joint_schedule/solver.py
git commit -m "feat: materialize joint schedules and add baseline solver"
```

### Task 6: Add Passes And Gate The New Path In The Compiler

**Files:**
- Create: `src/nnc_py/passes/joint_tiling_schedule.py`
- Modify: `src/nnc_py/passes/base.py`
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/compiler.py`
- Test: `tests/test_pipeline_pass_integration.py`

- [ ] **Step 1: Write the failing pass-integration tests**

```python
def test_joint_contract_pass_order_is_gated_and_explicit():
    names = [p.__class__.__name__ for p in PassManager.get_joint_tiling_schedule_o3_passes()]
    assert names.index("TiledLoweringPass") < names.index("JointTilingScheduleProblemPass")
    assert names.index("JointTilingScheduleProblemPass") < names.index("JointTilingScheduleSolvePass")
    assert names.index("JointTilingScheduleSolvePass") < names.index("JointTilingScheduleMaterializationPass")


def test_compiler_can_enable_joint_contract_path(monkeypatch, tmp_path):
    ctx = _compile_graph(
        monkeypatch,
        tmp_path,
        metadata={"enable_joint_tiling_schedule_contract": True},
    )
    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_pass_integration.py -v`
Expected: FAIL because the new passes and compiler flag do not exist.

- [ ] **Step 3: Implement the new passes**

Create `src/nnc_py/passes/joint_tiling_schedule.py` with:

- `JointTilingScheduleProblemPass`
- `JointTilingScheduleSolvePass`
- `JointTilingScheduleMaterializationPass`

Responsibilities:

- build and validate the joint problem
- solve through the configured solver
- validate the returned solution
- materialize to current internal schedule artifacts

- [ ] **Step 4: Register the gated O3 path**

Update `src/nnc_py/passes/base.py` to add:

- `get_joint_tiling_schedule_o3_passes()`

Initial pass order:

1. existing O3 passes through `TiledLoweringPass`
2. `JointTilingScheduleProblemPass`
3. `JointTilingScheduleSolvePass`
4. `JointTilingScheduleMaterializationPass`
5. `LivenessAnalysisPass`
6. `ScheduledMemoryPlanningPass`

Do not remove the current scheduled O3 path yet. Keep the new path behind a metadata flag until parity is demonstrated.

- [ ] **Step 5: Wire compiler metadata**

Update `src/nnc_py/compiler.py` to support:

- `enable_joint_tiling_schedule_contract`
- optional solver command metadata, e.g. `joint_tiling_schedule_solver_command`

If the new path is enabled and produces a `JointFailure`, raise a sanitized compile error that preserves the standardized failure category.
Add explicit coverage for solver-returned infeasibility so a failure payload with
`status="infeasible"` is surfaced to compiler consumers as
`error_category="solver_reported_infeasible"`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline_pass_integration.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_pipeline_pass_integration.py src/nnc_py/passes/joint_tiling_schedule.py src/nnc_py/passes/base.py src/nnc_py/passes/__init__.py src/nnc_py/compiler.py
git commit -m "feat: gate external joint tiling schedule path in compiler"
```

### Task 7: Add End-To-End Coverage And Freeze The External Contract

**Files:**
- Modify: `tests/test_pipeline_scheduler_e2e.py`
- Create: `tests/fake_joint_solver.py`
- Test: `tests/test_pipeline_scheduler_e2e.py`

- [ ] **Step 1: Write the failing end-to-end tests**

Add coverage for:

- compile with `enable_joint_tiling_schedule_contract=True`
- compile with a fake external solver command that returns:
  - a valid `joint_tiling_schedule_solution_v1`
  - a valid `joint_tiling_schedule_failure_v1`
- verify generated build still succeeds when the materialized internal schedule is feasible

Example test shape:

```python
def test_joint_contract_path_materializes_and_builds(tmp_path):
    ctx, output_dir = _compile_model(
        tmp_path,
        enable_pipeline_scheduler=True,
        metadata={"enable_joint_tiling_schedule_contract": True},
    )
    assert ctx.joint_tiling_schedule_problem is not None
    assert ctx.joint_tiling_schedule_solution is not None
    assert ctx.pipeline_schedule_result is not None
    _build_generated_x86_source(output_dir)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline_scheduler_e2e.py -v`
Expected: FAIL because the new path is not fully wired end to end yet.

- [ ] **Step 3: Implement the minimum e2e glue**

Adjust any remaining import / metadata / emitted-comment plumbing so the joint-contract path:

- stores the joint problem / solution on the context
- materializes valid internal schedule metadata
- still feeds the current scheduled codegen

Do not redesign x86 codegen here. Only make the new path compatible with the existing downstream contract.

- [ ] **Step 4: Run the target tests**

Run: `pytest tests/test_pipeline_scheduler_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Run the focused regression suite**

Run:

```bash
pytest \
  tests/test_joint_tiling_schedule_ir.py \
  tests/test_joint_tiling_schedule_problem_builder.py \
  tests/test_joint_tiling_schedule_solver.py \
  tests/test_joint_tiling_schedule_validation.py \
  tests/test_joint_tiling_schedule_materialize.py \
  tests/test_pipeline_pass_integration.py \
  tests/test_pipeline_scheduler_e2e.py \
  tests/test_scheduled_memory_planning.py \
  tests/test_pipeline_scheduler.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline_scheduler_e2e.py
git commit -m "test: add end-to-end coverage for joint tiling schedule path"
```

### Task 8: Cleanup, Export API, And Document Operator Limits

**Files:**
- Modify: `src/nnc_py/passes/__init__.py`
- Modify: `src/nnc_py/joint_schedule/__init__.py`
- Modify: `README.md`
- Test: `tests/test_project_metadata.py`

- [ ] **Step 1: Write the failing metadata / export test**

```python
def test_joint_schedule_modules_are_importable():
    import nnc_py.joint_schedule
    from nnc_py.passes import JointTilingScheduleProblemPass
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_project_metadata.py -v`
Expected: FAIL if exports or docs references are missing.

- [ ] **Step 3: Export the public API**

Update:

- `src/nnc_py/joint_schedule/__init__.py`
- `src/nnc_py/passes/__init__.py`

to export the new package and pass names cleanly.

- [ ] **Step 4: Update top-level documentation**

Add a short section to `README.md` covering:

- the new external joint contract path
- the opt-in metadata flag
- the fact that v1 is currently limited to regions/recipes the compiler already knows how to build

Keep this short. Do not duplicate the spec.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_project_metadata.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/nnc_py/joint_schedule/__init__.py src/nnc_py/passes/__init__.py README.md tests/test_project_metadata.py
git commit -m "docs: expose and document joint tiling schedule contract"
```
