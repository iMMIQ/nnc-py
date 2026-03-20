# P0 Consistency Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the public CLI/API surface, generated code behavior, docs, and test/config metadata so the project advertises only capabilities it actually supports.

**Architecture:** Keep the compiler pipeline intact and focus on contract cleanup. First lock desired behavior with tests, then update the CLI/compiler/codegen boundary, then sync docs and config to the real pipeline and test layout.

**Tech Stack:** Python 3.10+, pytest, Click, Rich, setuptools/pyproject, generated C code snapshots

---

### Task 1: Lock Public Contract Regressions

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_codegen_types.py`
- Modify: `tests/test_pass_manager.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- `--entry-name` changes the generated entry symbol or is no longer exposed
- `npu` is not advertised as a working compile target while the backend is placeholder-only
- pass-manager expectations reflect the actual O3 pipeline

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_cli.py tests/test_codegen_types.py tests/test_pass_manager.py`
Expected: FAIL on the new contract assertions before implementation changes.

- [ ] **Step 3: Write minimal implementation**

Update CLI/compiler/codegen behavior to satisfy the tests with the smallest coherent change set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_cli.py tests/test_codegen_types.py tests/test_pass_manager.py`
Expected: PASS

### Task 2: Sync Documentation and Packaging Metadata

**Files:**
- Modify: `README.md`
- Modify: `ROADMAP.md`
- Modify: `docs/OPTIMIZATION_PASSES.md`
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

Add focused tests where practical; otherwise use existing command/help output and config-sensitive tests to validate the changed contract.

- [ ] **Step 2: Run tests to verify they fail**

Run the smallest affected pytest selection.
Expected: FAIL or show mismatches that justify the doc/config changes.

- [ ] **Step 3: Write minimal implementation**

Update docs and metadata so they describe the actual file layout, pass ordering, snapshot setup, and supported targets.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_cli.py tests/test_pass_manager.py`
Expected: PASS

### Task 3: Verify Runtime-Test Layering

**Files:**
- Modify: `tests/test_common.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Add coverage for environment-sensitive runtime validation behavior only if it can be expressed without depending on ptrace-sensitive LSAN behavior.

- [ ] **Step 2: Run test to verify it fails**

Run the targeted test file only.
Expected: FAIL for the intended reason.

- [ ] **Step 3: Write minimal implementation**

Make runtime-test messaging and gating explicit so sanitizer failures caused by environment constraints do not masquerade as compiler regressions.

- [ ] **Step 4: Run test to verify it passes**

Run the targeted test file plus one representative snapshot test.
Expected: PASS or explicit skip in unsupported environments.
