"""Main compiler class."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from nnc_py.codegen.npu_backend import NPUBackend
from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager

_STAGED_VALUE_PATTERN = re.compile(
    r"sram\|node\|\d+:(?P<node>[^|]+)\|tensor\|\d+:(?P<tensor>[^'\"\s)]+)"
)


def parse_memory_size(size_str: str) -> int:
    """Parse a memory size string to bytes.

    Args:
        size_str: Size string like "256K", "1M", "16MB", "512"

    Returns:
        Size in bytes.

    Examples:
        >>> parse_memory_size("256K")
        262144
        >>> parse_memory_size("1M")
        1048576
        >>> parse_memory_size("16MB")
        16777216
    """
    if size_str is None:
        return None

    size_str = size_str.strip().upper()

    # Match pattern: number followed by optional unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)?$', size_str)
    if not match:
        raise ValueError(f"Invalid memory size format: {size_str}")

    value = float(match.group(1))
    unit = match.group(2) or ""

    # Convert to bytes
    multipliers = {
        "": 1,
        "B": 1,
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "G": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
        "T": 1024 * 1024 * 1024 * 1024,
    }

    multiplier = multipliers.get(unit, 1)
    return int(value * multiplier)


def sanitize_compile_error_message(message: str) -> str:
    """Make scheduled-O3 failures readable for both API and CLI surfaces."""
    if not message:
        return "Compilation failed."

    def _replace(match: re.Match[str]) -> str:
        node_name = match.group("node")
        tensor_name = match.group("tensor")
        return f"scheduled tensor '{tensor_name}' at node '{node_name}'"

    sanitized = _STAGED_VALUE_PATTERN.sub(_replace, message)
    if sanitized.startswith("'") and sanitized.endswith("'"):
        sanitized = sanitized[1:-1]
    if "unknown step id:" in sanitized:
        sanitized = (
            "Scheduled pipeline metadata was inconsistent: "
            + sanitized.replace("unknown step id:", "missing schedule step", 1)
        )
    return sanitized


def _raise_sanitized_compile_error(exc: Exception) -> None:
    """Re-raise scheduled compile failures without leaking staged SRAM keys."""
    sanitized = sanitize_compile_error_message(str(exc))
    error_category = getattr(exc, "error_category", None)
    failure_status = getattr(exc, "failure_status", None)
    if sanitized == str(exc):
        raise exc
    error = RuntimeError(sanitized)
    if error_category is not None:
        setattr(error, "error_category", error_category)
    if failure_status is not None:
        setattr(error, "failure_status", failure_status)
    raise error.with_traceback(exc.__traceback__) from None


class Compiler:
    """ONNX to C compiler main class."""

    def __init__(
        self,
        target: str = "x86",
        opt_level: int = 0,
        memory_strategy: str = None,
        enable_constant_folding: bool = True,
        debug_mode: bool = False,
        cost_model_cli_command: str | list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Initialize the compiler.

        Args:
            target: Target architecture ("x86" or "npu").
            opt_level: Optimization level (0-3).
            memory_strategy: Memory allocation strategy (e.g., "basic").
            enable_constant_folding: Whether to enable ONNX constant folding (via onnxsim).
            debug_mode: Whether to enable debug mode with intermediate tensor dumps.
            cost_model_cli_command: Optional external CLI command for schedule-step cost estimation.
            metadata: Default compile-context metadata to merge into each compilation.
        """
        self.target = target
        self.opt_level = opt_level
        self.memory_strategy = memory_strategy
        self.enable_constant_folding = enable_constant_folding
        self.debug_mode = debug_mode
        self.cost_model_cli_command = cost_model_cli_command
        self.default_metadata = self._normalize_metadata_mapping(metadata)
        self.console = Console()

        # Initialize compiler stages
        self.frontend = None
        self.backend = self._create_backend(target)
        self.pass_manager = self._build_pass_manager()

    def compile(
        self,
        onnx_path: str,
        output_dir: str,
        entry_point: str = "nnc_run",
        max_memory: str = None,
        memory_strategy: str = None,
        cost_model_cli_command: str | list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Compile an ONNX model to C code.

        Args:
            onnx_path: Path to the ONNX model file.
            output_dir: Directory to write generated code.
            entry_point: Name for the entry point function.
            max_memory: Maximum fast memory size (e.g., "256K", "1M", "16MB").
            memory_strategy: Memory allocation strategy (overrides constructor).
            cost_model_cli_command: Optional external CLI command for schedule-step cost estimation.
            metadata: Additional compile-context metadata for this invocation.
        """
        # Parse max_memory if provided
        max_memory_bytes = None
        if max_memory is not None:
            max_memory_bytes = parse_memory_size(max_memory)
            self.console.print(
                f"  Max memory: {max_memory} ({max_memory_bytes:,} bytes)"
            )

        # Stage 1: Frontend parsing
        if self.frontend is None:
            from nnc_py.frontend import ONNXFrontend

            self.frontend = ONNXFrontend(
                enable_simplify=self.enable_constant_folding
            )
        with self.console.status("[bold green]Loading ONNX model..."):
            graph = self.frontend.load(onnx_path)
            self.console.print(
                f"✓ Loaded graph with {len(graph.nodes)} nodes"
            )
            self.console.print(
                f"  Inputs: {len(graph.inputs)}, Outputs: {len(graph.outputs)}"
            )
            self.console.print(
                f"  Constants: {len(graph.constants)}"
            )

        # Use method parameter if provided, otherwise use constructor value
        strategy = memory_strategy or self.memory_strategy
        compile_metadata = self._base_compile_metadata()
        compile_metadata.update(self._normalize_metadata_mapping(metadata))
        if cost_model_cli_command is not None:
            compile_metadata["cost_model_cli_command"] = cost_model_cli_command

        # Stage 2: Create compilation context
        ctx = CompileContext(graph, self.target, self.opt_level)
        ctx.metadata.update(compile_metadata)
        ctx.metadata["entry_point"] = entry_point

        # Store max_memory in context for memory planning pass
        if max_memory_bytes is not None:
            ctx.metadata["max_memory"] = max_memory_bytes

        # Store memory strategy in context
        if strategy is not None:
            ctx.metadata["memory_strategy"] = strategy
            self.console.print(f"  Memory strategy: {strategy}")

        # O3 always uses joint tiling schedule
        joint_tiling_schedule_enabled = self.opt_level >= 3
        ctx.metadata["enable_joint_tiling_schedule_contract"] = (
            joint_tiling_schedule_enabled
        )
        ctx.metadata["joint_tiling_schedule_contract_enabled"] = (
            joint_tiling_schedule_enabled
        )
        ctx.metadata["pipeline_scheduler_enabled"] = joint_tiling_schedule_enabled

        # Stage 3: Run optimization passes
        # Re-register passes for current opt_level to avoid accumulation
        # when the same Compiler instance is reused across multiple compilations
        self.pass_manager = self._build_pass_manager()

        try:
            if self.pass_manager.passes:
                with self.console.status(
                    "[bold yellow]Running optimization passes..."
                ):
                    self.pass_manager.run(ctx)
                    self._validate_joint_tiling_schedule_result(ctx)
                    self.console.print(
                        f"✓ Applied {len(self.pass_manager.applied_passes)} passes"
                    )
            else:
                self.console.print("ℹ No passes registered")

            # Stage 4: Code generation
            with self.console.status("[bold blue]Generating C code..."):
                artifacts = self.backend.generate(ctx)
                self.console.print(
                    f"✓ Generated {len(artifacts.files)} files"
                )

            # Stage 5: Write output files
            with self.console.status("[bold cyan]Writing output files..."):
                self._write_output(artifacts, output_dir, entry_point)
                self.console.print(f"✓ Output written to {output_dir}")
        except Exception as exc:
            _raise_sanitized_compile_error(exc)

        self.console.print()
        self.console.print(
            "[bold green]✓ Compilation successful![/bold green]"
        )

        # Print generated files
        output_path = Path(output_dir)
        self.console.print("\n[bold]Generated files:[/bold]")
        for artifact in artifacts.files:
            file_path = output_path / artifact.filename
            if file_path.exists():
                size = file_path.stat().st_size
                self.console.print(f"  {artifact.filename} ({size} bytes)")

    def _create_backend(self, target: str):
        """Create a backend instance for the target.

        Args:
            target: Target architecture name.

        Returns:
            Backend instance.

        Raises:
            ValueError: If target is unknown.
        """
        if target == "x86":
            return X86Backend(debug_mode=self.debug_mode)
        elif target == "npu":
            raise NotImplementedError(
                "NPU backend is not implemented yet. Use target='x86' for now."
            )
        else:
            raise ValueError(f"Unknown target: {target}")

    def _write_output(
        self,
        artifacts,
        output_dir: str,
        entry_point: str,
    ) -> None:
        """Write generated artifacts to output directory.

        Args:
            artifacts: Generated code artifacts.
            output_dir: Output directory path.
            entry_point: Entry point name (for metadata).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for artifact in artifacts.files:
            file_path = output_path / artifact.filename
            if artifact.file_type == "binary":
                file_path.write_bytes(artifact.content)
            else:
                file_path.write_text(artifact.content)

        # Store metadata
        artifacts.metadata["entry_point"] = entry_point
        artifacts.metadata["target"] = self.target
        artifacts.metadata["opt_level"] = self.opt_level

    def _build_pass_manager(self) -> PassManager:
        """Build a fresh pass manager for the requested optimization path."""
        pass_manager = PassManager()
        for pass_obj in self._get_passes():
            pass_manager.register(pass_obj)
        return pass_manager

    def _get_passes(self):
        """Return the pass sequence for the current optimization level."""
        if self.opt_level < 3:
            return PassManager.get_default_passes(self.opt_level)
        return PassManager.get_joint_tiling_schedule_o3_passes()

    def _base_compile_metadata(self) -> dict[str, Any]:
        """Return normalized default metadata for a compile invocation."""
        metadata = dict(self.default_metadata)
        if self.cost_model_cli_command is not None:
            metadata["cost_model_cli_command"] = self.cost_model_cli_command
        return metadata

    def _validate_joint_tiling_schedule_result(self, ctx: CompileContext) -> None:
        """Require the joint-contract O3 path to yield a solution or structured failure."""
        if self.opt_level < 3:
            return
        if not bool(
            ctx.metadata.get("enable_joint_tiling_schedule_contract")
            or ctx.metadata.get("joint_tiling_schedule_contract_enabled")
        ):
            return

        failure = ctx.joint_tiling_schedule_failure
        if failure is not None:
            raise _make_joint_failure_compile_error(failure)
        if ctx.joint_tiling_schedule_problem is None:
            raise RuntimeError(
                "O3 joint tiling schedule path failed (missing_joint_problem)."
            )
        if ctx.joint_tiling_schedule_solution is None:
            raise RuntimeError(
                "O3 joint tiling schedule path failed (missing_joint_solution)."
            )

    def _normalize_metadata_mapping(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Copy optional user metadata into a mutable dictionary."""
        if metadata is None:
            return {}
        return {str(key): value for key, value in metadata.items()}


def _format_joint_failure_compile_error(failure) -> str:
    """Format a structured joint failure as a concise compile error."""
    error_category, status = _joint_failure_error_fields(failure)

    reason = None
    diagnostics = failure.diagnostics
    if isinstance(diagnostics, Mapping):
        candidate = diagnostics.get("reason")
        if isinstance(candidate, str) and candidate:
            reason = candidate

    detail_parts = [f"error_category={error_category}", f"status={status}"]
    if reason is not None:
        detail_parts.append(f"reason={reason}")
    return f"O3 joint tiling schedule path failed ({', '.join(detail_parts)})."


def _make_joint_failure_compile_error(failure) -> RuntimeError:
    """Attach normalized joint failure fields to the compile exception."""
    error = RuntimeError(_format_joint_failure_compile_error(failure))
    error_category, status = _joint_failure_error_fields(failure)
    setattr(error, "error_category", error_category)
    setattr(error, "failure_status", status)
    return error


def _joint_failure_error_fields(failure) -> tuple[str, str]:
    """Return normalized compile-surface joint failure labels."""
    error_category = failure.error_category.value
    status = failure.status.value
    if status == "infeasible":
        error_category = "solver_reported_infeasible"
    return error_category, status
