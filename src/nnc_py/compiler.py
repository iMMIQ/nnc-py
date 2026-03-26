"""Main compiler class."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from nnc_py.codegen.npu_backend import NPUBackend
from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.ir.pipeline_schedule import PipelineScheduleResult, set_pipeline_schedule_result
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
    if sanitized == str(exc):
        raise exc
    raise RuntimeError(sanitized).with_traceback(exc.__traceback__) from None


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
        enable_pipeline_scheduler: bool | None = None,
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
            enable_pipeline_scheduler: Override for the O3 pipeline scheduler path.
            metadata: Default compile-context metadata to merge into each compilation.
        """
        self.target = target
        self.opt_level = opt_level
        self.memory_strategy = memory_strategy
        self.enable_constant_folding = enable_constant_folding
        self.debug_mode = debug_mode
        self.cost_model_cli_command = cost_model_cli_command
        self.enable_pipeline_scheduler = enable_pipeline_scheduler
        self.default_metadata = self._normalize_metadata_mapping(metadata)
        self.console = Console()

        # Initialize compiler stages
        self.frontend = ONNXFrontend(enable_simplify=enable_constant_folding)
        self.backend = self._create_backend(target)
        self.pass_manager = self._build_pass_manager(
            enable_pipeline_scheduler=self._resolve_pipeline_scheduler_enabled(
                self._base_compile_metadata(),
                explicit_enable_pipeline_scheduler=None,
            )
        )

    def compile(
        self,
        onnx_path: str,
        output_dir: str,
        entry_point: str = "nnc_run",
        max_memory: str = None,
        memory_strategy: str = None,
        cost_model_cli_command: str | list[str] | None = None,
        enable_pipeline_scheduler: bool | None = None,
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
            enable_pipeline_scheduler: Override for the O3 pipeline scheduler path.
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

        scheduler_enabled = self._resolve_pipeline_scheduler_enabled(
            ctx.metadata,
            explicit_enable_pipeline_scheduler=enable_pipeline_scheduler,
        )
        ctx.metadata["pipeline_scheduler_enabled"] = scheduler_enabled
        if self.opt_level >= 3 and not scheduler_enabled:
            fallback_reason = self._pipeline_scheduler_fallback_reason(
                ctx.metadata,
                explicit_enable_pipeline_scheduler=enable_pipeline_scheduler,
            )
            ctx.metadata["pipeline_scheduler_fallback"] = fallback_reason
            set_pipeline_schedule_result(
                ctx,
                PipelineScheduleResult(
                    feasible=False,
                    solver_name="disabled",
                    diagnostics={
                        "strategy": "serial",
                        "reason": (
                            "pipeline_scheduler_disabled"
                            if fallback_reason == "legacy_o3_disabled"
                            else "pipeline_scheduler_default_off"
                        ),
                    },
                ),
            )

        # Stage 3: Run optimization passes
        # Re-register passes for current opt_level to avoid accumulation
        # when the same Compiler instance is reused across multiple compilations
        self.pass_manager = self._build_pass_manager(
            enable_pipeline_scheduler=scheduler_enabled
        )

        try:
            if self.pass_manager.passes:
                with self.console.status(
                    "[bold yellow]Running optimization passes..."
                ):
                    self.pass_manager.run(ctx)
                    self._validate_scheduled_o3_result(ctx)
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

    def _build_pass_manager(self, *, enable_pipeline_scheduler: bool) -> PassManager:
        """Build a fresh pass manager for the requested optimization path."""
        pass_manager = PassManager()
        for pass_obj in self._get_passes(enable_pipeline_scheduler=enable_pipeline_scheduler):
            pass_manager.register(pass_obj)
        return pass_manager

    def _get_passes(self, *, enable_pipeline_scheduler: bool):
        """Return the pass sequence for the current optimization level."""
        if self.opt_level < 3:
            return PassManager.get_default_passes(self.opt_level)
        if enable_pipeline_scheduler:
            return PassManager.get_scheduled_o3_passes()
        return PassManager.get_conservative_o3_passes()

    def _base_compile_metadata(self) -> dict[str, Any]:
        """Return normalized default metadata for a compile invocation."""
        metadata = dict(self.default_metadata)
        if self.cost_model_cli_command is not None:
            metadata["cost_model_cli_command"] = self.cost_model_cli_command
        if self.enable_pipeline_scheduler is not None:
            metadata["enable_pipeline_scheduler"] = self.enable_pipeline_scheduler
        return metadata

    def _resolve_pipeline_scheduler_enabled(
        self,
        metadata: Mapping[str, Any],
        *,
        explicit_enable_pipeline_scheduler: bool | None,
    ) -> bool:
        """Resolve the scheduler-path toggle with conservative defaults."""
        if explicit_enable_pipeline_scheduler is not None:
            return bool(explicit_enable_pipeline_scheduler)
        if "enable_pipeline_scheduler" in metadata:
            return bool(metadata["enable_pipeline_scheduler"])
        if "disable_pipeline_scheduler" in metadata:
            return not bool(metadata["disable_pipeline_scheduler"])
        return self.opt_level >= 3

    def _pipeline_scheduler_fallback_reason(
        self,
        metadata: Mapping[str, Any],
        *,
        explicit_enable_pipeline_scheduler: bool | None,
    ) -> str:
        """Return a deterministic label describing why the legacy path ran."""
        if explicit_enable_pipeline_scheduler is False:
            return "legacy_o3_disabled"
        if explicit_enable_pipeline_scheduler is True:
            return "legacy_o3_default"
        if "enable_pipeline_scheduler" in metadata:
            return (
                "legacy_o3_default"
                if bool(metadata["enable_pipeline_scheduler"])
                else "legacy_o3_disabled"
            )
        if bool(metadata.get("disable_pipeline_scheduler")):
            return "legacy_o3_disabled"
        return "legacy_o3_default"

    def _validate_scheduled_o3_result(self, ctx: CompileContext) -> None:
        """Require O3 scheduled compilation to produce a planned schedule."""
        if self.opt_level < 3:
            return
        if not bool(ctx.metadata.get("pipeline_scheduler_enabled")):
            return

        problem = ctx.pipeline_schedule_problem
        result = ctx.pipeline_schedule_result
        scheduled_memory_plan = ctx.metadata.get("scheduled_memory_plan")
        if (
            problem is not None
            and result is not None
            and result.feasible
            and scheduled_memory_plan is not None
        ):
            return

        reason = "missing_scheduled_memory_plan"
        if result is None:
            reason = "missing_schedule_result"
        if result is not None:
            diagnostic_reason = result.diagnostics.get("reason")
            if isinstance(diagnostic_reason, str) and diagnostic_reason:
                reason = diagnostic_reason
            elif result.solver_name:
                reason = result.solver_name
            elif not result.feasible:
                reason = "infeasible_schedule_result"
            elif scheduled_memory_plan is None:
                reason = "missing_scheduled_memory_plan"
        elif problem is None:
            reason = "missing_schedule_problem"

        raise RuntimeError(
            "O3 scheduled pipeline path failed "
            f"({reason})."
        )

    def _normalize_metadata_mapping(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Copy optional user metadata into a mutable dictionary."""
        if metadata is None:
            return {}
        return {str(key): value for key, value in metadata.items()}
