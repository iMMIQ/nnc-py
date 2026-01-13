"""Main compiler class."""

from pathlib import Path
from typing import Optional

from rich.console import Console

from nnc_py.codegen.npu_backend import NPUBackend
from nnc_py.codegen.x86_backend import X86Backend
from nnc_py.frontend.onnx_loader import ONNXFrontend
from nnc_py.ir.context import CompileContext
from nnc_py.passes.base import PassManager


class Compiler:
    """ONNX to C compiler main class."""

    def __init__(self, target: str = "x86", opt_level: int = 0):
        """Initialize the compiler.

        Args:
            target: Target architecture ("x86" or "npu").
            opt_level: Optimization level (0-3).
        """
        self.target = target
        self.opt_level = opt_level
        self.console = Console()

        # Initialize compiler stages
        self.frontend = ONNXFrontend()
        self.pass_manager = PassManager()
        self.backend = self._create_backend(target)

        # Register default passes based on optimization level
        for pass_obj in PassManager.get_default_passes(opt_level):
            self.pass_manager.register(pass_obj)

    def compile(
        self,
        onnx_path: str,
        output_dir: str,
        entry_point: str = "main",
    ) -> None:
        """Compile an ONNX model to C code.

        Args:
            onnx_path: Path to the ONNX model file.
            output_dir: Directory to write generated code.
            entry_point: Name for the entry point function.
        """
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

        # Stage 2: Create compilation context
        ctx = CompileContext(graph, self.target, self.opt_level)

        # Stage 3: Run optimization passes
        if self.opt_level > 0 and self.pass_manager.passes:
            with self.console.status(
                "[bold yellow]Running optimization passes..."
            ):
                self.pass_manager.run(ctx)
                self.console.print(
                    f"✓ Applied {len(self.pass_manager.applied_passes)} passes"
                )
        else:
            self.console.print("ℹ Skipping optimization passes (O0)")

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
            return X86Backend()
        elif target == "npu":
            return NPUBackend()
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
            file_path.write_text(artifact.content)

        # Store metadata
        artifacts.metadata["entry_point"] = entry_point
        artifacts.metadata["target"] = self.target
        artifacts.metadata["opt_level"] = self.opt_level
