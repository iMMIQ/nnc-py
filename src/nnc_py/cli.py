"""Command-line interface for NNC compiler."""

import click
from rich.console import Console
from rich.table import Table

from nnc_py import Compiler

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """NNC - Neural Network Compiler

    Compile ONNX models to C code for x86 and NPU targets.
    """
    pass


@main.command()
@click.argument("onnx_model", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./output",
    help="Output directory for generated code",
)
@click.option(
    "-t",
    "--target",
    type=click.Choice(["x86", "npu"]),
    default="x86",
    help="Target architecture",
)
@click.option(
    "-O",
    "--opt-level",
    type=click.IntRange(0, 3),
    default=0,
    help="Optimization level (0-3)",
)
@click.option(
    "--entry-name",
    default="main",
    help="Entry point function name",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--max-memory",
    type=str,
    default=None,
    help="Maximum fast memory size (e.g., 256K, 1M, 16MB). "
         "If model requires more, tensors will be spilled to slow memory.",
)
@click.option(
    "--memory-strategy",
    type=str,
    default="basic",
    help="Memory allocation strategy: basic (default) - Simple sequential allocation with spill-all",
)
@click.option(
    "--enable-constant-folding/--disable-constant-folding",
    default=True,
    help="Enable/disable ONNX constant folding optimization (default: enabled)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode: dump intermediate tensor outputs for comparison with ONNX Runtime",
)
def compile(onnx_model, output, target, opt_level, entry_name, verbose, max_memory, memory_strategy, enable_constant_folding, debug):
    """Compile an ONNX model to C code.

    Example:
        nnc compile model.onnx -o ./build -t x86 -O1
        nnc compile model.onnx -o ./build -O2 --max-memory 256K
        nnc compile model.onnx --memory-strategy basic
        nnc compile model.onnx --disable-constant-folding
    """
    console.print(f"[bold blue]Compiling[/bold blue]: {onnx_model}")
    console.print(f"[bold blue]Target[/bold blue]: {target}")
    console.print(f"[bold blue]Optimization[/bold blue]: O{opt_level}")
    console.print(f"[bold blue]Memory Strategy[/bold blue]: {memory_strategy}")
    console.print(f"[bold blue]Constant Folding[/bold blue]: {'enabled' if enable_constant_folding else 'disabled'}")
    console.print(f"[bold blue]Debug Mode[/bold blue]: {'enabled' if debug else 'disabled'}")
    if max_memory:
        console.print(f"[bold blue]Max Memory[/bold blue]: {max_memory}")
    console.print()

    try:
        compiler = Compiler(
            target=target,
            opt_level=opt_level,
            enable_constant_folding=enable_constant_folding,
            debug_mode=debug,
        )
        compiler.compile(
            onnx_path=onnx_model,
            output_dir=output,
            entry_point=entry_name,
            max_memory=max_memory,
            memory_strategy=memory_strategy,
        )

    except Exception as e:
        console.print(f"[bold red]✗ Compilation failed:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise click.ClickException(str(e))


@main.command()
@click.argument("onnx_model", type=click.Path(exists=True))
def info(onnx_model):
    """Display information about an ONNX model.

    Example:
        nnc info model.onnx
    """
    import onnx

    try:
        model = onnx.load(onnx_model)
        graph = model.graph

        table = Table(title=f"Model Information: {onnx_model}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model Name", graph.name or "unnamed")
        table.add_row("Nodes", str(len(graph.node)))
        table.add_row("Inputs", str(len(graph.input)))
        table.add_row("Outputs", str(len(graph.output)))
        table.add_row("Initializers", str(len(graph.initializer)))

        console.print(table)
        console.print()

        # Display input/output tensors
        if graph.input:
            console.print("[bold]Inputs:[/bold]")
            for inp in graph.input:
                shape_str = _format_tensor_shape(inp)
                console.print(f"  - {inp.name}: {shape_str}")

        if graph.output:
            console.print("[bold]Outputs:[/bold]")
            for out in graph.output:
                shape_str = _format_tensor_shape(out)
                console.print(f"  - {out.name}: {shape_str}")

        # Display operator summary
        op_counts = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        if op_counts:
            console.print()
            console.print("[bold]Operators:[/bold]")
            for op_type, count in sorted(op_counts.items()):
                console.print(f"  - {op_type}: {count}")

    except Exception as e:
        console.print(f"[bold red]✗ Error loading model:[/bold red] {e}")
        raise click.ClickException(str(e))


@main.command()
def targets():
    """List available target architectures."""
    table = Table(title="Available Targets")
    table.add_column("Target", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("x86", "Generate code for x86 simulation")
    table.add_row("npu", "Generate code for NPU acceleration")

    console.print(table)


@main.group()
def debug():
    """Debug mode commands for comparing NNC output with ONNX Runtime."""
    pass


@debug.command()
@click.argument("nnc_debug_file", type=click.Path(exists=True))
@click.argument("onnx_model", type=click.Path(exists=True))
@click.option(
    "--rtol",
    type=float,
    default=1e-4,
    help="Relative tolerance for comparison (default: 1e-4)",
)
@click.option(
    "--atol",
    type=float,
    default=1e-5,
    help="Absolute tolerance for comparison (default: 1e-5)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for detailed comparison results",
)
def compare(nnc_debug_file, onnx_model, rtol, atol, output):
    """Compare NNC debug output with ONNX Runtime output.

    Example:
        nnc debug compare nnc_debug_output.txt model.onnx
        nnc debug compare nnc_debug_output.txt model.onnx --rtol 1e-3 --output results.json
    """
    from nnc_py.tools.debug_compare import DebugComparator

    console.print(f"[bold blue]Comparing:[/bold blue] {nnc_debug_file} vs {onnx_model}")
    console.print(f"[bold blue]Tolerances:[/bold blue] rtol={rtol}, atol={atol}")
    console.print()

    comparator = DebugComparator(nnc_debug_file, onnx_model, rtol, atol)
    results = comparator.compare()

    # Print report
    exit_code = comparator.print_report(results)

    # Write JSON output if requested
    if output:
        import json

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results written to:[/green] {output}")

    return exit_code


def _format_tensor_shape(tensor):
    """Format tensor shape for display."""
    dims = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return f"[{', '.join(dims)}]"


if __name__ == "__main__":
    main()
