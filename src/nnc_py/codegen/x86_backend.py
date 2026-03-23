"""x86 backend for simulation."""

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np

from nnc_py.codegen.base import BackendBase, CodeGenResult
from nnc_py.codegen.c_emitter import CEmitter
from nnc_py.ir.context import CompileContext
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.types import DataType
from nnc_py.passes.memory_strategy import (
    MemoryAllocationPlan,
    MovePoint,
    ReloadPoint,
    SpillPoint,
    TensorAllocation,
)
from nnc_py.utils.name_manager import NameManager

if TYPE_CHECKING:
    from nnc_py.passes.spill import SpillPlan


class X86Backend(BackendBase):
    """x86 target backend - generates code for simulation."""

    # Map DataType to NNC_DTYPE constant
    DTYPE_MAP = {
        DataType.FLOAT32: "NNC_DTYPE_FLOAT32",
        DataType.FLOAT16: "NNC_DTYPE_FLOAT16",
        DataType.INT32: "NNC_DTYPE_INT32",
        DataType.INT64: "NNC_DTYPE_INT64",
        DataType.INT8: "NNC_DTYPE_INT8",
        DataType.UINT8: "NNC_DTYPE_UINT8",
        DataType.BOOL: "NNC_DTYPE_BOOL",
    }

    def __init__(self, debug_mode: bool = False):
        """Initialize the x86 backend.

        Args:
            debug_mode: Whether to enable debug mode with intermediate tensor dumps.
        """
        self.debug_mode = debug_mode

    def generate(self, ctx: CompileContext) -> CodeGenResult:
        """Generate x86 C code."""
        result = CodeGenResult()

        # Assign C symbol names
        self._assign_symbols(ctx)

        # Generate header file
        header = self._generate_header(ctx)
        result.add_file("model.h", header, "header")

        # Generate source file
        source = self._generate_source(ctx)
        result.add_file("model.c", source, "source")

        # Generate tensors definition file
        tensors = self._generate_tensors(ctx)
        result.add_file("tensors.c", tensors, "source")

        # Generate constants binary file and loader code
        if ctx.graph.constants:
            constants_binary, constants_metadata = self._generate_constants_binary(ctx)
            result.add_file("constants.bin", constants_binary, "binary")

            # Generate constants loader code
            constants_loader = self._generate_constants_loader(ctx, constants_metadata)
            result.add_file("constants_loader.c", constants_loader, "source")

        # Generate Makefile
        makefile = self._generate_makefile(ctx)
        result.add_file("Makefile", makefile, "build")

        # Generate test runner
        test_runner = self._generate_test_runner(ctx)
        result.add_file("test_runner.c", test_runner, "source")

        return result

    def _get_public_entry_point(self, ctx: CompileContext) -> str:
        """Get the public entry-point symbol name for generated code."""
        entry_point = ctx.metadata.get("entry_point", "nnc_run")
        if not isinstance(entry_point, str) or not entry_point:
            return "nnc_run"
        return entry_point

    def _generate_source(self, ctx: CompileContext) -> str:
        """Generate main source file with spill/reload support."""
        from nnc_py.codegen.c_emitter import CEmitter
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        from nnc_py.passes.spill import get_spill_plan

        # Check for new MemoryAllocationPlan
        alloc_plan = get_memory_allocation_plan(ctx)

        # Check for legacy spill plan
        spill_plan = get_spill_plan(ctx)

        # Determine which plan to use
        if alloc_plan is not None and (alloc_plan.has_spill or bool(alloc_plan.move_points)):
            # Use new MemoryAllocationPlan with reload code generation
            source = self._generate_source_with_unified_spill(ctx, alloc_plan)
        elif spill_plan is not None and spill_plan.has_overflow:
            # Use legacy spill plan
            source = self._generate_source_with_spill(ctx, spill_plan)
        else:
            # No overflow, use standard emitter
            emitter = CEmitter()
            source = emitter.emit(ctx)

        # Add debug macros if in debug mode
        source = self._add_debug_macros(source)

        # Inject debug code if enabled
        source = self._inject_debug_into_nnc_run(source, ctx)
        return self._append_entry_point_alias(source, ctx)

    def _append_entry_point_alias(self, source_code: str, ctx: CompileContext) -> str:
        """Append a public wrapper when the requested entry point is not nnc_run."""
        entry_point = self._get_public_entry_point(ctx)
        if entry_point == "nnc_run":
            return source_code

        return (
            source_code
            + "\n"
            + f"/* Public entry point alias */\nvoid {entry_point}(void) {{\n    nnc_run();\n}}\n"
        )

    def _add_debug_macros(self, source_code: str) -> str:
        """Add debug macro definitions to source code.

        Args:
            source_code: Generated C source code.

        Returns:
            Modified source code with debug macros if debug mode is enabled.
        """
        if not self.debug_mode:
            return source_code

        # Define debug macros - use debug_file for output (declared as extern in test_runner.c)
        debug_macros = """
/* Debug mode: macros for intermediate tensor output */
/* Note: debug_file is defined in test_runner.c as a FILE* */
extern FILE* debug_file;

#ifndef DEBUG_PRINTF
#define DEBUG_PRINTF(fmt, ...) do { \\
    if (debug_file) { \\
        fprintf(debug_file, fmt, ##__VA_ARGS__); \\
    } else { \\
        printf(fmt, ##__VA_ARGS__); \\
    } \\
} while(0)
#endif

#define DEBUG_PRINT_TENSOR_START(name, idx) DEBUG_PRINTF("DEBUG_TENSOR_START %s %d\\n", name, idx)
#define DEBUG_PRINT_SHAPE(ndim) DEBUG_PRINTF("SHAPE %d\\n", ndim)
#define DEBUG_PRINT_DIM(i, val) DEBUG_PRINTF("DIM %d %d\\n", i, val)
#define DEBUG_PRINT_DATA_START() DEBUG_PRINTF("DATA_START\\n")
#define DEBUG_PRINT_VALUE(val) DEBUG_PRINTF("%f\\n", val)
#define DEBUG_PRINT_DATA_END() DEBUG_PRINTF("DATA_END\\n")
#define DEBUG_PRINT_TENSOR_END(name) DEBUG_PRINTF("DEBUG_TENSOR_END %s\\n\\n", name)

"""

        # Insert after the includes
        lines = source_code.split("\n")
        output = []
        inserted = False

        for line in lines:
            output.append(line)
            # Insert after all #include and #define statements
            if not inserted and (line.startswith("#include") or line.startswith("/*")):
                # Check if next line is not an include
                continue
            elif not inserted and (line.startswith("#include") or line.startswith("/*") or line.strip() == ""):
                continue
            elif not inserted:
                # This is the first non-include line, insert macros here
                output.append(debug_macros)
                inserted = True

        return "\n".join(output)

    def _generate_source_with_spill(self, ctx: CompileContext, spill_plan: "SpillPlan") -> str:
        """Generate source file with spill/reload wrapper functions."""
        from nnc_py.codegen.c_emitter import CEmitter

        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "model.h"',
            '#include "nnc_ops.h"',
            "",
            "/* External memory pools (defined in tensors.c) */",
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
            "/* Forward declarations for spill/reload functions */",
        ]

        # Generate forward declarations for node functions
        nodes = ctx.graph.topological_sort()
        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            lines.append(f"static void {func_name}_body(void);")

        lines.append("")
        lines.append("/* Spill/Reload wrapper functions */")

        # Generate wrapper functions with spill/reload
        spill_plan_dict = {
            (p.tensor_name, p.after_node): p
            for p in spill_plan.spill_points
        }
        reload_plan_dict = {
            (p.tensor_name, p.before_node): p
            for p in spill_plan.reload_points
        }

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)

            # Collect spills after this node
            spills_after = [
                p for p in spill_plan.spill_points if p.after_node == node.name
            ]

            # Collect reloads before this node
            reloads_before = [
                p for p in spill_plan.reload_points if p.before_node == node.name
            ]

            lines.append(f"/* {func_name}: {node.op_type.value} */")
            lines.append(f"static void {func_name}(void) {{")

            # Reload before node
            for reload in reloads_before:
                lines.extend([
                    f"    /* Reload {reload.tensor_name} from slow memory */",
                    f"    memcpy(",
                    f"        _nnc_fast_pool + {reload.to_fast_offset},",
                    f"        _nnc_slow_pool + {reload.from_slow_offset},",
                    f"        {reload.size}",
                    f"    );",
                    "",
                ])

            # Call the body
            lines.append(f"    {func_name}_body();")

            # Spill after node
            for spill in spills_after:
                lines.extend([
                    "",
                    f"    /* Spill {spill.tensor_name} to slow memory */",
                    f"    memcpy(",
                    f"        _nnc_slow_pool + {spill.to_slow_offset},",
                    f"        _nnc_fast_pool + {spill.from_fast_offset},",
                    f"        {spill.size}",
                    f"    );",
                ])

            lines.append("}")
            lines.append("")

        lines.append("/* Node body functions */")

        # Generate body functions (using standard emitter)
        emitter = CEmitter()
        body_code = emitter.emit(ctx)

        # We need to modify the body_code to rename functions to _body
        # and remove the main entry point
        lines.append(self._process_body_code(body_code, ctx))

        # Generate main entry point
        lines.append("")
        lines.append("/* Main inference entry point */")
        lines.append("void nnc_run(void) {")

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            lines.append(f"    {func_name}();")

        lines.append("}")

        return "\n".join(lines)

    def _process_body_code(self, body_code: str, ctx: CompileContext) -> str:
        """Process the body code from CEmitter."""
        lines = body_code.split("\n")
        output = []
        skip_main = False

        for line in lines:
            # Skip the includes (we already added them)
            if line.startswith("#include") or line.startswith("/* Auto-generated"):
                continue

            # Skip the main nnc_run function (we'll generate our own)
            if "void nnc_run(void)" in line:
                skip_main = True
                continue

            if skip_main:
                if line.startswith("}"):
                    skip_main = False
                continue

            # Rename functions to _body
            for node_name, func_name in ctx.node_symbols.items():
                if f"void {func_name}(void)" in line:
                    line = line.replace(f"void {func_name}(void)", f"static void {func_name}_body(void)")

            output.append(line)

        return "\n".join(output)

    def _generate_source_with_unified_spill(self, ctx: CompileContext, plan: "MemoryAllocationPlan") -> str:
        """Generate source file with spill/reload wrapper functions using MemoryAllocationPlan.

        The key requirement: ALL operators must execute with inputs/outputs in fast memory.
        When tensors are spilled to slow memory, they must be:
        1. Reloaded to fast memory before operator execution
        2. Used via temp Tensor structs pointing to fast memory
        3. Spilled back to slow memory after operator execution
        """
        from nnc_py.codegen.c_emitter import CEmitter

        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            "",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "model.h"',
            '#include "nnc_ops.h"',
            "",
            "/* External memory pools (defined in tensors.c) */",
            "extern uint8_t _nnc_fast_pool[];",
            "extern uint8_t _nnc_slow_pool[];",
            "",
        ]

        def get_output_spill_points(node_idx: int) -> dict[str, SpillPoint]:
            return {
                point.tensor_name: point
                for point in plan.get_spill_points_after(node_idx)
            }

        def get_move_points_at(node_idx: int) -> list[MovePoint]:
            return plan.get_move_points_at(node_idx)

        def get_internal_storage_expr(tensor_name: str) -> str | None:
            alloc = plan.tensor_allocations.get(tensor_name)
            if alloc is None:
                return None
            pool_name = "_nnc_slow_pool" if alloc.is_spilled else "_nnc_fast_pool"
            return f"{pool_name} + {alloc.offset}"

        def get_saved_input_binding_name(tensor_name: str) -> str:
            symbol = ctx.tensor_symbols.get(tensor_name, tensor_name)
            return f"_nnc_bound_input_{symbol}"

        input_first_use: dict[str, int] = {}
        liveness_map = ctx.metadata.get("tensor_liveness", {})
        for tensor_name in ctx.graph.inputs:
            liveness = liveness_map.get(tensor_name)
            if liveness is None or not liveness.use_positions:
                continue
            input_first_use[tensor_name] = liveness.use_positions[0]

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            lines.append(f"static uint8_t* {get_saved_input_binding_name(tensor_name)} = NULL;")
        if any(tensor_name in plan.tensor_allocations for tensor_name in ctx.graph.inputs):
            lines.append("")

        # Calculate reload buffer requirements
        # Need buffers for:
        # 1. Inputs to reload
        # 2. Outputs that are spilled (need temp buffers in fast memory)
        max_reload_slots = plan.get_max_reload_slots()

        # Count max total buffers needed at any node (inputs + spilled outputs)
        nodes = ctx.graph.topological_sort()
        max_total_slots = 0
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            # Count inputs that need reload
            reloads_before = plan.get_reload_points_before(node_idx)
            input_slots = len(reloads_before)

            # Count outputs that spill after this node, even if they are later reloaded
            # and do not remain slow-backed in the final tensor allocation map.
            spill_outputs_after = get_output_spill_points(node_idx)
            output_slots = len(spill_outputs_after)

            max_total_slots = max(max_total_slots, input_slots + output_slots)

        # Use the larger of the two calculations
        max_reload_slots = max(max_reload_slots, max_total_slots)

        # Get max tensor size for reload buffer sizing
        max_tensor_size = 0
        transferred_tensors = {
            point.tensor_name for point in plan.spill_points
        } | {
            point.tensor_name for point in plan.reload_points
        } | plan.spilled_tensors
        for tensor_name in transferred_tensors:
            alloc = plan.tensor_allocations.get(tensor_name)
            if alloc is not None:
                max_tensor_size = max(max_tensor_size, alloc.size)
        for point in plan.spill_points:
            max_tensor_size = max(max_tensor_size, point.size)
        for point in plan.reload_points:
            max_tensor_size = max(max_tensor_size, point.size)

        if not plan.has_spill and not plan.move_points:
            emitter = CEmitter()
            return emitter.emit(ctx)

        # Generate reload buffers in fast memory
        if max_reload_slots > 0 and max_tensor_size > 0:
            lines.append("/* Reload buffers in fast memory for spilled tensors */")
            lines.append(f"/* Max reload slots: {max_reload_slots}, Slot size: {max_tensor_size} */")
            for i in range(max_reload_slots):
                lines.append(f"static uint8_t _nnc_reload_buffer_{i}[{max_tensor_size}];")
            lines.append("")

        # Generate forward declarations for node functions (only for those without spill/reload)
        nodes = ctx.graph.topological_sort()
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            spill_outputs_after = get_output_spill_points(node_idx)
            move_points_at = sorted(
                get_move_points_at(node_idx),
                key=lambda move: move.from_offset,
            )

            # Check if this node needs spill/reload
            has_spill = len(spill_outputs_after) > 0
            has_move = len(move_points_at) > 0
            has_reload = len(plan.get_reload_points_before(node_idx)) > 0
            has_spilled_output = any(
                (
                    o in spill_outputs_after
                    or (
                        plan.tensor_allocations.get(o) is not None
                        and plan.tensor_allocations.get(o).is_spilled  # type: ignore[union-attr]
                    )
                )
                for o in node.outputs
            )

            # Only declare _body if no spill/reload needed
            if not has_spill and not has_move and not has_reload and not has_spilled_output:
                lines.append(f"static void {func_name}_body(void);")

        lines.append("")
        lines.append("/* Spill/Reload wrapper functions */")

        # Generate wrapper functions with spill/reload
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)

            # Collect spills after this node
            spill_points_after = get_output_spill_points(node_idx)
            move_points_at = sorted(
                get_move_points_at(node_idx),
                key=lambda move: move.from_offset,
            )
            move_points_by_tensor = {
                move.tensor_name: move for move in move_points_at
            }

            # Collect reloads before this node
            reloads_before = plan.get_reload_points_before(node_idx)

            # Determine which inputs need reload (have reload points before this node)
            inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]] = {
                rp.tensor_name: rp for rp in reloads_before
            }

            # Also check if any input tensors are spilled (allocated in slow memory)
            # This handles model inputs that happen to be in slow memory
            additional_inputs: list[tuple[str, "TensorAllocation"]] = []
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    continue  # Already has a reload point
                alloc = plan.tensor_allocations.get(input_name)
                if alloc is not None and alloc.is_spilled:
                    # Input is spilled but no reload point - create a synthetic one
                    # Look up the correct slow offset from existing reload points
                    existing_reload = next((rp for rp in plan.reload_points if rp.tensor_name == input_name), None)
                    slow_offset = existing_reload.from_slow_offset if existing_reload else 0
                    additional_inputs.append((input_name, alloc))
                    new_reload_point = ReloadPoint(
                        tensor_name=input_name,
                        before_node=node.name,
                        before_node_idx=node_idx,
                        from_slow_offset=slow_offset,
                        to_buffer_id=alloc.buffer_id,
                        to_fast_offset=0,
                        size=alloc.size,
                        reload_slot_id=-1,  # Will assign below
                    )
                    inputs_need_reload[input_name] = new_reload_point

            # Assign slot IDs to synthetic reload points
            current_slot = max(
                [rp.reload_slot_id for rp in inputs_need_reload.values() if isinstance(rp, ReloadPoint)],
                default=-1,
            ) + 1
            for input_name, _ in additional_inputs:
                rp = inputs_need_reload[input_name]
                if isinstance(rp, ReloadPoint):
                    rp.reload_slot_id = current_slot
                current_slot += 1

            # Determine which outputs need temp tensors
            # An output needs a temp tensor if:
            # 1. It's spilled (allocated in slow memory), OR
            # 2. There's a spill point after this node for it
            outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]] = {}
            for output_name in node.outputs:
                spill_point = spill_points_after.get(output_name)
                if spill_point is not None:
                    outputs_need_temp[output_name] = spill_point
                    continue

                alloc = plan.tensor_allocations.get(output_name)
                if alloc is not None and alloc.is_spilled:
                    # Output lives in slow memory - need temp tensor
                    # Find the spill point for this output
                    # Create a synthetic spill point for temp allocation.
                    existing_spill = next((sp for sp in plan.spill_points if sp.tensor_name == output_name), None)
                    slow_offset = existing_spill.to_slow_offset if existing_spill else 0
                    outputs_need_temp[output_name] = SpillPoint(
                        tensor_name=output_name,
                        after_node=node.name,
                        after_node_idx=node_idx,
                        from_buffer_id=alloc.buffer_id,
                        from_fast_offset=0,
                        to_slow_offset=slow_offset,
                        size=alloc.size,
                    )

            lines.append(f"/* {func_name}: {node.op_type.value} */")
            lines.append(f"static void {func_name}(void) {{")

            moved_first_use_inputs = [
                input_name
                for input_name in node.inputs
                if input_first_use.get(input_name) == node_idx
                and input_name in move_points_by_tensor
            ]
            for input_name in moved_first_use_inputs:
                alloc = plan.tensor_allocations.get(input_name)
                move = move_points_by_tensor[input_name]
                var_name = ctx.tensor_symbols.get(input_name, input_name)
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                if alloc is None:
                    continue
                lines.extend([
                    f"    /* Stage graph input {input_name} into pre-move fast memory */",
                    f"    memcpy({from_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                    f"    {var_name}.data = {from_expr};",
                    "",
                ])

            for move in move_points_at:
                var_name = ctx.tensor_symbols.get(move.tensor_name, move.tensor_name)
                to_expr = f"_nnc_fast_pool + {move.to_offset}"
                from_expr = f"_nnc_fast_pool + {move.from_offset}"
                lines.extend([
                    f"    /* Move {move.tensor_name} within fast memory */",
                    f"    memmove({to_expr}, {from_expr}, {move.size});",
                    f"    {var_name}.data = {to_expr};",
                    "",
                ])

            for input_name in node.inputs:
                if input_first_use.get(input_name) != node_idx:
                    continue
                if input_name in move_points_by_tensor:
                    continue

                alloc = plan.tensor_allocations.get(input_name)
                target_expr = get_internal_storage_expr(input_name)
                if alloc is None or target_expr is None:
                    continue

                var_name = ctx.tensor_symbols.get(input_name, input_name)
                lines.extend([
                    f"    /* Stage graph input {input_name} into planned memory */",
                    f"    memcpy({target_expr}, {get_saved_input_binding_name(input_name)}, {alloc.size});",
                    f"    {var_name}.data = {target_expr};",
                    "",
                ])

            # Declare temp tensors for spilled inputs
            temp_tensor_decls: list[tuple[str, int, str, int, str]] = []
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    rp = inputs_need_reload[input_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(input_name, input_name)}"
                    slot_id = rp.reload_slot_id if isinstance(rp, ReloadPoint) else -1
                    temp_tensor_decls.append((temp_name, slot_id, input_name, rp.size, 'input'))

            # Declare temp tensors for spilled outputs
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    slot_id = output_slot_idx
                    temp_tensor_decls.append((temp_name, slot_id, output_name, sp_val.size, 'output'))
                    output_slot_idx += 1

            # Generate temp tensor declarations
            if temp_tensor_decls:
                lines.append("    /* Temp tensors pointing to fast memory */")
                for temp_name, slot_id, orig_name, size, _ in temp_tensor_decls:
                    lines.append(f"    static Tensor {temp_name};")

            # Reload spilled inputs to fast memory
            for input_name in node.inputs:
                if input_name in inputs_need_reload:
                    rp = inputs_need_reload[input_name]
                    if not isinstance(rp, ReloadPoint):
                        continue
                    temp_name = f"temp_{ctx.tensor_symbols.get(input_name, input_name)}"
                    var_name = ctx.tensor_symbols.get(input_name, input_name)
                    lines.extend([
                        f"    /* Reload {input_name} from slow to fast memory */",
                        f"    memcpy(_nnc_reload_buffer_{rp.reload_slot_id},",
                        f"           _nnc_slow_pool + {rp.from_slow_offset}, {rp.size});",
                        f"    {temp_name}.data = _nnc_reload_buffer_{rp.reload_slot_id};",
                        f"    {temp_name}.dtype = {var_name}.dtype;",
                        f"    {temp_name}.shape = {var_name}.shape;",
                        f"    {temp_name}.ndim = {var_name}.ndim;",
                        f"    {temp_name}.nbytes = {rp.size};",
                        "",
                    ])

            # Setup temp outputs
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    var_name = ctx.tensor_symbols.get(output_name, output_name)
                    # Get the slot_id from temp_tensor_decls
                    slot_id = next((s for tn, s, _, _, _ in temp_tensor_decls if tn == temp_name), output_slot_idx)
                    # Both ReloadPoint and SpillPoint have 'size' attribute
                    sp_size = sp_val.size
                    lines.extend([
                        f"    /* Setup temp output for {output_name} */",
                        f"    {temp_name}.data = _nnc_reload_buffer_{slot_id};",
                        f"    {temp_name}.dtype = {var_name}.dtype;",
                        f"    {temp_name}.shape = {var_name}.shape;",
                        f"    {temp_name}.ndim = {var_name}.ndim;",
                        f"    {temp_name}.nbytes = {sp_size};",
                        "",
                    ])

            # Generate operator call with potentially swapped arguments
            lines.append("    /* Execute operator in fast memory */")
            op_call = self._generate_operator_call_with_temps(
                ctx, node, inputs_need_reload, outputs_need_temp
            )
            lines.append(f"    {op_call}")

            # Spill outputs back to slow memory
            output_slot_idx = len(inputs_need_reload)
            for output_name in node.outputs:
                if output_name in outputs_need_temp:
                    sp_val = outputs_need_temp[output_name]
                    temp_name = f"temp_{ctx.tensor_symbols.get(output_name, output_name)}"
                    # Get the slot_id from temp_tensor_decls
                    slot_id = next((s for tn, s, _, _, _ in temp_tensor_decls if tn == temp_name), output_slot_idx)
                    # Both ReloadPoint and SpillPoint have these attributes (or should)
                    # SpillPoint has to_slow_offset, ReloadPoint doesn't (but we shouldn't spill to ReloadPoint)
                    if isinstance(sp_val, SpillPoint):
                        lines.extend([
                            "",
                            f"    /* Spill {output_name} from fast to slow memory */",
                            f"    memcpy(_nnc_slow_pool + {sp_val.to_slow_offset},",
                            f"           _nnc_reload_buffer_{slot_id}, {sp_val.size});",
                        ])

            lines.append("}")
            lines.append("")

        # Generate body functions for nodes without spill/reload
        lines.append("/* Node body functions (no spill/reload needed) */")

        # Generate operator calls for all nodes (for reference)
        # We generate simplified versions that don't need spill/reload
        for node_idx, node in enumerate(nodes):
            if node.op_type == OpType.CONSTANT:
                continue

            func_name = ctx.node_symbols.get(node.name, node.name)
            spill_outputs_after = get_output_spill_points(node_idx)
            move_points_at = get_move_points_at(node_idx)

            # Check if this node needs spill/reload
            # A node needs spill/reload wrapper if:
            # 1. It has reload points before it (inputs to reload), OR
            # 2. Any of its outputs are spilled (live in slow memory)
            has_spill = len(spill_outputs_after) > 0
            has_move = len(move_points_at) > 0
            has_reload = len(plan.get_reload_points_before(node_idx)) > 0
            has_spilled_output = any(
                (
                    o in spill_outputs_after
                    or (
                        plan.tensor_allocations.get(o) is not None
                        and plan.tensor_allocations.get(o).is_spilled  # type: ignore[union-attr]
                    )
                )
                for o in node.outputs
            )

            if not has_spill and not has_move and not has_reload and not has_spilled_output:
                # No spill/reload needed, generate standard operator call
                lines.append(f"/* {func_name}: {node.op_type.value} */")
                lines.append(f"static void {func_name}_body(void) {{")
                op_call = self._generate_operator_call(ctx, node, use_temps=False)
                lines.append(f"    {op_call}")
                lines.append("}")
                lines.append("")
            else:
                # Has spill/reload, don't generate _body function
                # The wrapper handles everything
                pass

        # Generate main entry point
        lines.append("")
        lines.append("/* Main inference entry point */")
        lines.append("void nnc_run(void) {")

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"    {get_saved_input_binding_name(tensor_name)} = {var_name}.data;")

        for node in nodes:
            if node.op_type == OpType.CONSTANT:
                continue
            func_name = ctx.node_symbols.get(node.name, node.name)
            lines.append(f"    {func_name}();")

        for tensor_name in ctx.graph.inputs:
            if tensor_name not in plan.tensor_allocations:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"    {var_name}.data = {get_saved_input_binding_name(tensor_name)};")

        lines.append("}")

        return "\n".join(lines)

    def _generate_operator_call_with_temps(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate operator call using temp tensors for spilled inputs/outputs.

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint (for spilled outputs)

        Returns:
            C code line for the operator call
        """
        # Special handling for operators with attribute-as-input (after onnxsim optimization)
        # These operators need special handling because some attributes became inputs
        if node.op_type == OpType.CONCAT:
            return self._generate_concat_call(ctx, node, inputs_need_reload, outputs_need_temp)
        if node.op_type == OpType.SPLIT:
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)

        # Operators that have extra "attribute" inputs after onnxsim
        if node.op_type in (OpType.REDUCE_SUM, OpType.REDUCE_MEAN, OpType.UNSQUEEZE, OpType.TRANSPOSE, OpType.TILE, OpType.RESHAPE, OpType.POW, OpType.CLIP):
            return self._generate_attr_as_input_op_call(ctx, node, inputs_need_reload, outputs_need_temp)

        op_name = f"nnc_{node.op_type.value.lower()}"
        args = []

        # Input tensors
        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            if input_name in inputs_need_reload:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                args.append(f"&{temp_name}")
            else:
                # Use original tensor
                args.append(f"&{var_name}")

        # Output tensors
        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            if output_name in outputs_need_temp:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                args.append(f"&{temp_name}")
            else:
                # Use original tensor
                args.append(f"&{var_name}")

        # Add operation-specific attributes
        if node.op_type == OpType.CONV2D:
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            strides = node.attrs.get("strides", [1, 1])
            pads = node.attrs.get("pads", [0, 0])
            args.extend([str(kernel_shape[0]), str(kernel_shape[1])])
            args.extend([str(strides[0]), str(strides[1])])
            if len(pads) == 4:
                args.append(str(pads[0]))
                args.append(str(pads[1]))
            elif len(pads) == 2:
                args.extend([str(pads[0]), str(pads[1])])
            else:
                args.extend(["0", "0"])
        elif node.op_type == OpType.LAYER_NORM:
            axis = node.attrs.get("axis", -1)
            epsilon = node.attrs.get("epsilon", 1e-5)
            args.append(f"{axis}")
            args.append(f"{epsilon}f")
        elif node.op_type == OpType.CAST:
            # Cast: input, output, to_dtype
            to_dtype = node.attrs.get("to")
            if to_dtype and isinstance(to_dtype, DataType):
                dtype_const = self.DTYPE_MAP.get(to_dtype, "NNC_DTYPE_FLOAT32")
                args.append(dtype_const)
            else:
                args.append("NNC_DTYPE_FLOAT32")
        elif node.op_type == OpType.SPLIT:
            # Split has signature: nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis)
            # We need to completely rewrite the call
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.REDUCE_MEAN:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.REDUCE_SUM:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.TRANSPOSE:
            # Transpose: input, output, perm array, ndim
            # perm can be in attributes or as an input (after onnxsim optimization)
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm might be the second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []  # Will use default
            ndim = len(perm) if perm else 0
            # Generate static perm array
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            # Need to return multi-line string
            output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
            return f"{perm_decl}\n    nnc_transpose(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){perm_name}, {ndim});"
        elif node.op_type == OpType.UNSQUEEZE:
            # Unsqueeze: input, output, axis
            # In ONNX opset 13+, axes can be an input instead of attribute
            axes = node.attrs.get("axes", None)
            if axes is None and len(node.inputs) > 1:
                # axes might be the second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axes = axes_val
                    elif hasattr(axes_val, 'tolist'):
                        axes = axes_val.tolist()
                    else:
                        axes = [axes_val]
            if axes is None:
                axes = [-1]  # Default to last axis
            axis_val = axes[0] if isinstance(axes, list) else axes
            args.append(str(axis_val))
        elif node.op_type == OpType.TILE:
            # Tile: input, output, repeats array, ndim
            # repeats is the second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    # Generate static repeats array
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    # Need to return multi-line string
                    output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
                    return f"{repeats_decl}\n    nnc_tile(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){repeats_name}, {ndim});"
            # Fallback: pass NULL for repeats
            args.append("NULL")
            args.append("0")
        elif node.op_type == OpType.GEMM:
            # Gemm has optional bias (3rd input), need to add attributes
            # nnc_gemm(Tensor* a, Tensor* b, Tensor* c, Tensor* output,
            #          float alpha, float beta, int trans_a, int trans_b)
            # Check if bias (C) is present
            if len(node.inputs) < 3:
                # No bias tensor, insert NULL
                args.insert(2, "NULL")
            # Add attributes
            alpha = node.attrs.get("alpha", 1.0)
            beta = node.attrs.get("beta", 1.0)
            trans_a = node.attrs.get("transA", 0)
            trans_b = node.attrs.get("transB", 0)
            args.append(f"{alpha}f")
            args.append(f"{beta}f")
            args.append(str(trans_a))
            args.append(str(trans_b))
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.LSTM:
            # LSTM has special handling
            # X, W, R, B inputs
            # Y, Y_h, Y_c outputs
            # direction, hidden_size attributes
            direction = node.attrs.get("direction", "forward")
            hidden_size = node.attrs.get("hidden_size", 0)
            direction_map = {"forward": 0, "reverse": 1, "bidirectional": 2}
            direction_val = direction_map.get(direction, 0)
            args.append(str(direction_val))
            args.append(str(hidden_size))

        return f"{op_name}({', '.join(args)});"

    def _generate_operator_call(self, ctx: CompileContext, node: Node, use_temps: bool = False) -> str:
        """Generate operator call (for non-spill nodes).

        Args:
            ctx: Compilation context
            node: Node to generate call for
            use_temps: Whether to use temp tensors (not used here, for compatibility)

        Returns:
            C code line for the operator call
        """
        # Special handling for operators with attribute-as-input (after onnxsim optimization)
        if node.op_type == OpType.CONCAT:
            return self._generate_concat_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})
        if node.op_type == OpType.SPLIT:
            return self._generate_split_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})
        if node.op_type in (OpType.REDUCE_SUM, OpType.REDUCE_MEAN, OpType.UNSQUEEZE, OpType.TRANSPOSE, OpType.TILE, OpType.RESHAPE, OpType.POW, OpType.CLIP):
            return self._generate_attr_as_input_op_call(ctx, node, inputs_need_reload={}, outputs_need_temp={})

        op_name = f"nnc_{node.op_type.value.lower()}"
        args = []

        # Input tensors
        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            args.append(f"&{var_name}")

        # Output tensors
        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            args.append(f"&{var_name}")

        # Add operation-specific attributes
        if node.op_type == OpType.CONV2D:
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            strides = node.attrs.get("strides", [1, 1])
            pads = node.attrs.get("pads", [0, 0])
            args.extend([str(kernel_shape[0]), str(kernel_shape[1])])
            args.extend([str(strides[0]), str(strides[1])])
            if len(pads) == 4:
                args.append(str(pads[0]))
                args.append(str(pads[1]))
            elif len(pads) == 2:
                args.extend([str(pads[0]), str(pads[1])])
            else:
                args.extend(["0", "0"])
        elif node.op_type == OpType.LAYER_NORM:
            axis = node.attrs.get("axis", -1)
            epsilon = node.attrs.get("epsilon", 1e-5)
            args.append(f"{axis}")
            args.append(f"{epsilon}f")
        elif node.op_type == OpType.CAST:
            # Cast: input, output, to_dtype
            to_dtype = node.attrs.get("to")
            if to_dtype and isinstance(to_dtype, DataType):
                dtype_const = self.DTYPE_MAP.get(to_dtype, "NNC_DTYPE_FLOAT32")
                args.append(dtype_const)
            else:
                args.append("NNC_DTYPE_FLOAT32")
        elif node.op_type == OpType.SPLIT:
            # Split has signature: nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis)
            # We need to completely rewrite the call
            return self._generate_split_call(ctx, node, inputs_need_reload, outputs_need_temp)
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.REDUCE_MEAN:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.REDUCE_SUM:
            # ONNX uses "axes" (plural) as a list, or "axis" (singular) as int
            axes = node.attrs.get("axes", None)
            axis = node.attrs.get("axis", None)
            keepdims = node.attrs.get("keepdims", 1)
            # Determine axis value (handle both "axes" list and "axis" int)
            if axes is not None and isinstance(axes, list) and len(axes) > 0:
                axis_val = axes[0]
            elif axis is not None:
                axis_val = axis
            else:
                axis_val = -1  # Default to last axis
            args.append(str(axis_val))
            args.append(str(keepdims))
        elif node.op_type == OpType.TRANSPOSE:
            # Transpose: input, output, perm array, ndim
            # perm can be in attributes or as an input (after onnxsim optimization)
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm might be the second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []  # Will use default
            ndim = len(perm) if perm else 0
            # Generate static perm array
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            # Need to return multi-line string
            output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
            return f"{perm_decl}\n    nnc_transpose(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){perm_name}, {ndim});"
        elif node.op_type == OpType.UNSQUEEZE:
            # Unsqueeze: input, output, axis
            # In ONNX opset 13+, axes can be an input instead of attribute
            axes = node.attrs.get("axes", None)
            if axes is None and len(node.inputs) > 1:
                # axes might be the second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axes = axes_val
                    elif hasattr(axes_val, 'tolist'):
                        axes = axes_val.tolist()
                    else:
                        axes = [axes_val]
            if axes is None:
                axes = [-1]  # Default to last axis
            axis_val = axes[0] if isinstance(axes, list) else axes
            args.append(str(axis_val))
        elif node.op_type == OpType.TILE:
            # Tile: input, output, repeats array, ndim
            # repeats is the second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    # Generate static repeats array
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    # Need to return multi-line string
                    output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
                    return f"{repeats_decl}\n    nnc_tile(&{ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])}, &{output_var}, (int64_t*){repeats_name}, {ndim});"
            # Fallback: pass NULL for repeats
            args.append("NULL")
            args.append("0")
        elif node.op_type == OpType.GEMM:
            # Gemm has optional bias (3rd input), need to add attributes
            # nnc_gemm(Tensor* a, Tensor* b, Tensor* c, Tensor* output,
            #          float alpha, float beta, int trans_a, int trans_b)
            # Check if bias (C) is present
            if len(node.inputs) < 3:
                # No bias tensor, insert NULL
                args.insert(2, "NULL")
            # Add attributes
            alpha = node.attrs.get("alpha", 1.0)
            beta = node.attrs.get("beta", 1.0)
            trans_a = node.attrs.get("transA", 0)
            trans_b = node.attrs.get("transB", 0)
            args.append(f"{alpha}f")
            args.append(f"{beta}f")
            args.append(str(trans_a))
            args.append(str(trans_b))
        elif node.op_type == OpType.GATHER:
            # Gather: data, indices, output, axis, data_dtype
            axis = node.attrs.get("axis", 0)
            # Determine data type: 1 for int64 (e.g., from Shape operator), 0 for float
            data_tensor = ctx.graph.get_tensor(node.inputs[0])
            data_dtype = 1 if data_tensor and data_tensor.dtype == DataType.INT64 else 0
            args.append(str(axis))
            args.append(str(data_dtype))
        elif node.op_type == OpType.LSTM:
            # LSTM has special handling
            # X, W, R, B inputs
            # Y, Y_h, Y_c outputs
            # direction, hidden_size attributes
            direction = node.attrs.get("direction", "forward")
            hidden_size = node.attrs.get("hidden_size", 0)
            direction_map = {"forward": 0, "reverse": 1, "bidirectional": 2}
            direction_val = direction_map.get(direction, 0)
            args.append(str(direction_val))
            args.append(str(hidden_size))

        return f"{op_name}({', '.join(args)});"

    def _generate_concat_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate concat operation call.

        Concat has signature: void nnc_concat(Tensor** inputs, Tensor* output, int num_inputs, int axis);

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the concat call (may be multi-line)
        """
        num_inputs = len(node.inputs)
        axis = node.attrs.get("axis", 0)
        output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])

        # Create static array of input pointers
        array_name = f"{ctx.node_symbols.get(node.name, node.name)}_inputs"
        input_ptrs = []

        for input_name in node.inputs:
            var_name = ctx.tensor_symbols.get(input_name, input_name)
            if input_name in inputs_need_reload:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                input_ptrs.append(f"&{temp_name}")
            else:
                # Use original tensor
                input_ptrs.append(f"&{var_name}")

        # Generate the array declaration and call
        array_decl = f"static Tensor* {array_name}[{num_inputs}] = {{{', '.join(input_ptrs)}}};"
        output_arg = f"&{output_var}"

        if node.outputs[0] in outputs_need_temp:
            temp_name = f"temp_{output_var}"
            output_arg = f"&{temp_name}"

        call = f"nnc_concat({array_name}, {output_arg}, {num_inputs}, {axis});"

        # Return multi-line string with array decl and call
        return f"{array_decl}\n    {call}"

    def _generate_split_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate split operation call.

        Split has signature: void nnc_split(Tensor* input, Tensor** outputs, int num_outputs, int axis);

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the split call (may be multi-line)
        """
        num_outputs = len(node.outputs)
        axis = node.attrs.get("axis", 0)
        input_var = ctx.tensor_symbols.get(node.inputs[0], node.inputs[0])

        # Create static array of output pointers
        array_name = f"{ctx.node_symbols.get(node.name, node.name)}_outputs"
        output_ptrs = []

        for output_name in node.outputs:
            var_name = ctx.tensor_symbols.get(output_name, output_name)
            if output_name in outputs_need_temp:
                # Use temp tensor
                temp_name = f"temp_{var_name}"
                output_ptrs.append(f"&{temp_name}")
            else:
                # Use original tensor
                output_ptrs.append(f"&{var_name}")

        # Generate the array declaration and call
        array_decl = f"static Tensor* {array_name}[{num_outputs}] = {{{', '.join(output_ptrs)}}};"
        input_arg = f"&{input_var}"

        if node.inputs[0] in inputs_need_reload:
            temp_name = f"temp_{input_var}"
            input_arg = f"&{temp_name}"

        call = f"nnc_split({input_arg}, {array_name}, {num_outputs}, {axis});"

        # Return multi-line string with array decl and call
        return f"{array_decl}\n    {call}"

    def _generate_attr_as_input_op_call(
        self,
        ctx: CompileContext,
        node: Node,
        inputs_need_reload: dict[str, Union["ReloadPoint", "SpillPoint"]],
        outputs_need_temp: dict[str, Union["ReloadPoint", "SpillPoint"]],
    ) -> str:
        """Generate operator call for ops where attributes became inputs (after onnxsim).

        This handles REDUCE_SUM, REDUCE_MEAN, UNSQUEEZE, TRANSPOSE, TILE
        where ONNX opset 13+ converted some attributes to inputs.

        Args:
            ctx: Compilation context
            node: Node to generate call for
            inputs_need_reload: Map of input_name -> ReloadPoint
            outputs_need_temp: Map of output_name -> SpillPoint

        Returns:
            C code for the operator call (may be multi-line)
        """
        op_type = node.op_type

        # Get input tensor (first input is always the data input)
        input_name = node.inputs[0]
        input_var = ctx.tensor_symbols.get(input_name, input_name)
        if input_name in inputs_need_reload:
            input_arg = f"&temp_{input_var}"
        else:
            input_arg = f"&{input_var}"

        # Get output tensor
        output_var = ctx.tensor_symbols.get(node.outputs[0], node.outputs[0])
        if node.outputs[0] in outputs_need_temp:
            output_arg = f"&temp_{output_var}"
        else:
            output_arg = f"&{output_var}"

        if op_type == OpType.REDUCE_SUM:
            # nnc_reducesum(Tensor* input, Tensor* output, int axis, int keepdims)
            # axis may be from attributes or from second input
            axis_val = node.attrs.get("axes", node.attrs.get("axis", None))
            if axis_val is None and len(node.inputs) > 1:
                # axis is from second input (constant)
                axis_input = node.inputs[1]
                if axis_input in ctx.graph.constants:
                    axis_val = ctx.graph.constants[axis_input]
                    if isinstance(axis_val, list):
                        axis_val = axis_val[0] if len(axis_val) > 0 else -1
                    elif hasattr(axis_val, 'item'):
                        axis_val = axis_val.item()
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            keepdims = node.attrs.get("keepdims", 1)
            return f"nnc_reducesum({input_arg}, {output_arg}, {axis_val}, {keepdims});"

        elif op_type == OpType.REDUCE_MEAN:
            # nnc_reducemean(Tensor* input, Tensor* output, int axis, int keepdims)
            axis_val = node.attrs.get("axes", node.attrs.get("axis", None))
            if axis_val is None and len(node.inputs) > 1:
                # axis is from second input (constant)
                axis_input = node.inputs[1]
                if axis_input in ctx.graph.constants:
                    axis_val = ctx.graph.constants[axis_input]
                    if isinstance(axis_val, list):
                        axis_val = axis_val[0] if len(axis_val) > 0 else -1
                    elif hasattr(axis_val, 'item'):
                        axis_val = axis_val.item()
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            keepdims = node.attrs.get("keepdims", 1)
            return f"nnc_reducemean({input_arg}, {output_arg}, {axis_val}, {keepdims});"

        elif op_type == OpType.UNSQUEEZE:
            # nnc_unsqueeze(Tensor* input, Tensor* output, int axis)
            # axes may be from attributes or from second input
            axis_val = node.attrs.get("axes", None)
            if axis_val is None and len(node.inputs) > 1:
                # axes is from second input (constant)
                axes_input = node.inputs[1]
                if axes_input in ctx.graph.constants:
                    axes_val = ctx.graph.constants[axes_input]
                    if isinstance(axes_val, list):
                        axis_val = axes_val[0] if len(axes_val) > 0 else -1
                    elif hasattr(axes_val, 'item'):
                        axis_val = axes_val.item()
                    else:
                        axis_val = axes_val
            if axis_val is None:
                axis_val = -1
            if isinstance(axis_val, list):
                axis_val = axis_val[0] if len(axis_val) > 0 else -1
            return f"nnc_unsqueeze({input_arg}, {output_arg}, {axis_val});"

        elif op_type == OpType.RESHAPE:
            # nnc_reshape(Tensor* input, Tensor* output, int64_t* shape, int ndim)
            # shape may be from attributes or from second input
            shape = node.attrs.get("shape", None)
            if shape is None and len(node.inputs) > 1:
                # shape is from second input (constant)
                shape_input = node.inputs[1]
                if shape_input in ctx.graph.constants:
                    shape = ctx.graph.constants[shape_input]
                    if hasattr(shape, 'tolist'):
                        shape = shape.tolist()
                    elif not isinstance(shape, list):
                        shape = [shape]
            if shape is None:
                shape = []
            ndim = len(shape)
            shape_name = f"{ctx.node_symbols.get(node.name, node.name)}_shape"
            shape_decl = f"static const int64_t {shape_name}[] = {{{', '.join(map(str, shape))}}};"
            return f"{shape_decl}\n    nnc_reshape({input_arg}, {output_arg}, (int64_t*){shape_name}, {ndim});"

        elif op_type == OpType.TRANSPOSE:
            # nnc_transpose(Tensor* input, Tensor* output, int64_t* perm, int ndim)
            # perm may be from attributes or from second input
            perm = node.attrs.get("perm", None)
            if perm is None and len(node.inputs) > 1:
                # perm is from second input (constant)
                perm_input = node.inputs[1]
                if perm_input in ctx.graph.constants:
                    perm = ctx.graph.constants[perm_input].tolist()
            if perm is None:
                perm = []
            ndim = len(perm)
            perm_name = f"{ctx.node_symbols.get(node.name, node.name)}_perm"
            perm_decl = f"static const int64_t {perm_name}[] = {{{', '.join(map(str, perm))}}};"
            return f"{perm_decl}\n    nnc_transpose({input_arg}, {output_arg}, (int64_t*){perm_name}, {ndim});"

        elif op_type == OpType.TILE:
            # nnc_tile(Tensor* input, Tensor* output, int64_t* repeats, int ndim)
            # repeats is from second input (constant)
            if len(node.inputs) > 1:
                repeats_input = node.inputs[1]
                if repeats_input in ctx.graph.constants:
                    repeats = ctx.graph.constants[repeats_input]
                    if isinstance(repeats, list):
                        repeats_list = repeats
                    elif hasattr(repeats, 'tolist'):
                        repeats_list = repeats.tolist()
                    else:
                        repeats_list = [repeats]
                    ndim = len(repeats_list)
                    repeats_name = f"{ctx.node_symbols.get(node.name, node.name)}_repeats"
                    repeats_decl = f"static const int64_t {repeats_name}[] = {{{', '.join(map(str, repeats_list))}}};"
                    return f"{repeats_decl}\n    nnc_tile({input_arg}, {output_arg}, (int64_t*){repeats_name}, {ndim});"
            # Fallback
            return f"nnc_tile({input_arg}, {output_arg}, NULL, 0);"

        elif op_type == OpType.POW:
            # nnc_pow(Tensor* input, Tensor* output) - computes x^2
            # ONNX Pow has two inputs (base, exponent), but nnc_pow only has input (implicitly x^2)
            # We only use the first input (base) and ignore the exponent
            return f"nnc_pow({input_arg}, {output_arg});"

        elif op_type == OpType.CLIP:
            # nnc_clip(Tensor* input, Tensor* output, float min_val, float max_val)
            # Clip may have min/max as inputs (after onnx opset 11) or as attributes
            min_val = node.attrs.get("min", None)
            max_val = node.attrs.get("max", None)

            # Try to get min from inputs (3rd input in ONNX opset 11+)
            if min_val is None and len(node.inputs) >= 2:
                min_input = node.inputs[1]
                if min_input in ctx.graph.constants:
                    min_val = ctx.graph.constants[min_input]
                    if hasattr(min_val, 'item'):
                        min_val = min_val.item()

            # Try to get max from inputs (4th input in ONNX opset 11+)
            if max_val is None and len(node.inputs) >= 3:
                max_input = node.inputs[2] if len(node.inputs) > 2 else node.inputs[-1]
                if max_input in ctx.graph.constants:
                    max_val = ctx.graph.constants[max_input]
                    if hasattr(max_val, 'item'):
                        max_val = max_val.item()

            if min_val is None:
                min_str = "-1e30f"
            else:
                min_str = f"{float(min_val)}f"

            if max_val is None:
                max_str = "1e30f"
            else:
                max_str = f"{float(max_val)}f"

            return f"nnc_clip({input_arg}, {output_arg}, {min_str}, {max_str});"

        return f"/* Unknown operator {op_type} */"

    def _generate_header(self, ctx: CompileContext) -> str:
        """Generate header file."""
        lines = [
            "/* Auto-generated by NTC - DO NOT EDIT */",
            "#ifndef MODEL_H",
            "#define MODEL_H",
            "",
            "#include <stdint.h>",
            '#include "nnc_types.h"',
            "",
            "/* Tensor declarations */",
        ]

        # Declare all non-constant tensors
        for tensor_name in ctx.graph.inputs:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"extern Tensor {var_name};")

        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.inputs or tensor_name in ctx.graph.constants:
                continue
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            lines.append(f"extern Tensor {var_name};")

        for tensor_name in ctx.graph.outputs:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            if f"extern Tensor {var_name};" not in lines:
                lines.append(f"extern Tensor {var_name};")

        # Declare constant tensors (defined in constants_loader.c)
        for tensor_name in ctx.graph.constants:
            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            if f"extern Tensor {var_name};" not in lines:
                lines.append(f"extern Tensor {var_name};")

        # Declare constant data arrays for use in shape operations
        import numpy as np
        for tensor_name, arr in ctx.graph.constants.items():
            # Only declare INT64 arrays (used for shapes)
            if arr.dtype == np.int64 or arr.dtype == np.int32:
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                dtype = "int64_t" if arr.dtype == np.int64 else "int32_t"
                size = arr.size
                lines.append(f"extern const {dtype} {var_name}_data[{size}];")

        lines.extend([
            "",
            "/* Constants loading function */",
            "int nnc_load_constants(const char* path);",
            "",
            "/* Main inference function */",
            "void nnc_run(void);",
        ])

        entry_point = self._get_public_entry_point(ctx)
        if entry_point != "nnc_run":
            lines.extend([
                "",
                "/* Public entry point alias */",
                f"void {entry_point}(void);",
            ])

        lines.extend([
            "",
            "#endif  // MODEL_H",
        ])

        return "\n".join(lines)

    def _generate_constants_binary(self, ctx: CompileContext) -> tuple[bytes, dict[str, Any]]:
        """Generate constants as binary file with metadata.

        Binary format:
        - Header: magic number (4 bytes), version (4 bytes), num_constants (4 bytes)
        - For each constant:
            - name_len (4 bytes), name (null-terminated string)
            - dtype (4 bytes: 0=float32, 1=float16, 2=int32, 3=int8, 4=uint8, 5=bool)
            - ndim (4 bytes)
            - shape (ndim * 8 bytes, int64_t each)
            - nbytes (8 bytes)
            - data (nbytes)

        Returns:
            Tuple of (binary_data, metadata_dict)
        """
        import struct
        import numpy as np

        MAGIC = b'NNCB'
        VERSION = 1

        # Build metadata dict for C code generation
        metadata: dict[str, Any] = {}

        # First pass: calculate offsets
        offset = 12  # header: magic (4) + version (4) + num_constants (4)

        constant_entries: list[dict[str, Any]] = []
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            shape = list(arr.shape)
            ndim = len(shape)
            nbytes = arr.nbytes

            # Map dtype to enum
            dtype_enum = {
                "float": 0,
                "double": 0,
                "uint16_t": 1,
                "int64_t": 2,
                "int32_t": 3,
                "int8_t": 4,
                "uint8_t": 5,
            }.get(dtype, 0)

            # Calculate entry size
            name_bytes = var_name.encode('utf-8')
            name_len = len(name_bytes) + 1  # null-terminated

            entry_header_size = 4 + name_len + 4 + 4 + (ndim * 8) + 8
            # name_len + name + dtype + ndim + shape + nbytes

            data_offset = offset + entry_header_size

            constant_entries.append({
                'name': var_name,
                'original_name': name,
                'dtype': dtype,
                'dtype_enum': dtype_enum,
                'nnc_dtype': nnc_dtype,
                'shape': shape,
                'ndim': ndim,
                'nbytes': nbytes,
                'element_size': element_size,
                'data': arr.tobytes(),
                'offset': data_offset,
            })

            metadata[var_name] = {
                'dtype': dtype,
                'nnc_dtype': nnc_dtype,
                'shape': shape,
                'ndim': ndim,
                'nbytes': nbytes,
                'offset': data_offset,
            }

            offset = data_offset + nbytes

        # Build binary data with interleaved headers and data
        parts: list[bytes] = []
        parts.append(MAGIC)
        parts.append(struct.pack('<I', VERSION))
        parts.append(struct.pack('<I', len(constant_entries)))

        for entry in constant_entries:
            name = str(entry['name'])
            name_bytes = name.encode('utf-8')
            parts.append(struct.pack('<I', len(name_bytes) + 1))  # name_len
            parts.append(name_bytes + b'\x00')  # name (null-terminated)
            parts.append(struct.pack('<I', int(entry['dtype_enum'])))  # dtype
            parts.append(struct.pack('<I', int(entry['ndim'])))  # ndim
            # shape
            shape = entry['shape']
            if isinstance(shape, list):
                for dim in shape:
                    parts.append(struct.pack('<q', int(dim)))
            parts.append(struct.pack('<Q', int(entry['nbytes'])))  # nbytes
            # Data immediately follows header
            data = entry['data']
            if isinstance(data, bytes):
                parts.append(data)
            else:
                parts.append(bytes(data))

        return b''.join(parts), metadata

    def _generate_constants_loader(self, ctx: CompileContext, metadata: dict[str, Any]) -> str:
        """Generate C code to load constants from binary file."""
        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            '#include "nnc_types.h"',
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <string.h>',
            "",
            "/* Constants data loaded from binary file */",
            "",
        ]

        import numpy as np

        # Generate static data buffers for each constant
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            size = arr.size

            # For INT64/INT32 constants (used as shapes), make data array non-static
            is_shape_constant = (arr.dtype == np.int64 or arr.dtype == np.int32)
            storage_class = "" if is_shape_constant else "static "

            lines.append(f"/* Constant: {name} */")
            lines.append(f"{storage_class}{dtype} {var_name}_data[{size}];")
            lines.append("")

        lines.append("/* Tensor structures */")
        for name, arr in ctx.graph.constants.items():
            var_name = ctx.tensor_symbols.get(name, name)
            shape = list(arr.shape)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            # Handle symbolic dimensions (strings) by converting to -1
            shape_str = ", ".join(str(s) if isinstance(s, (int, float)) else "-1" for s in shape)

            lines.append(f"static const int64_t {var_name}_shape[] = {{{shape_str}}};")
            lines.append(f"Tensor {var_name} = {{")
            lines.append(f"    .data = (void*){var_name}_data,")
            lines.append(f"    .dtype = {nnc_dtype},")
            lines.append(f"    .shape = (int64_t*){var_name}_shape,")
            lines.append(f"    .ndim = {len(shape)},")
            lines.append(f"    .nbytes = {arr.nbytes},")
            lines.append("};")
            lines.append("")

        # Generate loading code - one section per constant in order
        lines.extend([
            "/* Load constants from binary file */",
            "int nnc_load_constants(const char* path) {",
            "    FILE* f = fopen(path, \"rb\");",
            "    if (!f) {",
            "        fprintf(stderr, \"Failed to open constants file: %s\\n\", path);",
            "        return -1;",
            "    }",
            "",
            "    /* Check magic number */",
            "    char magic[4];",
            "    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, \"NNCB\", 4) != 0) {",
            "        fprintf(stderr, \"Invalid constants file format\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            "    /* Check version */",
            "    uint32_t version;",
            "    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {",
            "        fprintf(stderr, \"Failed to read version\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "    if (version != 1) {",
            "        fprintf(stderr, \"Unsupported version: %u\\n\", version);",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            "    /* Get number of constants */",
            "    uint32_t num_constants;",
            "    if (fread(&num_constants, sizeof(uint32_t), 1, f) != 1) {",
            "        fprintf(stderr, \"Failed to read num_constants\\n\");",
            "        fclose(f);",
            "        return -1;",
            "    }",
            "",
            f"    (void)num_constants;  /* Will be implicitly checked by loading each constant */",
            "",
        ])

        # Generate a loading block for each constant - they must match the file order
        for idx, (name, arr) in enumerate(ctx.graph.constants.items()):
            var_name = ctx.tensor_symbols.get(name, name)
            dtype, element_size, nnc_dtype = self._get_constant_type_info(arr)
            ndim = len(arr.shape)
            # Handle symbolic dimensions (strings) by converting to -1
            shape_str = ", ".join(str(s) if isinstance(s, (int, float)) else "-1" for s in arr.shape)
            data_size = arr.nbytes

            lines.extend([
                f"    /* Load constant {idx}: {name} */",
                f"    {{",
                f"        /* Read name */",
                f"        uint32_t name_len;",
                f"        if (fread(&name_len, sizeof(uint32_t), 1, f) != 1) {{",
                f"            fprintf(stderr, \"Failed to read name_len for constant {idx}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        char name[256];",
                f"        if (name_len >= sizeof(name)) {{",
                f"            fprintf(stderr, \"Name too long for constant {idx}: %u\\\\n\", name_len);",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        if (fread(name, 1, name_len, f) != name_len) {{",
                f"            fprintf(stderr, \"Failed to read name for constant {idx}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        if (strcmp(name, \"{var_name}\") != 0) {{",
                f"            fprintf(stderr, \"Name mismatch for constant {idx}: expected '{var_name}', got '%s'\\\\n\", name);",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"        /* Skip dtype, ndim, shape - we know them from codegen */",
                f"        fseek(f, 4, SEEK_CUR);  /* dtype */",
                f"        fseek(f, 4, SEEK_CUR);  /* ndim */",
                f"        fseek(f, {ndim * 8}, SEEK_CUR);  /* shape ({ndim} dims) */",
                f"        fseek(f, 8, SEEK_CUR);  /* nbytes */",
                f"        /* Read data */",
                f"        if (fread({var_name}_data, 1, {data_size}, f) != {data_size}) {{",
                f"            fprintf(stderr, \"Failed to read data for {var_name}\\\\n\");",
                f"            fclose(f);",
                f"            return -1;",
                f"        }}",
                f"    }}",
                f"",
            ])

        lines.extend([
            "    fclose(f);",
            "    return 0;",
            "}",
        ])

        return "\n".join(lines)

    def _get_constant_type_info(self, arr: np.ndarray) -> tuple[str, int, str]:
        """Get C type, element size, and NNC dtype enum for a constant array.

        Returns:
            Tuple of (c_dtype, element_size, nnc_dtype_enum)
        """
        import numpy as np

        if arr.dtype == np.float32:
            return "float", 4, "NNC_DTYPE_FLOAT32"
        elif arr.dtype == np.float64:
            return "double", 8, "NNC_DTYPE_FLOAT32"
        elif arr.dtype == np.float16:
            return "uint16_t", 2, "NNC_DTYPE_FLOAT16"
        elif arr.dtype == np.int64:
            return "int64_t", 8, "NNC_DTYPE_INT64"
        elif arr.dtype == np.int32:
            return "int32_t", 4, "NNC_DTYPE_INT32"
        elif arr.dtype == np.int8:
            return "int8_t", 1, "NNC_DTYPE_INT8"
        elif arr.dtype == np.uint8:
            return "uint8_t", 1, "NNC_DTYPE_UINT8"
        elif arr.dtype == np.bool_:
            return "uint8_t", 1, "NNC_DTYPE_BOOL"
        else:
            # Default to float32
            return "float", 4, "NNC_DTYPE_FLOAT32"

    def _generate_makefile(self, ctx: CompileContext) -> str:
        """Generate Makefile."""
        has_constants = bool(ctx.graph.constants)
        objs = "model.o tensors.o"
        if has_constants:
            objs += " constants_loader.o"
        objs += " test_runner.o ops.o"

        # Determine runtime path relative to the output directory
        # For installed package: use NNC_RUNTIME_PATH env var
        # For development: relative path from output to project root

        # Different makefile content based on whether we have constants
        if has_constants:
            makefile_body = f"""# Auto-generated by NNC - DO NOT EDIT
# Set NNC_RUNTIME_PATH to point to the runtime directory if not in dev tree
CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra
LDFLAGS = -lm

# Runtime include path - can be overridden by environment
NNC_RUNTIME ?= ../../runtime
CFLAGS += -I$(NNC_RUNTIME)/include

.PHONY: all clean run

all: model

model: {objs}
\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ops.o:
\t$(CC) $(CFLAGS) -c $(NNC_RUNTIME)/x86/ops.c

%.o: %.c
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f *.o model

run: model
\t./model
"""
        else:
            makefile_body = f"""# Auto-generated by NNC - DO NOT EDIT
# Set NNC_RUNTIME_PATH to point to the runtime directory if not in dev tree
CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra
LDFLAGS = -lm

# Runtime include path - can be overridden by environment
NNC_RUNTIME ?= ../../runtime
CFLAGS += -I$(NNC_RUNTIME)/include

.PHONY: all clean run

all: model

model: {objs}
\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ops.o:
\t$(CC) $(CFLAGS) -c $(NNC_RUNTIME)/x86/ops.c

%.o: %.c
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f *.o model

run: model
\t./model
"""

        return makefile_body

    def _generate_tensors(self, ctx: CompileContext) -> str:
        """Generate tensors definition file."""
        lines = [
            "/* Auto-generated by NNC - DO NOT EDIT */",
            '#include "nnc_types.h"',
            "",
        ]

        # Check for new memory allocation plan
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        alloc_plan = get_memory_allocation_plan(ctx)

        has_slow_memory_tensors = False
        if alloc_plan is not None:
            has_slow_memory_tensors = any(
                alloc.is_spilled for alloc in alloc_plan.tensor_allocations.values()
            )
        has_spill = alloc_plan is not None and alloc_plan.has_spill
        has_moves = alloc_plan is not None and bool(alloc_plan.move_points)
        uses_unified_runtime = alloc_plan is not None and (
            has_spill or has_slow_memory_tensors or has_moves
        )

        # Generate slow pool if we have spill points OR slow memory tensors
        needs_slow_pool = has_spill or has_slow_memory_tensors

        # Check if memory planning was performed
        has_memory_plan = alloc_plan is not None or "memory_plan" in ctx.metadata

        if has_memory_plan:
            # Generate static memory pool(s)
            lines.extend(self._generate_memory_pool(ctx))
            lines.append("")

        # Determine which pool names to use
        if uses_unified_runtime:
            fast_pool_name = "_nnc_fast_pool"
            slow_pool_name = "_nnc_slow_pool" if needs_slow_pool else None
        else:
            fast_pool_name = "_nnc_memory_pool"
            slow_pool_name = None

        # Get tensor offsets from allocation plan
        tensor_offsets = {}
        spilled_tensors = set()

        if alloc_plan is not None:
            # Use new MemoryAllocationPlan
            for tensor_name, alloc in alloc_plan.tensor_allocations.items():
                if alloc.is_spilled:
                    # Spilled tensors go to slow memory
                    tensor_offsets[tensor_name] = ("slow", alloc.offset)
                    spilled_tensors.add(tensor_name)
                else:
                    # Non-spilled tensors go to fast memory
                    # Use alloc.offset which is the tensor's offset within the buffer
                    tensor_offsets[tensor_name] = ("fast", alloc.offset)
        elif "memory_plan" in ctx.metadata:
            # Use legacy memory plan
            from nnc_py.passes.memory_plan import get_memory_plan
            from nnc_py.passes.spill import get_spill_plan
            plan = get_memory_plan(ctx)
            spill_plan = get_spill_plan(ctx)

            if spill_plan is not None and spill_plan.has_overflow:
                # Has spill from legacy plan
                spilled_tensors = set(spill_plan.spilled_tensors)

            for tensor_name, mem_info in plan.tensor_info.items():
                pool_type = "slow" if tensor_name in spilled_tensors else "fast"
                tensor_offsets[tensor_name] = (pool_type, mem_info.pool_offset)

        # Define all non-constant tensors
        for tensor_name, tensor in ctx.graph.tensors.items():
            if tensor_name in ctx.graph.constants:
                continue

            var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
            shape_list = tensor.shape.dims
            # Handle symbolic dimensions (strings) by converting to -1
            shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

            data_init = "NULL"
            if tensor_name in tensor_offsets:
                pool_type, offset = tensor_offsets[tensor_name]
                if uses_unified_runtime and tensor_name in ctx.graph.inputs:
                    input_buffer_name = f"_nnc_input_buffer_{var_name}"
                    lines.append(f"/* Input staging buffer: {tensor_name} */")
                    lines.append(
                        f"uint8_t {input_buffer_name}[{tensor.byte_size()}] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};"
                    )
                    lines.append("")
                    data_init = input_buffer_name
                else:
                    if pool_type == "slow":
                        pool_to_use = slow_pool_name
                    else:
                        pool_to_use = fast_pool_name
                    data_init = f"{pool_to_use} + {offset}" if pool_to_use else "NULL"

            lines.append(f"/* Tensor: {tensor_name} */")
            lines.append(f"static int64_t {var_name}_shape[] = {{{shape_init}}};")

            # Map dtype
            dtype_enum = self._map_dtype_to_enum(tensor.dtype)

            lines.append(f"Tensor {var_name} = {{")
            lines.append(f"    .data = {data_init},")
            lines.append(f"    .dtype = {dtype_enum},")
            lines.append(f"    .shape = {var_name}_shape,")
            lines.append(f"    .ndim = {len(shape_list)},")
            lines.append(f"    .nbytes = {tensor.byte_size()},")
            lines.append("};")
            lines.append("")

        return "\n".join(lines)

    def _generate_memory_pool(self, ctx: CompileContext) -> List[str]:
        """Generate static memory pool declaration."""
        from nnc_py.passes.memory_planning import get_memory_allocation_plan
        from nnc_py.passes.memory_plan import get_memory_plan
        from nnc_py.passes.spill import get_spill_plan

        # Check for new memory allocation plan first
        alloc_plan = get_memory_allocation_plan(ctx)

        if alloc_plan is not None:
            has_slow_memory_tensors = any(
                alloc.is_spilled for alloc in alloc_plan.tensor_allocations.values()
            )
            uses_unified_runtime = (
                alloc_plan.has_spill
                or has_slow_memory_tensors
                or bool(alloc_plan.move_points)
            )

            if uses_unified_runtime:
                # Use max_memory when available so transient pre-compaction offsets fit.
                # Move-only plans still need the unified fast pool symbol.
                move_count = len(alloc_plan.move_points)
                spill_count = alloc_plan.spill_count
                reload_count = alloc_plan.reload_count
                needs_slow_pool = alloc_plan.has_spill or has_slow_memory_tensors
                max_memory = ctx.metadata.get('max_memory', alloc_plan.total_fast_memory)
                fast_memory_size = max(alloc_plan.total_fast_memory, max_memory)
                lines = [
                    "/* Unified Memory Pools */",
                    f"/* Fast memory: {fast_memory_size} bytes ({fast_memory_size / 1024:.2f} KB) */",
                    f"/* Buffers: {alloc_plan.num_buffers}, Spill points: {spill_count}, Reload points: {reload_count}, Move points: {move_count} */",
                    "",
                    "/* Fast Memory Pool (SRAM/On-chip) */",
                    f"#define NNC_FAST_MEMORY_SIZE {fast_memory_size}",
                    "#define NNC_MEMORY_ALIGNMENT 16",
                    f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                    f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                    "",
                ]

                if needs_slow_pool:
                    lines.extend([
                        "/* Slow Memory Pool (DRAM/External) */",
                        f"#define NNC_SLOW_MEMORY_SIZE {alloc_plan.total_slow_memory}",
                        f"uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] "
                        f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                        "",
                    ])
                return lines

        # Fall back to legacy implementation
        plan = get_memory_plan(ctx)
        spill_plan = get_spill_plan(ctx)

        # Check if we have spill (overflow)
        has_spill = spill_plan is not None and spill_plan.has_overflow

        if has_spill and spill_plan is not None:
            # Generate dual memory pools
            lines = [
                "/* Dual Memory Pools (Fast + Slow for overflow) */",
                f"/* Fast memory limit: {spill_plan.max_memory} bytes ({spill_plan.max_memory / 1024:.2f} KB) */",
                f"/* Original requirement: {plan.total_size} bytes ({plan.total_size / 1024:.2f} KB) */",
                f"/* Slow memory used: {spill_plan.slow_memory_size} bytes ({spill_plan.slow_memory_size / 1024:.2f} KB) */",
                f"/* Spilled tensors: {len(spill_plan.spilled_tensors)} */",
                "",
                f"/* Fast Memory Pool (SRAM/On-chip) */",
                f"#define NNC_FAST_MEMORY_SIZE {spill_plan.max_memory}",
                f"#define NNC_MEMORY_ALIGNMENT {plan.alignment}",
                f"uint8_t _nnc_fast_pool[NNC_FAST_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
                f"/* Slow Memory Pool (DRAM/External) */",
                f"#define NNC_SLOW_MEMORY_SIZE {spill_plan.slow_memory_size}",
                f"uint8_t _nnc_slow_pool[NNC_SLOW_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
            ]
            return lines
        else:
            # Single memory pool (no overflow)
            if plan is None:
                # Fallback if no plan available
                lines = [
                    "/* Static Memory Pool */",
                    "#define NNC_MEMORY_SIZE 4096",
                    "#define NNC_MEMORY_ALIGNMENT 16",
                    "static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                    "__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {0};",
                    "",
                ]
                return lines
            lines = [
                "/* Static Memory Pool */",
                f"/* Total size: {plan.total_size} bytes ({plan.total_size / 1024:.2f} KB) */",
                f"/* Buffers: {plan.num_buffers}, Tensors: {plan.num_tensors} */",
                f"#define NNC_MEMORY_SIZE {plan.total_size}",
                f"#define NNC_MEMORY_ALIGNMENT {plan.alignment}",
                f"static uint8_t _nnc_memory_pool[NNC_MEMORY_SIZE] "
                f"__attribute__((aligned(NNC_MEMORY_ALIGNMENT))) = {{0}};",
                "",
            ]
            return lines

    def _map_dtype_to_enum(self, dtype: "DataType") -> str:
        """Map IR dtype to NNC dtype enum."""
        from nnc_py.ir.types import DataType

        mapping = {
            DataType.FLOAT32: "NNC_DTYPE_FLOAT32",
            DataType.FLOAT16: "NNC_DTYPE_FLOAT16",
            DataType.INT32: "NNC_DTYPE_INT32",
            DataType.INT64: "NNC_DTYPE_INT64",
            DataType.INT8: "NNC_DTYPE_INT8",
            DataType.UINT8: "NNC_DTYPE_UINT8",
            DataType.BOOL: "NNC_DTYPE_BOOL",
        }
        return mapping.get(dtype, "NNC_DTYPE_FLOAT32")

    def _generate_test_runner(self, ctx: CompileContext) -> str:
        """Generate test runner."""
        # Check if memory planning was performed
        has_memory_plan = "memory_plan" in ctx.metadata

        # Debug mode: add file setup for debug output
        debug_decl = ""
        debug_setup = ""
        debug_cleanup = ""
        debug_macros = ""
        if self.debug_mode:
            # Declare debug_file as a global variable (extern declaration in model.c)
            debug_decl = """
/* Global debug file pointer (also declared as extern in model.c) */
FILE* debug_file = NULL;
"""
            debug_setup = """
    /* Open debug output file */
    debug_file = fopen("nnc_debug_output.txt", "w");
    if (!debug_file) {
        fprintf(stderr, "Failed to open debug output file\\n");
        return 1;
    }
"""
            debug_cleanup = """
    /* Close debug output file */
    if (debug_file) {
        fclose(debug_file);
        debug_file = NULL;
    }
    printf("Debug output written to nnc_debug_output.txt\\n");"""
            # Define macros to redirect debug printf to file
            debug_macros = """
/* Debug mode: redirect debug output to file */
#define DEBUG_PRINTF(fmt, ...) fprintf(debug_file, fmt, ##__VA_ARGS__)
#define DEBUG_PRINT_TENSOR_START(name, idx) DEBUG_PRINTF("DEBUG_TENSOR_START %s %d\\n", name, idx)
#define DEBUG_PRINT_SHAPE(ndim) DEBUG_PRINTF("SHAPE %d\\n", ndim)
#define DEBUG_PRINT_DIM(i, val) DEBUG_PRINTF("DIM %d %d\\n", i, val)
#define DEBUG_PRINT_DATA_START() DEBUG_PRINTF("DATA_START\\n")
#define DEBUG_PRINT_VALUE(val) DEBUG_PRINTF("%f\\n", val)
#define DEBUG_PRINT_DATA_END() DEBUG_PRINTF("DATA_END\\n")
#define DEBUG_PRINT_TENSOR_END(name) DEBUG_PRINTF("DEBUG_TENSOR_END %s\\n\\n", name)
"""

        # Generate tensor setup code
        tensor_setups = []

        if has_memory_plan:
            # Static allocation - memory is pre-allocated in memory pool
            # Only need to initialize input tensors with test data
            from nnc_py.passes.memory_plan import get_memory_plan
            plan = get_memory_plan(ctx)

            tensor_setups.append(f"    /* Using static memory pool: {plan.total_size} bytes */")
            tensor_setups.append("")

            # Setup input tensors - only initialize with test data
            for tensor_name in ctx.graph.inputs:
                tensor = ctx.graph.get_tensor(tensor_name)
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()

                # Get memory info
                if tensor_name in plan.tensor_info:
                    mem_info = plan.tensor_info[tensor_name]
                    tensor_setups.append(f"    /* Initialize {var_name} at offset {mem_info.pool_offset} */")
                else:
                    tensor_setups.append(f"    /* Initialize {var_name} */")

                # Initialize with test pattern
                num_elements = size // 4
                tensor_setups.append(f"    for (int i = 0; i < {num_elements}; i++) {{")
                tensor_setups.append(f"        ((float*){var_name}.data)[i] = (float)i * 0.01f;  /* Test pattern */")
                tensor_setups.append(f"    }}")
                tensor_setups.append("")
        else:
            # Dynamic allocation - use malloc/calloc
            # Setup input tensors
            for tensor_name in ctx.graph.inputs:
                tensor = ctx.graph.get_tensor(tensor_name)
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()

                # Generate shape array
                shape_list = tensor.shape.dims
                # Handle symbolic dimensions (strings) by converting to -1
                shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

                # Setup code
                tensor_setups.append(f"    /* Setup {var_name} */")
                tensor_setups.append(f"    static int64_t {var_name}_shape[] = {{{shape_init}}};")
                tensor_setups.append(f"    {var_name}.shape = {var_name}_shape;")
                tensor_setups.append(f"    {var_name}.ndim = {len(shape_list)};")
                tensor_setups.append(f"    {var_name}.dtype = NNC_DTYPE_FLOAT32;")
                tensor_setups.append(f"    {var_name}.nbytes = {size};")
                tensor_setups.append(f"    {var_name}.data = malloc({size});")
                tensor_setups.append(f"    if (!{var_name}.data) {{ fprintf(stderr, \"Failed to allocate {var_name}\\\\n\"); return 1; }}")

                # Initialize with test pattern
                num_elements = size // 4
                tensor_setups.append(f"    for (int i = 0; i < {num_elements}; i++) {{")
                tensor_setups.append(f"        ((float*){var_name}.data)[i] = (float)i * 0.01f;  /* Test pattern */")
                tensor_setups.append(f"    }}")
                tensor_setups.append("")

            # Setup intermediate and output tensors
            for tensor_name, tensor in ctx.graph.tensors.items():
                if tensor_name in ctx.graph.inputs or tensor_name in ctx.graph.constants:
                    continue

                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                size = tensor.byte_size()
                shape_list = tensor.shape.dims
                # Handle symbolic dimensions (strings) by converting to -1
                shape_init = ", ".join(str(d) if isinstance(d, (int, float)) else "-1" for d in shape_list)

                tensor_setups.append(f"    /* Setup {var_name} */")
                tensor_setups.append(f"    static int64_t {var_name}_shape[] = {{{shape_init}}};")
                tensor_setups.append(f"    {var_name}.shape = {var_name}_shape;")
                tensor_setups.append(f"    {var_name}.ndim = {len(shape_list)};")
                tensor_setups.append(f"    {var_name}.dtype = NNC_DTYPE_FLOAT32;")
                tensor_setups.append(f"    {var_name}.nbytes = {size};")
                tensor_setups.append(f"    {var_name}.data = calloc({size // 4}, sizeof(float));  /* Initialize to zero */")
                tensor_setups.append(f"    if (!{var_name}.data) {{ fprintf(stderr, \"Failed to allocate {var_name}\\\\n\"); return 1; }}")
                tensor_setups.append("")

        setup_code = "\n".join(tensor_setups)

        # Free memory - only for dynamic allocation
        if has_memory_plan:
            frees_code = "    /* Static allocation - no free needed */"
        else:
            tensor_frees = []
            for tensor_name in ctx.graph.tensors:
                if tensor_name in ctx.graph.constants:
                    continue
                var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)
                tensor_frees.append(f"    free({var_name}.data);")
            frees_code = "\n".join(tensor_frees)

        # Get input and output tensor variable names
        input_var = None
        if ctx.graph.inputs:
            input_name = ctx.graph.inputs[0]
            input_var = ctx.tensor_symbols.get(input_name, input_name)

        output_var = None
        if ctx.graph.outputs:
            output_name = ctx.graph.outputs[0]
            output_var = ctx.tensor_symbols.get(output_name, output_name)

        # Generate print code only if we have input/output tensors
        print_code = ""

        # Add constants loading code if there are constants
        has_constants = bool(ctx.graph.constants)
        if has_constants:
            constants_load_code = """
    /* Load constants from binary file */
    printf("Loading constants from constants.bin...\\n");
    if (nnc_load_constants("constants.bin") != 0) {
        fprintf(stderr, "Failed to load constants\\n");
        return 1;
    }
    printf("Constants loaded successfully.\\n");"""
        else:
            constants_load_code = ""

        if input_var and output_var:
            print_code = f"""
{constants_load_code}
    /* Print input info */
    printf("Input data (first 10):\\n");
    int64_t input_size = 1;
    for (int i = 0; i < {input_var}.ndim; i++) {{
        input_size *= {input_var}.shape[i];
    }}
    for (int i = 0; i < 10 && i < input_size; i++) {{
        printf("  input[%d] = %f\\n", i, ((float*){input_var}.data)[i]);
    }}

    /* Run inference */
    printf("\\nRunning inference...\\n");
    nnc_run();

    /* Print output results */
    printf("\\nOutput results (first 10):\\n");
    int64_t output_size = 1;
    for (int i = 0; i < {output_var}.ndim; i++) {{
        output_size *= {output_var}.shape[i];
    }}
    for (int i = 0; i < 10 && i < output_size; i++) {{
        printf("  output[%d] = %f\\n", i, ((float*){output_var}.data)[i]);
    }}"""
        else:
            if constants_load_code:
                print_code = f"""
{constants_load_code}

    /* Run inference */
    printf("Running inference...\\n");
    nnc_run();"""
            else:
                print_code = """
    /* Run inference */
    printf("Running inference...\\n");
    nnc_run();"""

        return f"""/* Auto-generated by NNC - DO NOT EDIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnc_types.h"
#include "model.h"
{debug_macros}
{debug_decl}
int main(void) {{
    printf("NNC Model Runner\\n");
    printf("================\\n");
{debug_setup}
    /* Setup all tensors */
{setup_code}
{print_code}

    /* Free memory */
{frees_code}
{debug_cleanup}
    printf("\\nInference complete.\\n");
    return 0;
}}
"""

    def _assign_symbols(self, ctx: CompileContext) -> None:
        """Assign C symbol names using NameManager."""
        name_manager = NameManager()

        # Assign names for tensors
        for tensor_name, tensor in ctx.graph.tensors.items():
            # Use NameManager to sanitize the name
            c_name = name_manager.get_symbol(tensor_name, prefix="tensor_")
            ctx.tensor_symbols[tensor_name] = c_name

        # Assign names for nodes
        for node_name in ctx.graph.nodes:
            c_name = name_manager.get_symbol(node_name, prefix="node_")
            ctx.node_symbols[node_name] = c_name

    def _map_dtype(self, dtype: DataType) -> str:
        """Map IR dtype to C dtype."""
        mapping = {
            DataType.FLOAT32: "float",
            DataType.FLOAT16: "uint16_t",
            DataType.INT32: "int32_t",
            DataType.INT8: "int8_t",
            DataType.UINT8: "uint8_t",
            DataType.BOOL: "uint8_t",
        }
        return mapping.get(dtype, "float")

    def _map_numpy_dtype(self, np_dtype: Any) -> str:
        """Map numpy dtype to C dtype."""
        import numpy as np

        if np_dtype == np.float32:
            return "float"
        elif np_dtype == np.float16:
            return "uint16_t"
        elif np_dtype == np.int32:
            return "int32_t"
        elif np_dtype == np.int8:
            return "int8_t"
        elif np_dtype == np.uint8:
            return "uint8_t"
        else:
            return "float"

    def _generate_debug_dump_code(self, ctx: CompileContext, tensor_name: str, node_idx: int, node_name: str) -> str:
        """Generate C code to dump tensor values for debug comparison.

        Args:
            ctx: Compilation context.
            tensor_name: Name of the tensor to dump.
            node_idx: Index of the node that produced this tensor.
            node_name: Name of the node that produced this tensor.

        Returns:
            C code string that dumps tensor values.
        """
        tensor = ctx.graph.get_tensor(tensor_name)
        if tensor is None:
            return ""

        var_name = ctx.tensor_symbols.get(tensor_name, tensor_name)

        # Calculate element size from dtype
        elem_size_map = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
        }
        elem_size = elem_size_map.get(tensor.dtype, 4)

        # Calculate number of elements from byte size
        byte_size = tensor.byte_size()
        if byte_size < 0:
            # Unknown size due to symbolic dimensions
            num_elements = -1
        else:
            num_elements = byte_size // elem_size

        # Get number of dimensions
        ndim = len(tensor.shape.dims)

        # Determine how to read the tensor data based on dtype
        # BOOL tensors are stored as uint8_t, need conversion to float for output
        # INT64 tensors (e.g., from Shape operator) are stored as int64_t
        is_bool = tensor.dtype == DataType.BOOL
        is_int64 = tensor.dtype == DataType.INT64
        if is_bool:
            # For BOOL, read as uint8_t and convert to float
            data_read_expr = f"((uint8_t*){var_name}.data)[i]"
        elif is_int64:
            # For INT64, read as int64_t and convert to float
            data_read_expr = f"(float)((int64_t*){var_name}.data)[i]"
        else:
            # For other types, read as float
            data_read_expr = f"((float*){var_name}.data)[i]"

        if self.debug_mode:
            # Use debug macros for file output
            code = f"""
    /* Debug dump: {tensor_name} after node {node_idx} ({node_name}) */
    DEBUG_PRINT_TENSOR_START("{tensor_name}", {node_idx});
    DEBUG_PRINT_SHAPE({ndim});
"""
            for i, dim in enumerate(tensor.shape.dims):
                if isinstance(dim, int):
                    code += f'    DEBUG_PRINT_DIM({i}, {dim});\n'
                else:
                    # For symbolic dimensions, use the actual index value (not Python variable i)
                    code += f'    DEBUG_PRINT_DIM({i}, (int){var_name}.shape[{i}]);\n'

            code += f"""
    DEBUG_PRINT_DATA_START();
    for (int i = 0; i < {num_elements}; i++) {{
        DEBUG_PRINT_VALUE((float){data_read_expr});
    }}
    DEBUG_PRINT_DATA_END();
    DEBUG_PRINT_TENSOR_END("{tensor_name}");
"""
        else:
            # Standard printf output (for debugging compiler itself)
            code = f"""
    /* Debug dump: {tensor_name} after node {node_idx} ({node_name}) */
    printf("DEBUG_TENSOR_START %s %d\\n", "{tensor_name}", {node_idx});
    printf("SHAPE %d\\n", {ndim});
"""
            for i, dim in enumerate(tensor.shape.dims):
                if isinstance(dim, int):
                    code += f'    printf("DIM {i} %d\\\\n", {dim});\n'
                else:
                    # For symbolic dimensions, use the actual index value (not Python variable i)
                    code += f'    printf("DIM {i} %d\\\\n", (int){var_name}.shape[{i}]);\n'

            code += f"""
    printf("DATA_START\\\\n");
    for (int i = 0; i < {num_elements}; i++) {{
        printf("%f\\\\n", (float){data_read_expr});
    }}
    printf("DATA_END\\\\n");
    printf("DEBUG_TENSOR_END %s\\\\n\\n", "{tensor_name}");
"""
        return code

    def _inject_debug_into_nnc_run(self, source_code: str, ctx: CompileContext) -> str:
        """Inject debug dump code into nnc_run function.

        Args:
            source_code: Generated C source code.
            ctx: Compilation context.

        Returns:
            Modified source code with debug dumps.
        """
        if not self.debug_mode:
            return source_code

        lines = source_code.split("\n")
        output = []
        in_nnc_run = False
        brace_count = 0
        node_idx = 0
        nodes = ctx.graph.topological_sort()

        for i, line in enumerate(lines):
            output.append(line)

            # Detect nnc_run function start
            if "void nnc_run(void)" in line:
                in_nnc_run = True
                brace_count = 0
                continue

            if in_nnc_run:
                # Count braces to track function scope
                brace_count += line.count("{") - line.count("}")

                # After each function call, add debug dump for outputs
                for node in nodes:
                    if node.op_type == OpType.CONSTANT:
                        continue
                    func_name = ctx.node_symbols.get(node.name, node.name)
                    if f"{func_name}();" in line:
                        # Add debug dump for each output tensor of this node
                        for out_name in node.outputs:
                            tensor = ctx.graph.get_tensor(out_name)
                            if tensor is not None and out_name not in ctx.graph.constants:
                                debug_code = self._generate_debug_dump_code(
                                    ctx, out_name, node_idx, node.name
                                )
                                # Add indentation and lines
                                for debug_line in debug_code.strip().split("\n"):
                                    output.append(debug_line)
                        node_idx += 1
                        break

                # End of function
                if brace_count == 0 and "}" in line:
                    in_nnc_run = False

        return "\n".join(output)
