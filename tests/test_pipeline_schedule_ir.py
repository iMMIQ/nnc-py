import json

import pytest

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ResidencyWindow,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduledStep,
    ScheduleStep,
    ScheduleStepKind,
    ScheduledValue,
    ScheduledValueHomeTier,
    SramAllocationInterval,
    SramValue,
    TransferStep,
    TransferStepKind,
    get_pipeline_schedule_problem,
    get_pipeline_schedule_result,
    set_pipeline_schedule_problem,
    set_pipeline_schedule_result,
)


def test_pipeline_schedule_problem_and_result_keep_expected_values():
    step = ScheduleStep(
        id="conv0.compute",
        node_name="conv0",
        tile_id="tile0",
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=PipelineResourceKind.MATMUL,
        duration=17,
        launch_overhead=3,
        sram_input_names=("conv0.tile0.in",),
        sram_output_names=("conv0.tile0.out",),
        sram_temp_bytes=1024,
        attrs={"op_type": "Conv"},
    )
    value = SramValue(
        name="conv0.tile0.out",
        size_bytes=4096,
        producer_step_id="conv0.compute",
        consumer_step_ids=("relu0.compute",),
        must_reside_in_sram=True,
    )
    edge = ScheduleEdge(
        src_step_id="conv0.compute",
        dst_step_id="relu0.compute",
        kind=ScheduleDependencyKind.DATA,
    )
    scheduled_step = ScheduledStep(
        step_id="conv0.compute",
        resource_kind=PipelineResourceKind.MATMUL,
        resource_slot=0,
        start_time=5,
        end_time=22,
    )
    interval = SramAllocationInterval(
        value_name="conv0.tile0.out",
        buffer_id="buf0",
        start_time=5,
        end_time=30,
        size_bytes=4096,
    )
    problem = PipelineScheduleProblem(
        steps=(step,),
        edges=(edge,),
        sram_values=(value,),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.DMA),
        sram_capacity_bytes=64 * 1024,
        metadata={"origin": "test"},
    )
    result = PipelineScheduleResult(
        scheduled_steps=(scheduled_step,),
        sram_intervals=(interval,),
        makespan=22,
        feasible=True,
        solver_name="list",
        diagnostics={"strategy": "serial"},
    )

    assert problem.steps == (step,)
    assert problem.edges == (edge,)
    assert problem.sram_values == (value,)
    assert problem.resources == (
        PipelineResourceKind.MATMUL,
        PipelineResourceKind.DMA,
    )
    assert problem.sram_capacity_bytes == 64 * 1024
    assert problem.objective == "min_makespan"
    assert problem.metadata == {"origin": "test"}
    assert result.scheduled_steps == (scheduled_step,)
    assert result.sram_intervals == (interval,)
    assert result.makespan == 22
    assert result.feasible is True
    assert result.solver_name == "list"
    assert result.diagnostics == {"strategy": "serial"}


def test_pipeline_schedule_payload_mappings_are_read_only():
    step = ScheduleStep(
        id="conv0.compute",
        node_name="conv0",
        attrs={"nested": {"op_type": "Conv"}, "dims": [1, 2, 3]},
    )
    problem = PipelineScheduleProblem(
        steps=(step,),
        metadata={"scheduler": {"name": "list"}},
    )
    result = PipelineScheduleResult(diagnostics={"solver": {"passes": 1}})

    with pytest.raises(TypeError):
        step.attrs["extra"] = "value"
    with pytest.raises(TypeError):
        step.attrs["nested"]["op_type"] = "Gemm"
    with pytest.raises(TypeError):
        problem.metadata["new"] = "value"
    with pytest.raises(TypeError):
        result.diagnostics["solver"]["passes"] = 2

    assert step.attrs["dims"] == (1, 2, 3)


def test_pipeline_schedule_constructors_copy_sequences_and_normalize_enum_strings():
    sram_inputs = ["act0", "act1"]
    sram_outputs = ["out0"]
    consumers = ["relu0.compute"]
    steps = [
        ScheduleStep(
            id="conv0.compute",
            node_name="conv0",
            step_kind="compute",
            resource_kind="matmul",
            sram_input_names=sram_inputs,
            sram_output_names=sram_outputs,
        )
    ]
    edges = [ScheduleEdge("conv0.compute", "relu0.compute", "data")]
    sram_values = [
        SramValue(
            name="conv0.out",
            size_bytes=64,
            consumer_step_ids=consumers,
        )
    ]
    resources = ["matmul", "dma"]
    scheduled_steps = [
        ScheduledStep(
            step_id="conv0.compute",
            resource_kind="matmul",
            resource_slot=0,
            start_time=0,
            end_time=5,
        )
    ]
    intervals = [
        SramAllocationInterval(
            value_name="conv0.out",
            buffer_id="buf0",
            start_time=0,
            end_time=5,
            size_bytes=64,
        )
    ]

    problem = PipelineScheduleProblem(
        steps=steps,
        edges=edges,
        sram_values=sram_values,
        resources=resources,
        sram_capacity_bytes=1024,
    )
    result = PipelineScheduleResult(
        scheduled_steps=scheduled_steps,
        sram_intervals=intervals,
        feasible=True,
        solver_name="list",
    )

    sram_inputs.append("act2")
    sram_outputs.append("out1")
    consumers.append("pool0.compute")
    steps.append(ScheduleStep(id="extra.compute", node_name="extra"))
    edges.append(ScheduleEdge("relu0.compute", "pool0.compute"))
    sram_values.append(SramValue(name="relu0.out", size_bytes=32))
    resources.append("shape")
    scheduled_steps.append(
        ScheduledStep(
            step_id="extra.compute",
            resource_kind=PipelineResourceKind.OTHER,
        )
    )
    intervals.append(
        SramAllocationInterval(
            value_name="relu0.out",
            buffer_id="buf1",
            start_time=5,
            end_time=9,
            size_bytes=32,
        )
    )

    assert problem.steps == (steps[0],)
    assert problem.edges == (edges[0],)
    assert problem.sram_values == (sram_values[0],)
    assert problem.resources == (
        PipelineResourceKind.MATMUL,
        PipelineResourceKind.DMA,
    )
    assert result.scheduled_steps == (scheduled_steps[0],)
    assert result.sram_intervals == (intervals[0],)
    assert problem.steps[0].step_kind is ScheduleStepKind.COMPUTE
    assert problem.steps[0].resource_kind is PipelineResourceKind.MATMUL
    assert problem.edges[0].kind is ScheduleDependencyKind.DATA
    assert result.scheduled_steps[0].resource_kind is PipelineResourceKind.MATMUL
    assert problem.steps[0].sram_input_names == ("act0", "act1")
    assert problem.steps[0].sram_output_names == ("out0",)
    assert problem.sram_values[0].consumer_step_ids == ("relu0.compute",)


def test_scheduled_value_tracks_home_tier_and_graph_tensor_name():
    value = ScheduledValue(
        name="sram|node|4:add0|tensor|1:y",
        graph_tensor_name="y",
        size_bytes=64,
        home_tier=ScheduledValueHomeTier.SLOW,
    )

    assert value.graph_tensor_name == "y"
    assert value.home_tier is ScheduledValueHomeTier.SLOW


def test_transfer_step_uses_dma_resource_and_names_moved_value():
    step = TransferStep(
        id="y.reload0",
        node_name="reload:y",
        transfer_kind=TransferStepKind.RELOAD_DMA,
        moved_value_name="y",
        bytes=64,
    )

    assert step.resource_kind is PipelineResourceKind.DMA
    assert step.step_kind is ScheduleStepKind.RELOAD_DMA


def test_pipeline_schedule_ir_exposes_json_ready_data():
    step = ScheduleStep(
        id="dma0.in",
        node_name="conv0",
        step_kind=ScheduleStepKind.DMA_IN,
        resource_kind=PipelineResourceKind.DMA,
        attrs={"shape": [1, 64], "tags": {"phase": "ingress"}},
    )
    problem = PipelineScheduleProblem(
        steps=(step,),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.MATMUL),
        metadata={"notes": ("alpha", "beta")},
    )
    result_value = ScheduledValue(
        name="conv0.out",
        graph_tensor_name="y",
        size_bytes=64,
        home_tier=ScheduledValueHomeTier.SLOW,
    )
    result_window = ResidencyWindow(
        value_name="conv0.out",
        residency_id="conv0.out@0",
        opened_by_step_id="dma0.in",
        closed_by_step_id="conv0.compute",
    )
    result = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="dma0.in",
                resource_kind=PipelineResourceKind.DMA,
                resource_slot=0,
                start_time=0,
                end_time=4,
            ),
        ),
        scheduled_values=(result_value,),
        residency_windows=(result_window,),
        feasible=True,
        solver_name="list",
        diagnostics={"status": {"kind": "ok"}},
    )

    problem_json = problem.to_json()
    result_json = result.to_json()

    assert json.loads(json.dumps(problem_json)) == {
        "steps": [
            {
                "id": "dma0.in",
                "node_name": "conv0",
                "tile_id": None,
                "step_kind": "dma_in",
                "resource_kind": "dma",
                "duration": 0,
                "launch_overhead": 0,
                "sram_input_names": [],
                "sram_output_names": [],
                "sram_temp_bytes": 0,
                "attrs": {
                    "shape": [1, 64],
                    "tags": {"phase": "ingress"},
                },
            }
        ],
        "edges": [],
        "sram_values": [],
        "scheduled_values": [],
        "residency_windows": [],
        "resources": ["dma", "matmul"],
        "sram_capacity_bytes": 0,
        "objective": "min_makespan",
        "metadata": {"notes": ["alpha", "beta"]},
    }
    assert json.loads(json.dumps(result_json)) == {
        "scheduled_steps": [
            {
                "step_id": "dma0.in",
                "resource_kind": "dma",
                "resource_slot": 0,
                "start_time": 0,
                "end_time": 4,
            }
        ],
        "sram_intervals": [],
        "scheduled_values": [
            {
                "name": "conv0.out",
                "graph_tensor_name": "y",
                "size_bytes": 64,
                "producer_step_id": None,
                "consumer_step_ids": [],
                "must_reside_in_sram": False,
                "can_alias": False,
                "home_tier": "slow",
            }
        ],
        "residency_windows": [
            {
                "value_name": "conv0.out",
                "residency_id": "conv0.out@0",
                "opened_by_step_id": "dma0.in",
                "closed_by_step_id": "conv0.compute",
            }
        ],
        "makespan": 0,
        "feasible": True,
        "solver_name": "list",
        "diagnostics": {"status": {"kind": "ok"}},
        "transfer_diagnostics": {},
    }


def test_pipeline_schedule_rejects_non_json_payloads():
    with pytest.raises(TypeError):
        ScheduleStep(id="conv0.compute", node_name="conv0", attrs={"bad": object()})

    with pytest.raises(TypeError, match="ScheduleStep.attrs must be a mapping"):
        ScheduleStep(id="conv0.compute", node_name="conv0", attrs=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        PipelineScheduleProblem(metadata={1: "bad-key"})

    with pytest.raises(
        TypeError, match="PipelineScheduleProblem.metadata must be a mapping"
    ):
        PipelineScheduleProblem(metadata=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        PipelineScheduleResult(diagnostics={"bad": object()})

    with pytest.raises(
        TypeError, match="PipelineScheduleResult.diagnostics must be a mapping"
    ):
        PipelineScheduleResult(diagnostics=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        ScheduleStep(id="conv0.compute", node_name="conv0", attrs={"bad": float("nan")})

    with pytest.raises(TypeError):
        PipelineScheduleProblem(metadata={"bad": float("inf")})

    with pytest.raises(TypeError):
        PipelineScheduleResult(diagnostics={"bad": float("-inf")})


def test_pipeline_schedule_rejects_invalid_enum_values():
    with pytest.raises(ValueError):
        ScheduleStep(
            id="conv0.compute",
            node_name="conv0",
            step_kind="not-a-step-kind",
        )

    with pytest.raises(ValueError):
        ScheduleEdge("a", "b", "not-a-dependency")

    with pytest.raises(ValueError):
        ScheduledStep(
            step_id="conv0.compute",
            resource_kind="not-a-resource",
        )


def test_pipeline_schedule_problem_rejects_inconsistent_value_shims():
    with pytest.raises(ValueError, match="scheduled_values must match sram_values"):
        PipelineScheduleProblem(
            sram_values=(SramValue(name="x", size_bytes=64),),
            scheduled_values=(
                ScheduledValue(
                    name="x",
                    graph_tensor_name="x",
                    size_bytes=128,
                    home_tier=ScheduledValueHomeTier.SLOW,
                ),
            ),
        )


def test_pipeline_schedule_metadata_helpers_validate_runtime_types():
    ctx = CompileContext(graph=Graph("typed_pipeline_ctx"), target="x86")
    problem = PipelineScheduleProblem()
    result = PipelineScheduleResult()

    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)

    assert get_pipeline_schedule_problem(ctx) is problem
    assert get_pipeline_schedule_result(ctx) is result

    with pytest.raises(TypeError):
        set_pipeline_schedule_problem(ctx, object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        set_pipeline_schedule_result(ctx, object())  # type: ignore[arg-type]

    ctx.metadata["pipeline_schedule_problem"] = "bad"
    ctx.metadata["pipeline_schedule_result"] = 123

    with pytest.raises(TypeError):
        get_pipeline_schedule_problem(ctx)

    with pytest.raises(TypeError):
        get_pipeline_schedule_result(ctx)

    with pytest.raises(TypeError):
        _ = ctx.pipeline_schedule_problem

    with pytest.raises(TypeError):
        _ = ctx.pipeline_schedule_result


def test_pipeline_schedule_accessors_store_and_load_typed_metadata():
    ctx = CompileContext(graph=Graph("pipeline_ctx"), target="x86")
    problem_value = ScheduledValue(
        name="shape0.out",
        graph_tensor_name="shape0_out",
        size_bytes=64,
        home_tier=ScheduledValueHomeTier.SRAM,
    )
    result_value = ScheduledValue(
        name="shape0.reload",
        graph_tensor_name="shape0_reload",
        size_bytes=128,
        home_tier=ScheduledValueHomeTier.SLOW,
    )
    problem_window = ResidencyWindow(
        value_name="shape0.out",
        residency_id="shape0.out@0",
        opened_by_step_id="shape0.prepare",
    )
    result_window = ResidencyWindow(
        value_name="shape0.reload",
        residency_id="shape0.reload@1",
        opened_by_step_id="shape0.prepare",
        closed_by_step_id="shape0.compute",
    )
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="shape0.prepare",
                node_name="shape0",
                step_kind=ScheduleStepKind.SHAPE_PREP,
                resource_kind=PipelineResourceKind.SHAPE,
                duration=4,
                launch_overhead=1,
            ),
        ),
        scheduled_values=(problem_value,),
        residency_windows=(problem_window,),
        sram_capacity_bytes=8 * 1024,
    )
    result = PipelineScheduleResult(
        scheduled_steps=(
            ScheduledStep(
                step_id="shape0.prepare",
                resource_kind=PipelineResourceKind.SHAPE,
                resource_slot=0,
                start_time=0,
                end_time=4,
            ),
        ),
        scheduled_values=(result_value,),
        residency_windows=(result_window,),
        makespan=4,
        feasible=True,
        solver_name="list",
        transfer_diagnostics={"shape0.out": {"kind": "resident"}},
    )

    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)

    assert ctx.metadata["pipeline_schedule_problem"] is problem
    assert ctx.metadata["pipeline_schedule_result"] is result
    assert get_pipeline_schedule_problem(ctx) is problem
    assert get_pipeline_schedule_result(ctx) is result
    assert ctx.pipeline_schedule_problem is problem
    assert ctx.pipeline_schedule_result is result
    assert problem.scheduled_values == (problem_value,)
    assert result.scheduled_values == (result_value,)
    assert problem.residency_windows == (problem_window,)
    assert result.residency_windows == (result_window,)
    assert ctx.pipeline_scheduled_values == (result_value,)
    assert ctx.get_pipeline_scheduled_values() == (result_value,)
    assert ctx.pipeline_residency_windows == (result_window,)
    assert ctx.get_pipeline_residency_windows() == (result_window,)
    assert ctx.pipeline_transfer_diagnostics == {"shape0.out": {"kind": "resident"}}
    assert ctx.get_pipeline_transfer_diagnostics() == {
        "shape0.out": {"kind": "resident"}
    }


def test_pipeline_schedule_accessors_respect_explicit_empty_result_metadata():
    ctx = CompileContext(graph=Graph("pipeline_ctx_empty_result"), target="x86")
    problem = PipelineScheduleProblem(
        scheduled_values=(
            ScheduledValue(
                name="shape0.out",
                graph_tensor_name="shape0_out",
                size_bytes=64,
                home_tier=ScheduledValueHomeTier.SRAM,
            ),
        ),
        residency_windows=(
            ResidencyWindow(
                value_name="shape0.out",
                residency_id="shape0.out@0",
                opened_by_step_id="shape0.prepare",
            ),
        ),
    )
    result = PipelineScheduleResult(
        scheduled_values=(),
        residency_windows=(),
        feasible=True,
        solver_name="list",
    )

    set_pipeline_schedule_problem(ctx, problem)
    set_pipeline_schedule_result(ctx, result)

    assert ctx.pipeline_scheduled_values == ()
    assert ctx.get_pipeline_scheduled_values() == ()
    assert ctx.pipeline_residency_windows == ()
    assert ctx.get_pipeline_residency_windows() == ()


def test_pipeline_schedule_accessors_do_not_mutate_missing_metadata_on_read():
    ctx = CompileContext(graph=Graph("empty_pipeline_ctx"), target="x86")

    assert "pipeline_schedule_problem" not in ctx.metadata
    assert "pipeline_schedule_result" not in ctx.metadata
    assert get_pipeline_schedule_problem(ctx) is None
    assert get_pipeline_schedule_result(ctx) is None
    assert ctx.pipeline_schedule_problem is None
    assert ctx.pipeline_schedule_result is None
    assert ctx.get_pipeline_schedule_problem() is None
    assert ctx.get_pipeline_schedule_result() is None
    assert ctx.pipeline_scheduled_values == ()
    assert ctx.get_pipeline_scheduled_values() == ()
    assert ctx.pipeline_residency_windows == ()
    assert ctx.get_pipeline_residency_windows() == ()
    assert ctx.pipeline_transfer_diagnostics == {}
    assert ctx.get_pipeline_transfer_diagnostics() == {}
    assert "pipeline_schedule_problem" not in ctx.metadata
    assert "pipeline_schedule_result" not in ctx.metadata
