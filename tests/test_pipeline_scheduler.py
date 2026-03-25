from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    ScheduleDependencyKind,
    ScheduleEdge,
    ScheduleStep,
    ScheduleStepKind,
    SramValue,
)
from nnc_py.scheduler import ListPipelineScheduler


def _scheduled_steps_by_id(result):
    return {step.step_id: step for step in result.scheduled_steps}


def _intervals_by_value_name(result):
    return {interval.value_name: interval for interval in result.sram_intervals}


def test_list_scheduler_allows_dma_to_overlap_with_matmul():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="dma0",
                node_name="load0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=5,
                sram_output_names=("input0",),
            ),
            ScheduleStep(
                id="compute0",
                node_name="matmul0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=7,
            ),
        ),
        sram_values=(SramValue(name="input0", size_bytes=8, producer_step_id="dma0"),),
        resources=(PipelineResourceKind.DMA, PipelineResourceKind.MATMUL),
        sram_capacity_bytes=64,
    )

    result = ListPipelineScheduler().solve(problem)

    scheduled = _scheduled_steps_by_id(result)
    assert result.feasible is True
    assert scheduled["dma0"].start_time == 0
    assert scheduled["dma0"].end_time == 5
    assert scheduled["compute0"].start_time == 0
    assert scheduled["compute0"].end_time == 7
    assert result.makespan == 7


def test_list_scheduler_serializes_dma_steps():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="dma0",
                node_name="load0",
                step_kind=ScheduleStepKind.DMA_IN,
                resource_kind=PipelineResourceKind.DMA,
                duration=3,
            ),
            ScheduleStep(
                id="dma1",
                node_name="store0",
                step_kind=ScheduleStepKind.DMA_OUT,
                resource_kind=PipelineResourceKind.DMA,
                duration=4,
            ),
        ),
        resources=(PipelineResourceKind.DMA,),
        sram_capacity_bytes=0,
    )

    result = ListPipelineScheduler().solve(problem)

    scheduled = _scheduled_steps_by_id(result)
    intervals = sorted(
        [
            (scheduled["dma0"].start_time, scheduled["dma0"].end_time),
            (scheduled["dma1"].start_time, scheduled["dma1"].end_time),
        ]
    )
    assert result.feasible is True
    assert intervals[0][0] == 0
    assert intervals[0][1] == intervals[1][0]
    assert intervals[1][1] == 7
    assert result.makespan == 7


def test_list_scheduler_delays_ready_step_until_sram_budget_allows_it():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="producer_a",
                node_name="producer_a",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_output_names=("value_a",),
            ),
            ScheduleStep(
                id="consumer_a",
                node_name="consumer_a",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_input_names=("value_a",),
            ),
            ScheduleStep(
                id="producer_b",
                node_name="producer_b",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
                sram_output_names=("value_b",),
            ),
        ),
        edges=(
            ScheduleEdge("producer_a", "consumer_a", ScheduleDependencyKind.DATA),
        ),
        sram_values=(
            SramValue(
                name="value_a",
                size_bytes=6,
                producer_step_id="producer_a",
                consumer_step_ids=("consumer_a",),
            ),
            SramValue(name="value_b", size_bytes=6, producer_step_id="producer_b"),
        ),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=6,
    )

    result = ListPipelineScheduler().solve(problem)

    scheduled = _scheduled_steps_by_id(result)
    intervals = _intervals_by_value_name(result)
    assert result.feasible is True
    assert scheduled["producer_a"].start_time == 0
    assert scheduled["producer_a"].end_time == 2
    assert scheduled["consumer_a"].start_time == 2
    assert scheduled["consumer_a"].end_time == 5
    assert scheduled["producer_b"].start_time == 5
    assert scheduled["producer_b"].end_time == 7
    assert intervals["value_a"].start_time == 2
    assert intervals["value_a"].end_time == 5
    assert intervals["value_b"].start_time == 7
    assert intervals["value_b"].end_time == 7
    assert result.makespan == 7


def test_list_scheduler_accounts_for_overlapping_temp_sram_across_step_interval():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="matmul0",
                node_name="matmul0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=6,
                sram_temp_bytes=5,
            ),
            ScheduleStep(
                id="other0",
                node_name="other0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.OTHER,
                duration=3,
                sram_temp_bytes=4,
            ),
        ),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=7,
    )

    result = ListPipelineScheduler().solve(problem)

    scheduled = _scheduled_steps_by_id(result)
    assert result.feasible is True
    assert scheduled["matmul0"].start_time == 0
    assert scheduled["matmul0"].end_time == 6
    assert scheduled["other0"].start_time == 6
    assert scheduled["other0"].end_time == 9
    assert result.makespan == 9


def test_list_scheduler_is_deterministic_for_tied_ready_steps_and_favors_matmul():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="shape0",
                node_name="shape0",
                step_kind=ScheduleStepKind.SHAPE_PREP,
                resource_kind=PipelineResourceKind.SHAPE,
                duration=2,
            ),
            ScheduleStep(
                id="matmul0",
                node_name="matmul0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=2,
            ),
        ),
        resources=(PipelineResourceKind.SHAPE, PipelineResourceKind.MATMUL),
        sram_capacity_bytes=0,
    )

    scheduler = ListPipelineScheduler()
    result_a = scheduler.solve(problem)
    result_b = scheduler.solve(problem)

    assert result_a == result_b
    assert result_a.diagnostics["scheduled_order"] == ("matmul0", "shape0")


def test_list_scheduler_returns_infeasible_result_when_step_can_never_fit_in_sram():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="too_large",
                node_name="too_large",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
                sram_output_names=("big_value",),
            ),
        ),
        sram_values=(SramValue(name="big_value", size_bytes=9, producer_step_id="too_large"),),
        resources=(PipelineResourceKind.MATMUL,),
        sram_capacity_bytes=8,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.makespan == 0
    assert result.diagnostics["reason"] == "sram_capacity_exceeded"
    assert result.diagnostics["step_id"] == "too_large"


def test_list_scheduler_returns_infeasible_result_for_cyclic_dependencies():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="step0",
                node_name="step0",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
            ),
            ScheduleStep(
                id="step1",
                node_name="step1",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
            ),
        ),
        edges=(
            ScheduleEdge("step0", "step1", ScheduleDependencyKind.DATA),
            ScheduleEdge("step1", "step0", ScheduleDependencyKind.DATA),
        ),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=0,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.diagnostics["reason"] == "cyclic_dependencies"


def test_list_scheduler_rejects_missing_tracked_sram_value_reference():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="producer0",
                node_name="producer0",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
                sram_output_names=("missing_value",),
            ),
        ),
        resources=(PipelineResourceKind.MATMUL,),
        sram_capacity_bytes=16,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.diagnostics["reason"] == "unknown_sram_value_reference"
    assert result.diagnostics["step_id"] == "producer0"
    assert result.diagnostics["value_name"] == "missing_value"


def test_list_scheduler_rejects_unknown_step_references_in_sram_values():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="consumer0",
                node_name="consumer0",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
                sram_input_names=("value0",),
            ),
        ),
        sram_values=(
            SramValue(
                name="value0",
                size_bytes=4,
                producer_step_id="missing_producer",
                consumer_step_ids=("consumer0", "missing_consumer"),
            ),
        ),
        resources=(PipelineResourceKind.OTHER,),
        sram_capacity_bytes=16,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.diagnostics["reason"] == "unknown_sram_step_reference"
    assert result.diagnostics["value_name"] == "value0"


def test_list_scheduler_rejects_mismatched_sram_producer_ownership():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="producer0",
                node_name="producer0",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
                sram_output_names=("value0",),
            ),
            ScheduleStep(
                id="other0",
                node_name="other0",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
            ),
        ),
        sram_values=(
            SramValue(
                name="value0",
                size_bytes=4,
                producer_step_id="other0",
            ),
        ),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=16,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.diagnostics["reason"] == "sram_producer_mismatch"
    assert result.diagnostics["step_id"] == "producer0"
    assert result.diagnostics["value_name"] == "value0"


def test_list_scheduler_rejects_missing_sram_consumer_ownership():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="producer0",
                node_name="producer0",
                resource_kind=PipelineResourceKind.MATMUL,
                duration=1,
                sram_output_names=("value0",),
            ),
            ScheduleStep(
                id="consumer0",
                node_name="consumer0",
                resource_kind=PipelineResourceKind.OTHER,
                duration=1,
                sram_input_names=("value0",),
            ),
        ),
        sram_values=(
            SramValue(
                name="value0",
                size_bytes=4,
                producer_step_id="producer0",
                consumer_step_ids=(),
            ),
        ),
        resources=(PipelineResourceKind.MATMUL, PipelineResourceKind.OTHER),
        sram_capacity_bytes=16,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is False
    assert result.scheduled_steps == ()
    assert result.diagnostics["reason"] == "sram_consumer_mismatch"
    assert result.diagnostics["step_id"] == "consumer0"
    assert result.diagnostics["value_name"] == "value0"


def test_list_scheduler_treats_zero_capacity_as_unbounded_default():
    problem = PipelineScheduleProblem(
        steps=(
            ScheduleStep(
                id="compute0",
                node_name="compute0",
                step_kind=ScheduleStepKind.COMPUTE,
                resource_kind=PipelineResourceKind.MATMUL,
                duration=3,
                sram_output_names=("value0",),
                sram_temp_bytes=4,
            ),
        ),
        sram_values=(SramValue(name="value0", size_bytes=8, producer_step_id="compute0"),),
        resources=(PipelineResourceKind.MATMUL,),
        sram_capacity_bytes=0,
    )

    result = ListPipelineScheduler().solve(problem)

    assert result.feasible is True
    assert len(result.scheduled_steps) == 1
    assert result.scheduled_steps[0].step_id == "compute0"
    assert result.makespan == 3
