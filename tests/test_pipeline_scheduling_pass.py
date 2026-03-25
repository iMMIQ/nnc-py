from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.pipeline_schedule import (
    PipelineResourceKind,
    PipelineScheduleProblem,
    PipelineScheduleResult,
    ScheduledStep,
    ScheduleStep,
    ScheduleStepKind,
)
from nnc_py.ir.pipeline_schedule import set_pipeline_schedule_problem
from nnc_py.passes import PipelineSchedulingPass


class _FalseyScheduler:
    def __bool__(self) -> bool:
        return False

    def solve(self, problem: PipelineScheduleProblem) -> PipelineScheduleResult:
        return PipelineScheduleResult(
            scheduled_steps=(
                ScheduledStep(
                    step_id=problem.steps[0].id,
                    resource_kind=problem.steps[0].resource_kind,
                    resource_slot=0,
                    start_time=1,
                    end_time=2,
                ),
            ),
            makespan=2,
            feasible=True,
            solver_name="falsey-test",
        )


def test_pipeline_scheduling_pass_stores_typed_schedule_result():
    ctx = CompileContext(graph=Graph("schedule_pass"), target="x86", optimization_level=3)
    set_pipeline_schedule_problem(
        ctx,
        PipelineScheduleProblem(
            steps=(
                ScheduleStep(
                    id="compute0",
                    node_name="compute0",
                    step_kind=ScheduleStepKind.COMPUTE,
                    resource_kind=PipelineResourceKind.MATMUL,
                    duration=4,
                ),
            ),
            resources=(PipelineResourceKind.MATMUL,),
            sram_capacity_bytes=0,
        ),
    )

    PipelineSchedulingPass().run(ctx)

    assert isinstance(ctx.pipeline_schedule_result, PipelineScheduleResult)
    assert ctx.pipeline_schedule_result.feasible is True
    assert ctx.pipeline_schedule_result.solver_name == "list"
    assert ctx.pipeline_schedule_result.scheduled_steps == (
        ScheduledStep(
            step_id="compute0",
            resource_kind=PipelineResourceKind.MATMUL,
            resource_slot=0,
            start_time=0,
            end_time=4,
        ),
    )


def test_pipeline_scheduling_pass_is_noop_when_problem_is_absent():
    ctx = CompileContext(graph=Graph("schedule_pass_no_problem"), target="x86", optimization_level=3)

    PipelineSchedulingPass().run(ctx)

    assert ctx.pipeline_schedule_result is None


def test_pipeline_scheduling_pass_preserves_explicit_falsey_scheduler_injection():
    ctx = CompileContext(graph=Graph("schedule_pass_falsey_scheduler"), target="x86", optimization_level=3)
    set_pipeline_schedule_problem(
        ctx,
        PipelineScheduleProblem(
            steps=(
                ScheduleStep(
                    id="compute0",
                    node_name="compute0",
                    step_kind=ScheduleStepKind.COMPUTE,
                    resource_kind=PipelineResourceKind.MATMUL,
                    duration=4,
                ),
            ),
            resources=(PipelineResourceKind.MATMUL,),
            sram_capacity_bytes=0,
        ),
    )

    PipelineSchedulingPass(scheduler=_FalseyScheduler()).run(ctx)

    assert ctx.pipeline_schedule_result is not None
    assert ctx.pipeline_schedule_result.solver_name == "falsey-test"
    assert ctx.pipeline_schedule_result.scheduled_steps[0].start_time == 1
