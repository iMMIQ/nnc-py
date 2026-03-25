import json
import subprocess
from types import MappingProxyType

import pytest

from nnc_py.ir.pipeline_schedule import PipelineResourceKind, ScheduleStepKind


def _base_estimate_kwargs() -> dict[str, object]:
    return {
        "op_type": "Reshape",
        "step_kind": ScheduleStepKind.SHAPE_PREP,
        "resource_kind": PipelineResourceKind.SHAPE,
        "input_shapes": ((1, 64, 1, 1),),
        "output_shapes": ((1, 64),),
        "dtypes": ("float32",),
        "tensor_bytes": 256,
        "attrs": None,
    }


def test_simple_cost_model_applies_non_zero_launch_overhead():
    from nnc_py.cost_model.simple import SimpleCostModelProvider

    provider = SimpleCostModelProvider()

    estimate = provider.estimate_step(**_base_estimate_kwargs())

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.latency >= estimate.launch_overhead
    assert estimate.source == "simple"


def test_simple_cost_model_keeps_tiny_workloads_positive():
    from nnc_py.cost_model.simple import SimpleCostModelProvider

    provider = SimpleCostModelProvider()

    estimate = provider.estimate_step(
        op_type="Identity",
        step_kind=ScheduleStepKind.DMA_IN,
        resource_kind=PipelineResourceKind.DMA,
        input_shapes=((1,),),
        output_shapes=((1,),),
        dtypes=("float32",),
        tensor_bytes=1,
        attrs=None,
    )

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.breakdown["work_units"] == 1


def test_simple_cost_model_accepts_mapping_like_attrs():
    from nnc_py.cost_model.simple import SimpleCostModelProvider

    provider = SimpleCostModelProvider()

    estimate = provider.estimate_step(
        op_type="MatMul",
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=PipelineResourceKind.MATMUL,
        input_shapes=((1, 4), (4, 8)),
        output_shapes=((1, 8),),
        dtypes=("float32",),
        tensor_bytes=48,
        attrs=MappingProxyType({"macs": 512}),
    )

    assert estimate.breakdown["macs"] == 512


def test_simple_cost_model_labels_inferred_matmul_work_honestly():
    from nnc_py.cost_model.simple import SimpleCostModelProvider

    provider = SimpleCostModelProvider()

    estimate = provider.estimate_step(
        op_type="MatMul",
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=PipelineResourceKind.MATMUL,
        input_shapes=((1, 4), (4, 8)),
        output_shapes=((1, 8),),
        dtypes=("float32",),
        tensor_bytes=48,
        attrs=None,
    )

    assert "macs" not in estimate.breakdown
    assert estimate.breakdown["inferred_work"] > 0


def test_cost_estimate_breakdown_is_read_only():
    from nnc_py.cost_model.simple import SimpleCostModelProvider

    provider = SimpleCostModelProvider()
    estimate = provider.estimate_step(**_base_estimate_kwargs())

    with pytest.raises(TypeError):
        estimate.breakdown["extra"] = 1


def test_cli_cost_model_falls_back_when_command_is_missing():
    from nnc_py.cost_model.cli import CliCostModelProvider

    provider = CliCostModelProvider(command=["/definitely/missing/cost-model"])

    estimate = provider.estimate_step(**_base_estimate_kwargs())

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.source == "simple"


def test_cli_cost_model_falls_back_on_timeout(monkeypatch):
    from nnc_py.cost_model.cli import CliCostModelProvider

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    estimate = provider.estimate_step(**_base_estimate_kwargs())

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.source == "simple"


def test_cli_cost_model_falls_back_on_non_zero_exit(monkeypatch):
    from nnc_py.cost_model.cli import CliCostModelProvider

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=7,
            stdout="",
            stderr="bad request",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    estimate = provider.estimate_step(**_base_estimate_kwargs())

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.source == "simple"


def test_cli_cost_model_uses_cache_for_repeat_queries(monkeypatch):
    from nnc_py.cost_model.cli import CliCostModelProvider

    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "latency": 41,
                    "launch_overhead": 7,
                    "breakdown": {"work_units": 34},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    first = provider.estimate_step(**_base_estimate_kwargs())
    second = provider.estimate_step(**_base_estimate_kwargs())

    assert first == second
    assert first.source == "cli"
    assert len(calls) == 1


def test_cli_cost_model_accepts_mapping_like_attrs_and_serializes_supported_payload(
    monkeypatch,
):
    from nnc_py.cost_model.cli import CliCostModelProvider

    seen_inputs: list[dict[str, object]] = []

    def fake_run(*args, **kwargs):
        seen_inputs.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"latency": 19, "launch_overhead": 5}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    estimate = provider.estimate_step(
        op_type="MatMul",
        step_kind=ScheduleStepKind.COMPUTE,
        resource_kind=PipelineResourceKind.MATMUL,
        input_shapes=((1, 4), (4, 8)),
        output_shapes=((1, 8),),
        dtypes=("float32",),
        tensor_bytes=48,
        attrs=MappingProxyType(
            {
                "macs": 512,
                "layout": MappingProxyType({"tile": (1, 2)}),
            }
        ),
    )

    assert estimate.source == "cli"
    assert seen_inputs == [
        {
            "op_type": "MatMul",
            "step_kind": "compute",
            "resource_kind": "matmul",
            "input_shapes": [[1, 4], [4, 8]],
            "output_shapes": [[1, 8]],
            "dtypes": ["float32"],
            "tensor_bytes": 48,
            "attrs": {"layout": {"tile": [1, 2]}, "macs": 512},
        }
    ]


def test_cli_cost_model_rejects_unsupported_attr_payload():
    from nnc_py.cost_model.cli import CliCostModelProvider

    provider = CliCostModelProvider(command=["fake-cost-model"])

    with pytest.raises(TypeError, match="unsupported attr"):
        provider.estimate_step(
            op_type="MatMul",
            step_kind=ScheduleStepKind.COMPUTE,
            resource_kind=PipelineResourceKind.MATMUL,
            input_shapes=((1, 4), (4, 8)),
            output_shapes=((1, 8),),
            dtypes=("float32",),
            tensor_bytes=48,
            attrs=MappingProxyType({"bad": object()}),
        )


def test_cli_cost_model_falls_back_on_invalid_result(monkeypatch):
    from nnc_py.cost_model.cli import CliCostModelProvider

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"latency": -5, "launch_overhead": 0}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    estimate = provider.estimate_step(**_base_estimate_kwargs())

    assert estimate.latency > 0
    assert estimate.launch_overhead > 0
    assert estimate.source == "simple"


def test_cli_cost_model_does_not_pin_fallback_after_recovery(monkeypatch):
    from nnc_py.cost_model.cli import CliCostModelProvider

    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=0,
                stdout=json.dumps({"latency": -5, "launch_overhead": 0}),
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"latency": 31, "launch_overhead": 9}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CliCostModelProvider(command=["fake-cost-model"])

    first = provider.estimate_step(**_base_estimate_kwargs())
    second = provider.estimate_step(**_base_estimate_kwargs())

    assert first.source == "simple"
    assert second.source == "cli"
    assert call_count == 2
