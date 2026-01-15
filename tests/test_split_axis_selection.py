"""Tests for split axis selection (TDD Cycle 4)."""

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.ir.split_rules import (
    SplitRegistry,
    SplitAxisBehavior,
    SplitAxisRule,
    OperatorSplitRule,
)
from nnc_py.passes.split_analysis import SplitAnalysisPass


class TestSplitAxisSelection:
    """Test split axis selection logic."""

    def test_select_largest_axis(self):
        """Test that the largest axis is selected for splitting."""
        # Create a rule where all axes are splittable
        rule = OperatorSplitRule(
            op_type=OpType.ADD,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(1, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(2, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(3, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )
        SplitRegistry.register(rule)

        # [1024, 512, 224, 224] -> should select axis 0 (1024)
        graph = Graph(name="test_axis_select")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 10 * 1024 * 1024

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="output"
        ))

        node = Node(
            op_type=OpType.ADD,
            name="add1",
            inputs=["input", "input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        split_plan = ctx.metadata["split_plan"]
        assert len(split_plan.splits) > 0
        # Should select axis 0 (largest dimension)
        assert split_plan.splits[0].split_axis == 0

    def test_reduction_axis_not_selected(self):
        """Test that reduction axis is not selected for splitting."""
        # Create a rule where axis 0 is NOT splittable (reduction)
        rule = OperatorSplitRule(
            op_type=OpType.REDUCE_MEAN,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.REDUCTION_FORBIDDEN),
                SplitAxisRule(1, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(2, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )
        SplitRegistry.register(rule)

        # [100, 512, 224, 224] -> should NOT select axis 0 even though it's not largest
        graph = Graph(name="test_no_reduction")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 5 * 1024 * 1024

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 256, 112, 112]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 256, 112, 112]),
            name="output"
        ))

        node = Node(
            op_type=OpType.REDUCE_MEAN,
            name="reduce1",
            inputs=["input"],
            outputs=["output"],
            attrs={"axes": [0]}  # Reducing on axis 0
        )
        graph.add_node(node)

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        split_plan = ctx.metadata.get("split_plan")
        # Either no split or axis is not 0
        if split_plan and len(split_plan.splits) > 0:
            assert split_plan.splits[0].split_axis != 0

    def test_select_second_largest_when_first_unavailable(self):
        """Test selecting the second largest axis when the largest is not splittable."""
        # Create a rule where only axes 1, 2, 3 are splittable
        rule = OperatorSplitRule(
            op_type=OpType.MUL,
            input_split_rules=[[
                SplitAxisRule(1, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(2, SplitAxisBehavior.FULLY_SPLITTABLE),
                SplitAxisRule(3, SplitAxisBehavior.FULLY_SPLITTABLE),
            ]],
            output_split_behavior=[SplitAxisBehavior.FULLY_SPLITTABLE],
            reused_inputs=set(),
        )
        SplitRegistry.register(rule)

        # [1024, 512, 224, 224] -> should select axis 1 (512) since 0 is not available
        graph = Graph(name="test_second_largest")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 10 * 1024 * 1024

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[1024, 512, 224, 224]),
            name="output"
        ))

        node = Node(
            op_type=OpType.MUL,
            name="mul1",
            inputs=["input", "input"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        split_plan = ctx.metadata["split_plan"]
        assert len(split_plan.splits) > 0
        # Should select axis 1 (second largest, since 0 is not in rule)
        assert split_plan.splits[0].split_axis == 1

    def test_no_splittable_axis_returns_none(self):
        """Test that None is returned when no axis is splittable."""
        # Create a rule with NO splittable axes
        rule = OperatorSplitRule(
            op_type=OpType.CONCAT,
            input_split_rules=[[
                SplitAxisRule(0, SplitAxisBehavior.SHAPE_CHANGE_FORBIDDEN),
            ]],
            output_split_behavior=[SplitAxisBehavior.SHAPE_CHANGE_FORBIDDEN],
            reused_inputs=set(),
        )
        SplitRegistry.register(rule)

        graph = Graph(name="test_no_split")
        ctx = CompileContext(graph=graph, target="x86")
        ctx.metadata["max_memory"] = 1 * 1024 * 1024

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 256, 112, 112]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[100, 256, 112, 112]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONCAT,
            name="concat1",
            inputs=["input", "input2"],
            outputs=["output"],
            attrs={"axis": 0}
        )
        graph.add_node(node)

        pass_obj = SplitAnalysisPass()
        pass_obj.run(ctx)

        # Should not create any splits
        split_plan = ctx.metadata.get("split_plan")
        if split_plan:
            assert len(split_plan.splits) == 0
