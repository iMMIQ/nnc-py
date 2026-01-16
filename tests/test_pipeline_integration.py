"""Tests for split passes pipeline integration (TDD Cycle 9)."""

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.ir.node import Node, OpType
from nnc_py.ir.tensor import TensorType, TensorShape
from nnc_py.ir.types import DataType
from nnc_py.passes.base import PassManager
from nnc_py.passes.operators.conv_rules import register_conv2d_split_rule


class TestPipelineIntegration:
    """Test that split passes integrate correctly with the pass pipeline."""

    def test_split_passes_are_available(self):
        """Test that split passes can be imported."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass

        analysis = SplitAnalysisPass()
        transform = SplitTransformPass()

        assert analysis.name == "split_analysis"
        assert transform.name == "split_transform"

    def test_split_passes_have_correct_base(self):
        """Test that split passes inherit from PassBase."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass
        from nnc_py.passes.base import PassBase

        assert issubclass(SplitAnalysisPass, PassBase)
        assert issubclass(SplitTransformPass, PassBase)

    def test_passes_can_be_manually_registered(self):
        """Test that split passes can be manually added to PassManager."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass

        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())

        assert len(manager.passes) == 2
        assert manager.passes[0].name == "split_analysis"
        assert manager.passes[1].name == "split_transform"

    def test_passes_run_in_sequence(self):
        """Test that passes run in the correct sequence."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass

        graph = Graph(name="test_sequence")
        ctx = CompileContext(graph=graph, target="x86")

        # Create a simple graph
        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 3, 64, 64]),
            name="input"
        ))
        graph.inputs.append("input")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[64, 64, 64, 64]),
            name="output"
        ))

        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "weights", "bias"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        # Set memory limit
        ctx.metadata["max_memory"] = 10 * 1024 * 1024
        register_conv2d_split_rule()

        # Run passes in sequence
        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        # Check that both passes ran
        assert "split_plan" in ctx.metadata
        assert "conv1_split0" in graph.nodes or "conv1" in graph.nodes

    def test_split_analysis_before_transform(self):
        """Test that analysis pass runs before transform pass."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass

        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())

        assert manager.passes[0].name == "split_analysis"
        assert manager.passes[1].name == "split_transform"

    def test_pass_execution_order_is_recorded(self):
        """Test that PassManager records which passes ran."""
        from nnc_py.passes.split_analysis import SplitAnalysisPass
        from nnc_py.passes.split_transform import SplitTransformPass

        graph = Graph(name="test_record")
        ctx = CompileContext(graph=graph, target="x86")

        graph.add_tensor(TensorType(
            dtype=DataType.FLOAT32,
            shape=TensorShape(dims=[4, 3, 4, 4]),
            name="input"
        ))
        graph.inputs.append("input")

        node = Node(
            op_type=OpType.CONV2D,
            name="conv1",
            inputs=["input", "w", "b"],
            outputs=["output"],
            attrs={}
        )
        graph.add_node(node)

        manager = PassManager()
        manager.register(SplitAnalysisPass())
        manager.register(SplitTransformPass())
        manager.run(ctx)

        assert len(manager.applied_passes) == 2
        assert "split_analysis" in manager.applied_passes
        assert "split_transform" in manager.applied_passes

    def test_split_passes_in_default_pipeline_o2(self):
        """Test that split passes are in the default O2 pipeline."""
        passes = PassManager.get_default_passes(opt_level=2)
        pass_names = [p.name for p in passes]

        assert "split_analysis" in pass_names
        assert "split_transform" in pass_names

    def test_split_passes_correct_order_in_pipeline(self):
        """Test that split passes are in the correct order in the pipeline."""
        passes = PassManager.get_default_passes(opt_level=2)
        pass_names = [p.name for p in passes]

        # Liveness should be before split analysis
        assert pass_names.index("LivenessAnalysis") < pass_names.index("split_analysis")
        # Split analysis should be before split transform
        assert pass_names.index("split_analysis") < pass_names.index("split_transform")
        # Split transform should be before memory planning (MemoryPlanningV2)
        assert pass_names.index("split_transform") < pass_names.index("MemoryPlanningV2")

    def test_split_passes_not_in_o0_or_o1(self):
        """Test that split passes are not in O0 or O1 pipelines."""
        passes_o0 = PassManager.get_default_passes(opt_level=0)
        passes_o1 = PassManager.get_default_passes(opt_level=1)

        pass_names_o0 = [p.name for p in passes_o0]
        pass_names_o1 = [p.name for p in passes_o1]

        assert "split_analysis" not in pass_names_o0
        assert "split_transform" not in pass_names_o0
        assert "split_analysis" not in pass_names_o1
        assert "split_transform" not in pass_names_o1
