"""Test pass manager integration of higher-level optimization passes."""

from nnc_py.passes.base import PassManager


def test_o3_includes_prepack_and_dominator_fusion():
    """Verify O3 includes PrepackLoweringPass between pattern and dominator fusion."""
    o3_passes = PassManager.get_default_passes(3)

    pattern_fusion_index = None
    prepack_lowering_index = None
    dominator_fusion_index = None

    for i, pass_obj in enumerate(o3_passes):
        if pass_obj.__class__.__name__ == "PatternFusionPass":
            pattern_fusion_index = i
        elif pass_obj.__class__.__name__ == "PrepackLoweringPass":
            prepack_lowering_index = i
        elif pass_obj.__class__.__name__ == "DominatorFusionPass":
            dominator_fusion_index = i

    assert pattern_fusion_index is not None, "PatternFusionPass should be in O3 passes"
    assert prepack_lowering_index is not None, "PrepackLoweringPass should be in O3 passes"
    assert dominator_fusion_index is not None, "DominatorFusionPass should be in O3 passes"
    assert pattern_fusion_index < prepack_lowering_index < dominator_fusion_index, (
        "PrepackLoweringPass should run after PatternFusionPass and before DominatorFusionPass"
    )


def test_o2_no_dominator_fusion():
    """Verify O2 passes do not include DominatorFusionPass"""
    o2_passes = PassManager.get_default_passes(2)

    # Check that DominatorFusionPass is not in O2 passes
    for pass_obj in o2_passes:
        assert pass_obj.__class__.__name__ != "DominatorFusionPass", \
            "DominatorFusionPass should not be in O2 passes"
