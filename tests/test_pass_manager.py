"""Test pass manager integration of DominatorFusionPass"""

from nnc_py.passes.base import PassManager


def test_o3_includes_dominator_fusion():
    """Verify O3 passes include DominatorFusionPass after PatternFusionPass"""
    o3_passes = PassManager.get_default_passes(3)

    # Find indices of PatternFusionPass and DominatorFusionPass
    pattern_fusion_index = None
    dominator_fusion_index = None

    for i, pass_obj in enumerate(o3_passes):
        if pass_obj.__class__.__name__ == "PatternFusionPass":
            pattern_fusion_index = i
        elif pass_obj.__class__.__name__ == "DominatorFusionPass":
            dominator_fusion_index = i

    # Both passes should be present
    assert pattern_fusion_index is not None, "PatternFusionPass should be in O3 passes"
    assert dominator_fusion_index is not None, "DominatorFusionPass should be in O3 passes"

    # DominatorFusionPass should come after PatternFusionPass
    assert dominator_fusion_index > pattern_fusion_index, \
        "DominatorFusionPass should come after PatternFusionPass in O3 passes"


def test_o2_no_dominator_fusion():
    """Verify O2 passes do not include DominatorFusionPass"""
    o2_passes = PassManager.get_default_passes(2)

    # Check that DominatorFusionPass is not in O2 passes
    for pass_obj in o2_passes:
        assert pass_obj.__class__.__name__ != "DominatorFusionPass", \
            "DominatorFusionPass should not be in O2 passes"