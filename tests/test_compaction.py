"""Tests for intra-fast-memory compaction."""

import pytest

from nnc_py.passes.memory_strategy import (
    MovePoint,
    MemoryAllocationPlan,
)


def test_move_point_fields():
    mp = MovePoint(
        tensor_name="a",
        at_node_idx=2,
        from_offset=80,
        to_offset=0,
        size=64,
    )
    assert mp.tensor_name == "a"
    assert mp.at_node_idx == 2
    assert mp.from_offset == 80
    assert mp.to_offset == 0
    assert mp.size == 64


def test_plan_move_fields_default_empty():
    plan = MemoryAllocationPlan(strategy_name="test", total_fast_memory=256)
    assert plan.move_points == []
    assert plan.move_bytes == 0


def test_plan_get_move_points_at():
    mp0 = MovePoint("a", at_node_idx=2, from_offset=80, to_offset=0, size=64)
    mp1 = MovePoint("b", at_node_idx=2, from_offset=160, to_offset=64, size=64)
    mp2 = MovePoint("c", at_node_idx=5, from_offset=32, to_offset=0, size=32)
    plan = MemoryAllocationPlan(
        strategy_name="test",
        total_fast_memory=256,
        move_points=[mp0, mp1, mp2],
        move_bytes=160,
    )
    assert plan.get_move_points_at(2) == [mp0, mp1]
    assert plan.get_move_points_at(5) == [mp2]
    assert plan.get_move_points_at(0) == []
