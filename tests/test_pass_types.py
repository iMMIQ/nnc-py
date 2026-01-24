"""测试 Pass 系统类型定义。"""

import pytest
from nnc_py.passes.base import PassBase, PassManager
from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph


def test_pass_manager_typed():
    """PassManager 应具有正确的类型。"""
    manager = PassManager()
    assert isinstance(manager.passes, list)
    assert isinstance(manager.applied_passes, list)


def test_pass_base_abstract():
    """PassBase 应该是抽象类。"""
    # 尝试实例化抽象类应该失败
    with pytest.raises(TypeError):
        PassBase()
