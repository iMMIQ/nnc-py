"""Memory allocation strategy interface and registry.

This module provides a pluggable interface for memory allocation algorithms,
allowing different strategies to be selected at runtime.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Union, Type

from nnc_py.ir.context import CompileContext
from nnc_py.passes.memory_plan import MemoryBuffer


class AllocationStrategy(Enum):
    """Available memory allocation strategies."""
    BASIC = "basic"    # Basic sequential allocation with spill-all
    COST_AWARE = "cost_aware"


@dataclass
class TensorAllocation:
    """Memory allocation information for a single tensor."""
    tensor_name: str
    buffer_id: int                       # Which buffer/color this tensor uses
    offset: int                          # Offset within buffer (bytes)
    size: int                            # Tensor size (bytes)

    # For strategies supporting spill
    is_spilled: bool = False
    spill_after: Optional[int] = None           # Node index to spill after
    reload_before: Optional[List[int]] = None   # Node indices to reload before


@dataclass
class SpillPoint:
    """Spill operation point."""
    tensor_name: str
    after_node: str                       # Node name to spill after
    after_node_idx: int                   # Node index
    from_buffer_id: int                   # Source buffer ID
    from_fast_offset: int                 # Source offset in fast memory
    to_slow_offset: int                   # Destination offset in slow memory
    size: int


@dataclass
class ReloadPoint:
    """Reload operation point."""
    tensor_name: str
    before_node: str                      # Node name to reload before
    before_node_idx: int                  # Node index
    from_slow_offset: int                 # Source offset in slow memory
    to_buffer_id: int                     # Destination buffer ID
    to_fast_offset: int                   # Destination offset in fast memory
    size: int
    reload_slot_id: int = -1              # Which reload slot to use in fast memory


@dataclass
class MemoryAllocationPlan:
    """Unified memory allocation plan result.

    This is the single output format for all allocation strategies.
    """
    # Input/summary metadata
    strategy_name: str
    total_fast_memory: int                # Total fast memory used
    total_slow_memory: int = 0            # Slow memory for spilled tensors (0 if no spill)
    peak_memory: int = 0                  # Peak memory at any point
    num_buffers: int = 0                  # Number of buffers/colors

    # Buffer definitions (each buffer = a color in graph coloring)
    buffers: List[MemoryBuffer] = field(default_factory=list)

    # Per-tensor allocation info
    tensor_allocations: Dict[str, TensorAllocation] = field(default_factory=dict)
    tensor_to_buffer: Dict[str, int] = field(default_factory=dict)

    # Spill information (empty if no spill)
    spill_points: List[SpillPoint] = field(default_factory=list)
    reload_points: List[ReloadPoint] = field(default_factory=list)
    spill_bytes: int = 0
    reload_bytes: int = 0
    total_transfer_bytes: int = 0

    # Timing/liveness info for each node (optional, for debugging)
    node_memory_usage: List[int] = field(default_factory=list)

    @property
    def has_spill(self) -> bool:
        """Whether spill operations are needed."""
        return len(self.spill_points) > 0

    @property
    def spill_count(self) -> int:
        """Number of spill operations."""
        return len(self.spill_points)

    @property
    def reload_count(self) -> int:
        """Number of reload operations."""
        return len(self.reload_points)

    def get_buffer_for_tensor(self, tensor_name: str) -> int:
        """Get buffer ID for a tensor."""
        return self.tensor_to_buffer.get(tensor_name, -1)

    def get_allocation(self, tensor_name: str) -> Optional[TensorAllocation]:
        """Get allocation info for a tensor."""
        return self.tensor_allocations.get(tensor_name)

    def get_spill_points_after(self, node_idx: int) -> List[SpillPoint]:
        """Get spill points after a specific node."""
        return [sp for sp in self.spill_points if sp.after_node_idx == node_idx]

    def get_reload_points_before(self, node_idx: int) -> List[ReloadPoint]:
        """Get reload points before a specific node."""
        return [rp for rp in self.reload_points if rp.before_node_idx == node_idx]

    @property
    def spilled_tensors(self) -> Set[str]:
        """Get set of spilled tensor names."""
        return {
            alloc.tensor_name
            for alloc in self.tensor_allocations.values()
            if alloc.is_spilled
        }

    def get_max_reload_slots(self) -> int:
        """Get maximum number of concurrent reload slots needed.

        This is the maximum number of inputs to any single node that
        may be spilled and need to be reloaded.
        """
        from nnc_py.ir.context import CompileContext

        # Group reload points by node
        reloads_by_node: Dict[int, List[ReloadPoint]] = {}
        for rp in self.reload_points:
            reloads_by_node.setdefault(rp.before_node_idx, []).append(rp)

        # Find the maximum concurrent reloads
        max_slots = 0
        for node_reloads in reloads_by_node.values():
            max_slots = max(max_slots, len(node_reloads))

        return max_slots


class MemoryAllocationStrategy(ABC):
    """Abstract base class for memory allocation strategies.

    Each strategy implements a different algorithm for allocating
    memory buffers to tensors based on their liveness information.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @property
    def strategy_type(self) -> AllocationStrategy:
        """Return the strategy type enum."""
        # Default implementation - subclasses can override
        return AllocationStrategy.BASIC

    @abstractmethod
    def allocate(
        self,
        ctx: CompileContext,
        liveness_map: Dict[str, 'TensorLiveness'],
        max_memory: Optional[int] = None,
    ) -> MemoryAllocationPlan:
        """Execute the allocation algorithm.

        Args:
            ctx: Compilation context containing the graph
            liveness_map: Tensor liveness information from LivenessAnalysisPass
            max_memory: Optional memory limit (if None, unlimited)

        Returns:
            MemoryAllocationPlan with allocation results
        """
        pass

    def _validate_tensor_fits(self, tensor_size: int, max_memory: int) -> None:
        """Validate that a single tensor can fit in memory."""
        if max_memory != float("inf") and tensor_size > max_memory:
            raise ValueError(
                f"Tensor size ({tensor_size}) exceeds memory limit ({max_memory})"
            )


class StrategyRegistry:
    """Registry for memory allocation strategies.

    Allows runtime selection and registration of strategies.
    """
    _strategies: Dict[AllocationStrategy, Type[MemoryAllocationStrategy]] = {}
    _aliases: Dict[str, AllocationStrategy] = {}

    @classmethod
    def register(cls, strategy_cls: Type[MemoryAllocationStrategy]) -> None:
        """Register a strategy class."""
        # Create instance to get properties
        try:
            instance = strategy_cls()
            strategy_type = instance.strategy_type
            strategy_name = instance.name
        except Exception:
            # If we can't create instance, try to get from class attributes
            strategy_type = getattr(strategy_cls, 'strategy_type', AllocationStrategy.BASIC)
            strategy_name = getattr(strategy_cls, 'name', strategy_cls.__name__)

        cls._strategies[strategy_type] = strategy_cls
        cls._aliases[strategy_name] = strategy_type

    @classmethod
    def get(cls, strategy: Union[AllocationStrategy, str]) -> MemoryAllocationStrategy:
        """Get a strategy instance by enum or name."""
        if isinstance(strategy, str):
            if strategy in cls._aliases:
                strategy_type = cls._aliases[strategy]
            else:
                try:
                    strategy_type = AllocationStrategy(strategy)
                except ValueError:
                    raise ValueError(f"Unknown strategy: {strategy}")
        else:
            strategy_type = strategy

        strategy_cls = cls._strategies.get(strategy_type)
        if strategy_cls is None:
            raise ValueError(f"Strategy not registered: {strategy_type}")

        return strategy_cls()

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._aliases.keys())

    @classmethod
    def is_registered(cls, strategy: Union[AllocationStrategy, str]) -> bool:
        """Check if a strategy is registered."""
        if isinstance(strategy, str):
            return strategy in cls._aliases
        return strategy in cls._strategies

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies (mainly for testing)."""
        cls._strategies.clear()
        cls._aliases.clear()


# Import and register built-in strategies
def _register_default_strategies() -> None:
    """Import and register default strategies."""
    from nnc_py.passes.strategies.basic_allocator import BasicAllocator
    from nnc_py.passes.strategies.cost_aware_allocator import CostAwareAllocator

    for strategy_cls in (BasicAllocator, CostAwareAllocator):
        strategy = strategy_cls()
        if not StrategyRegistry.is_registered(strategy.strategy_type):
            StrategyRegistry.register(strategy_cls)


def get_allocation_plan(ctx: CompileContext) -> Optional[MemoryAllocationPlan]:
    """Get the memory allocation plan from context.

    Args:
        ctx: Compilation context

    Returns:
        MemoryAllocationPlan if available, None otherwise
    """
    return ctx.metadata.get("memory_allocation_plan")


def get_default_allocation_strategy(
    optimization_level: int,
) -> AllocationStrategy:
    """Get the default allocation strategy for an optimization level."""
    if optimization_level <= 0:
        return AllocationStrategy.BASIC
    return AllocationStrategy.COST_AWARE


def get_memory_strategy(ctx: CompileContext) -> Optional[MemoryAllocationStrategy]:
    """Get a memory strategy instance based on context configuration.

    Args:
        ctx: Compilation context

    Returns:
        MemoryAllocationStrategy instance
    """
    strategy_config = ctx.metadata.get("memory_strategy")

    if strategy_config is None:
        strategy_config = get_default_allocation_strategy(ctx.optimization_level)

    return StrategyRegistry.get(strategy_config)


_register_default_strategies()
