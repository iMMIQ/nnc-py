"""统一内存分配与交换算法

此 pass 统一处理内存分配和 spill 优化，基于使用点分析
实现精确的内存管理。

核心思想：
- 不再基于"存活区间重叠"决定 buffer 共享
- 而是基于"使用点"分析每个节点的实际内存需求
- 主动使用 spill/reload 来管理内存
- 确保峰值内存不超过限制
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

from nnc_py.ir.context import CompileContext
from nnc_py.ir.graph import Graph
from nnc_py.passes.base import PassBase
from nnc_py.passes.memory_plan import MemoryBuffer


# ========== 数据结构定义 ==========


@dataclass
class UsePointInfo:
    """张量的使用点信息

    记录张量在哪些节点被产生和使用，用于精确分析内存需求。
    """
    tensor_name: str
    produce_at: int              # 生产节点的索引
    use_at: List[int]            # 使用节点的索引列表（可能有多个）
    size: int                    # 张量大小（字节）
    is_input: bool = False       # 是否为模型输入
    is_output: bool = False      # 是否为模型输出
    is_constant: bool = False    # 是否为常量

    @property
    def last_use(self) -> int:
        """最后一次使用的节点索引"""
        return max(self.use_at) if self.use_at else self.produce_at

    @property
    def use_count(self) -> int:
        """使用次数"""
        return len(self.use_at)

    @property
    def lifetime_span(self) -> int:
        """从产生到最后使用的跨度"""
        return self.last_use - self.produce_at + 1

    def has_use_at(self, node_idx: int) -> bool:
        """检查是否在指定节点使用"""
        return node_idx in self.use_at

    def has_future_use_after(self, node_idx: int) -> bool:
        """检查在指定节点后是否有未来使用"""
        return any(u > node_idx for u in self.use_at)

    def distance_to_next_use(self, node_idx: int) -> int:
        """计算到下次使用的距离"""
        future_uses = [u for u in self.use_at if u > node_idx]
        return min(future_uses) - node_idx if future_uses else float('inf')


@dataclass
class ResidencyDecision:
    """单个张量的驻留决策

    定义张量在哪些节点区间驻留在快速内存，
    以及在哪些节点需要 spill/reload 操作。
    """
    tensor_name: str
    resident_ranges: List[Tuple[int, int]] = field(default_factory=list)
    spill_after: Set[int] = field(default_factory=set)
    reload_before: Set[int] = field(default_factory=set)
    slow_pool_offset: int = 0

    def is_resident_at(self, node_idx: int) -> bool:
        """检查张量是否在指定节点驻留"""
        for start, end in self.resident_ranges:
            if start <= node_idx <= end:
                return True
        return False

    def add_range(self, start: int, end: int) -> None:
        """添加驻留区间（可能会合并重叠区间）"""
        self.resident_ranges.append((start, end))
        self.resident_ranges.sort(key=lambda x: x[0])
        self._merge_overlapping_ranges()

    def _merge_overlapping_ranges(self) -> None:
        """合并重叠的驻留区间"""
        if not self.resident_ranges:
            return

        merged = [self.resident_ranges[0]]
        for current_start, current_end in self.resident_ranges[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end + 1:  # 重叠或相邻
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        self.resident_ranges = merged

    def truncate_after(self, node_idx: int) -> None:
        """在指定节点截断所有驻留区间"""
        new_ranges = []
        for start, end in self.resident_ranges:
            if start <= node_idx <= end:
                # 截断：保留 [start, node_idx]
                new_ranges.append((start, node_idx))
            elif end < node_idx:
                # 完全在截断点之前，保留
                new_ranges.append((start, end))
            # start > node_idx 的区间被丢弃

        self.resident_ranges = new_ranges


@dataclass
class NodeMemoryInfo:
    """单个节点的内存需求信息"""
    node_idx: int
    node_name: str
    tensors_resident: Set[str] = field(default_factory=set)
    tensors_produced: Set[str] = field(default_factory=set)
    fast_memory_usage: int = 0

    @property
    def resident_count(self) -> int:
        """驻留张量数量"""
        return len(self.tensors_resident)

    @property
    def produced_count(self) -> int:
        """产生张量数量"""
        return len(self.tensors_produced)


@dataclass
class SpillPoint:
    """Spill 操作点"""
    tensor_name: str
    after_node: str           # 在此节点执行后 spill
    after_node_idx: int       # 节点索引
    from_buffer_id: int       # 源 buffer ID
    from_fast_offset: int     # 源在快速内存中的偏移
    to_slow_offset: int       # 目标在慢速内存中的偏移
    size: int


@dataclass
class ReloadPoint:
    """Reload 操作点"""
    tensor_name: str
    before_node: str          # 在此节点执行前 reload
    before_node_idx: int      # 节点索引
    from_slow_offset: int     # 源在慢速内存中的偏移
    to_buffer_id: int         # 目标 buffer ID
    to_fast_offset: int       # 目标在快速内存中的偏移
    size: int


@dataclass
class UnifiedMemoryPlan:
    """统一的内存计划

    合并了内存分配和 spill 优化的完整计划。
    """
    # Buffer 信息
    buffers: List[MemoryBuffer] = field(default_factory=list)
    tensor_to_buffer: Dict[str, int] = field(default_factory=dict)

    # 节点内存信息
    node_memory: List[NodeMemoryInfo] = field(default_factory=list)

    # 驻留决策
    residency: Dict[str, ResidencyDecision] = field(default_factory=dict)

    # Spill/Reload 点
    spill_points: List[SpillPoint] = field(default_factory=list)
    reload_points: List[ReloadPoint] = field(default_factory=list)

    # 内存统计
    peak_memory: int = 0
    fast_memory_limit: int = 0
    slow_memory_size: int = 0

    # 张量信息
    use_points: Dict[str, UsePointInfo] = field(default_factory=dict)

    @property
    def has_spill(self) -> bool:
        """是否需要 spill 操作"""
        return len(self.spill_points) > 0

    @property
    def spill_count(self) -> int:
        """Spill 操作数量"""
        return len(self.spill_points)

    @property
    def reload_count(self) -> int:
        """Reload 操作数量"""
        return len(self.reload_points)

    @property
    def buffer_count(self) -> int:
        """Buffer 数量"""
        return len(self.buffers)

    @property
    def total_fast_memory(self) -> int:
        """快速内存总量"""
        if not self.buffers:
            return 0
        return max(b.offset + b.size for b in self.buffers)

    def get_resident_tensors_at(self, node_idx: int) -> Set[str]:
        """获取在指定节点驻留的张量"""
        result = set()
        for tensor_name, decision in self.residency.items():
            if decision.is_resident_at(node_idx):
                result.add(tensor_name)
        return result

    def get_memory_usage_at(self, node_idx: int) -> int:
        """获取指定节点的内存使用量"""
        if 0 <= node_idx < len(self.node_memory):
            return self.node_memory[node_idx].fast_memory_usage
        return 0

    def get_spill_points_after(self, node_idx: int) -> List[SpillPoint]:
        """获取在指定节点后的所有 spill 点"""
        return [sp for sp in self.spill_points if sp.after_node_idx == node_idx]

    def get_reload_points_before(self, node_idx: int) -> List[ReloadPoint]:
        """获取在指定节点前的所有 reload 点"""
        return [rp for rp in self.reload_points if rp.before_node_idx == node_idx]


# ========== 主算法类 ==========


class UnifiedMemoryPass(PassBase):
    """统一内存分配与交换算法

    算法流程：
    1. 使用点分析 - 收集每个张量的产生和使用节点
    2. 初始驻留决策 - 为每个张量决定初始驻留区间
    3. 峰值约束优化 - 迭代优化使峰值内存满足限制
    4. Buffer 分配 - 基于驻留区间分配共享 buffer
    5. 生成 spill/reload 点 - 生成代码生成所需信息
    """

    DEFAULT_ALIGNMENT = 16
    MAX_OPTIMIZATION_ITERATIONS = 100
    MIN_BUFFER_SIZE = 16

    @property
    def name(self) -> str:
        return "UnifiedMemory"

    def _execute(self, ctx: CompileContext) -> None:
        """执行统一内存规划"""
        graph = ctx.graph
        nodes = graph.topological_sort()
        node_index = {node.name: i for i, node in enumerate(nodes)}
        num_nodes = len(nodes)

        # 获取内存限制
        memory_limit = ctx.metadata.get("max_memory", float("inf"))

        self._log_start(ctx, memory_limit)

        # 阶段1: 使用点分析
        use_points = self._analyze_use_points(ctx, nodes, node_index)
        self._log_use_points(use_points, ctx.debug)

        # 验证最大张量能放入快速内存
        self._validate_single_tensor_fits(use_points, memory_limit)

        # 阶段2: 初始驻留决策
        residency = self._initial_residency(use_points, num_nodes)
        self._log_initial_residency(residency, use_points, nodes, ctx.debug)

        # 阶段3: 峰值约束优化
        if memory_limit != float("inf"):
            residency, node_memory = self._optimize_for_peak(
                nodes, use_points, residency, memory_limit, ctx.debug
            )
        else:
            node_memory = self._calculate_node_memory(nodes, use_points, residency)

        self._log_after_optimization(node_memory, memory_limit, ctx.debug)

        # 阶段4: Buffer 分配
        buffers, tensor_to_buffer = self._allocate_buffers(
            residency, use_points
        )
        self._log_buffer_allocation(buffers, tensor_to_buffer, use_points, ctx.debug)

        # 阶段5: 生成 spill/reload 点
        spill_points, reload_points = self._generate_spill_reload(
            nodes, node_index, use_points, residency, tensor_to_buffer, buffers
        )
        self._log_spill_reload_points(spill_points, reload_points, ctx.debug)

        # 计算慢速内存大小
        slow_memory_size = self._calculate_slow_memory_size(use_points, residency)

        # 创建统一内存计划
        plan = UnifiedMemoryPlan(
            buffers=buffers,
            tensor_to_buffer=tensor_to_buffer,
            node_memory=node_memory,
            residency=residency,
            spill_points=spill_points,
            reload_points=reload_points,
            peak_memory=max(nm.fast_memory_usage for nm in node_memory),
            fast_memory_limit=memory_limit,
            slow_memory_size=slow_memory_size,
            use_points=use_points,
        )

        # 存储到上下文
        ctx.metadata["unified_memory_plan"] = plan

        self._log_summary(ctx, plan)

    # ========== 阶段1: 使用点分析 ==========

    def _analyze_use_points(
        self,
        ctx: CompileContext,
        nodes: List,
        node_index: Dict[str, int],
    ) -> Dict[str, UsePointInfo]:
        """分析每个张量的使用点"""
        graph = ctx.graph
        use_points = {}

        for tensor_name in graph.tensors:
            # 获取张量信息
            tensor = graph.get_tensor(tensor_name)
            size = tensor.byte_size()

            if size < 0:
                # 未知大小，跳过
                if ctx.debug:
                    print(f"Warning: Tensor {tensor_name} has unknown size, skipping")
                continue

            # 生产节点
            producers = graph.get_producers(tensor_name)
            produce_at = node_index[producers[0].name] if producers else 0

            # 使用节点
            consumers = graph.get_consumers(tensor_name)
            use_at = [node_index[c.name] for c in consumers]

            # 张量类型
            is_input = tensor_name in graph.inputs
            is_output = tensor_name in graph.outputs
            is_constant = tensor_name in graph.constants

            use_points[tensor_name] = UsePointInfo(
                tensor_name=tensor_name,
                produce_at=produce_at,
                use_at=use_at,
                size=size,
                is_input=is_input,
                is_output=is_output,
                is_constant=is_constant,
            )

        return use_points

    def _validate_single_tensor_fits(
        self,
        use_points: Dict[str, UsePointInfo],
        memory_limit: int,
    ) -> None:
        """验证最大张量能放入快速内存"""
        if memory_limit == float("inf"):
            return

        max_tensor_size = max(
            (info.size for info in use_points.values() if not info.is_constant),
            default=0
        )

        if memory_limit < max_tensor_size:
            raise RuntimeError(
                f"Memory limit ({memory_limit}) is too small for the largest tensor "
                f"({max_tensor_size}). Each tensor must fit in fast memory when "
                f"being computed."
            )

    # ========== 阶段2: 初始驻留决策 ==========

    def _initial_residency(
        self,
        use_points: Dict[str, UsePointInfo],
        num_nodes: int,
    ) -> Dict[str, ResidencyDecision]:
        """生成初始驻留决策

        策略：
        - 输入张量：从开始到最后
        - 输出张量：从产生到最后
        - 中间张量：从产生到最后使用
        - 常量：不驻留（存储在 ROM）
        """
        residency = {}

        for tensor_name, info in use_points.items():
            if info.is_constant:
                continue

            if info.is_input:
                # 输入：从开始到最后
                resident_ranges = [(0, num_nodes - 1)]
            elif info.is_output:
                # 输出：从产生到最后
                resident_ranges = [(info.produce_at, num_nodes - 1)]
            else:
                # 中间张量：从产生到最后使用
                last_use = info.last_use
                resident_ranges = [(info.produce_at, last_use)]

            residency[tensor_name] = ResidencyDecision(
                tensor_name=tensor_name,
                resident_ranges=resident_ranges,
            )

        return residency

    # ========== 阶段3: 峰值约束优化 ==========

    def _optimize_for_peak(
        self,
        nodes: List,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        memory_limit: int,
        debug: bool = False,
    ) -> Tuple[Dict[str, ResidencyDecision], List[NodeMemoryInfo]]:
        """迭代优化使峰值内存满足限制

        算法：
        1. 计算当前峰值内存
        2. 如果满足限制，退出
        3. 找到峰值节点
        4. 选择可 spill 的张量
        5. 应用 spill
        6. 重复
        """
        for iteration in range(self.MAX_OPTIMIZATION_ITERATIONS):
            # 计算节点内存
            node_memory = self._calculate_node_memory(nodes, use_points, residency)

            # 找到峰值
            peak_memory = max(nm.fast_memory_usage for nm in node_memory)
            peak_nodes = [nm for nm in node_memory if nm.fast_memory_usage == peak_memory]

            if debug:
                print(f"Iteration {iteration}: peak_memory={peak_memory}, limit={memory_limit}")

            if peak_memory <= memory_limit:
                break

            # 尝试优化每个峰值节点
            improved = False
            for peak_node in peak_nodes:
                candidates = self._select_spill_candidates(
                    peak_node, use_points, residency, memory_limit
                )

                if candidates:
                    for tensor_name in candidates:
                        self._apply_spill(tensor_name, peak_node.node_idx, residency, use_points)
                    improved = True

            if not improved:
                # 无法进一步优化
                raise RuntimeError(
                    f"Cannot satisfy memory limit {memory_limit}. "
                    f"Minimum required: {peak_memory}"
                )

        return residency, node_memory

    def _calculate_node_memory(
        self,
        nodes: List,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
    ) -> List[NodeMemoryInfo]:
        """计算每个节点的内存需求"""
        node_memory = []

        for node_idx, node in enumerate(nodes):
            # 找出在此节点驻留的张量
            tensors_resident = set()
            for tensor_name, decision in residency.items():
                if decision.is_resident_at(node_idx):
                    tensors_resident.add(tensor_name)

            # 此节点产生的张量
            tensors_produced = set(node.outputs)

            # 计算内存使用
            usage = sum(
                use_points[t].size
                for t in tensors_resident
                if t in use_points
            )

            node_memory.append(NodeMemoryInfo(
                node_idx=node_idx,
                node_name=node.name,
                tensors_resident=tensors_resident,
                tensors_produced=tensors_produced,
                fast_memory_usage=usage,
            ))

        return node_memory

    def _select_spill_candidates(
        self,
        peak_node: NodeMemoryInfo,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        memory_limit: int,
    ) -> List[str]:
        """选择可 spill 的张量

        策略：
        1. 不能在当前节点立即使用的
        2. 优先选择大张量
        3. 优先选择距离下次使用较远的
        """
        candidates = []
        excess = peak_node.fast_memory_usage - memory_limit

        for tensor_name in peak_node.tensors_resident:
            info = use_points.get(tensor_name)
            if not info:
                continue

            # 跳过在当前节点使用的
            if peak_node.node_idx in info.use_at:
                continue

            # 计算优先级
            distance = info.distance_to_next_use(peak_node.node_idx)
            if distance == float('inf'):
                # 没有未来使用，优先级最高
                priority = info.size * 1000
            else:
                priority = info.size / max(distance, 1)

            candidates.append((tensor_name, priority, info.size))

        # 按优先级排序
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 选择直到释放足够内存
        selected = []
        released = 0
        for name, _, size in candidates:
            selected.append(name)
            released += size
            if released >= excess:
                break

        return selected

    def _apply_spill(
        self,
        tensor_name: str,
        spill_at_idx: int,
        residency: Dict[str, ResidencyDecision],
        use_points: Dict[str, UsePointInfo],
    ) -> None:
        """应用 spill 决策

        将张量的驻留区间在 spill_at_idx 处截断，
        后续使用前需要 reload。
        """
        decision = residency[tensor_name]
        info = use_points[tensor_name]

        # 截断驻留区间
        decision.truncate_after(spill_at_idx)

        # 记录 spill 点
        decision.spill_after.add(spill_at_idx)

        # 为未来的使用添加 reload 点和短期驻留
        for use_idx in info.use_at:
            if use_idx > spill_at_idx:
                decision.reload_before.add(use_idx)
                decision.resident_ranges.append((use_idx, use_idx))

        # 重新合并可能相邻的区间
        decision._merge_overlapping_ranges()

    # ========== 阶段4: Buffer 分配 ==========

    def _allocate_buffers(
        self,
        residency: Dict[str, ResidencyDecision],
        use_points: Dict[str, UsePointInfo],
    ) -> Tuple[List[MemoryBuffer], Dict[str, int]]:
        """基于驻留区间分配共享 buffer"""
        buffers: List[MemoryBuffer] = []
        tensor_to_buffer: Dict[str, int] = {}
        current_offset = 0
        buffer_id = 0

        # 按大小排序，优先处理大张量
        sorted_tensors = sorted(
            [(name, info) for name, info in use_points.items() if name in residency],
            key=lambda x: x[1].size,
            reverse=True
        )

        for tensor_name, info in sorted_tensors:
            decision = residency[tensor_name]
            tensor_size = info.size

            # 尝试复用现有 buffer
            assigned_buffer = None

            for buf in buffers:
                if self._can_reuse_buffer(buf, tensor_size, decision, residency):
                    assigned_buffer = buf
                    break

            if assigned_buffer is None:
                # 创建新 buffer
                aligned_offset = self._align(current_offset, self.DEFAULT_ALIGNMENT)
                buffer_size = max(self._align(tensor_size, self.DEFAULT_ALIGNMENT), self.MIN_BUFFER_SIZE)

                new_buffer = MemoryBuffer(
                    id=buffer_id,
                    offset=aligned_offset,
                    size=buffer_size,
                    alignment=self.DEFAULT_ALIGNMENT,
                )
                buffers.append(new_buffer)

                current_offset = aligned_offset + buffer_size
                buffer_id += 1
                assigned_buffer = new_buffer

            assigned_buffer.add_tensor(tensor_name)
            tensor_to_buffer[tensor_name] = assigned_buffer.id

        return buffers, tensor_to_buffer

    def _can_reuse_buffer(
        self,
        buffer: MemoryBuffer,
        tensor_size: int,
        decision: ResidencyDecision,
        residency: Dict[str, ResidencyDecision],
    ) -> bool:
        """检查 buffer 是否可复用"""
        if not buffer.can_hold(tensor_size, self.DEFAULT_ALIGNMENT):
            return False

        for existing_tensor in buffer.tensors:
            existing_decision = residency[existing_tensor]
            if self._residency_ranges_overlap(decision, existing_decision):
                return False

        return True

    def _residency_ranges_overlap(
        self,
        a: ResidencyDecision,
        b: ResidencyDecision,
    ) -> bool:
        """检查两个驻留决策是否重叠"""
        for a_start, a_end in a.resident_ranges:
            for b_start, b_end in b.resident_ranges:
                if not (a_end < b_start or b_end < a_start):
                    return True
        return False

    def _align(self, size: int, alignment: int) -> int:
        """对齐大小"""
        return ((size + alignment - 1) // alignment) * alignment

    # ========== 阶段5: 生成 spill/reload 点 ==========

    def _generate_spill_reload(
        self,
        nodes: List,
        node_index: Dict[str, int],
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
        tensor_to_buffer: Dict[str, int],
        buffers: List[MemoryBuffer],
    ) -> Tuple[List[SpillPoint], List[ReloadPoint]]:
        """生成 spill/reload 点"""
        spill_points = []
        reload_points = []

        # 分配慢速内存偏移
        slow_offset = 0
        for tensor_name, decision in residency.items():
            if decision.spill_after or decision.reload_before:
                info = use_points[tensor_name]
                decision.slow_pool_offset = slow_offset
                slow_offset += self._align(info.size, 16)

        # 生成 spill 点
        for tensor_name, decision in residency.items():
            if not decision.spill_after:
                continue

            info = use_points[tensor_name]
            buffer_id = tensor_to_buffer.get(tensor_name, 0)
            buf = self._get_buffer_by_id(buffers, buffer_id)
            if buf is None:
                continue

            for spill_after_idx in decision.spill_after:
                node = nodes[spill_after_idx]
                spill_points.append(SpillPoint(
                    tensor_name=tensor_name,
                    after_node=node.name,
                    after_node_idx=spill_after_idx,
                    from_buffer_id=buffer_id,
                    from_fast_offset=buf.offset,
                    to_slow_offset=decision.slow_pool_offset,
                    size=info.size,
                ))

        # 生成 reload 点
        for tensor_name, decision in residency.items():
            if not decision.reload_before:
                continue

            info = use_points[tensor_name]
            buffer_id = tensor_to_buffer.get(tensor_name, 0)
            buf = self._get_buffer_by_id(buffers, buffer_id)
            if buf is None:
                continue

            for reload_before_idx in decision.reload_before:
                node = nodes[reload_before_idx]
                reload_points.append(ReloadPoint(
                    tensor_name=tensor_name,
                    before_node=node.name,
                    before_node_idx=reload_before_idx,
                    from_slow_offset=decision.slow_pool_offset,
                    to_buffer_id=buffer_id,
                    to_fast_offset=buf.offset,
                    size=info.size,
                ))

        # 按执行顺序排序
        spill_points.sort(key=lambda p: p.after_node_idx)
        reload_points.sort(key=lambda p: p.before_node_idx)

        return spill_points, reload_points

    def _calculate_slow_memory_size(
        self,
        use_points: Dict[str, UsePointInfo],
        residency: Dict[str, ResidencyDecision],
    ) -> int:
        """计算慢速内存总大小"""
        total = 0
        for tensor_name, decision in residency.items():
            if decision.spill_after or decision.reload_before:
                info = use_points[tensor_name]
                total += self._align(info.size, 16)
        return total

    # ========== 辅助方法 ==========

    def _get_buffer_by_id(
        self,
        buffers: List[MemoryBuffer],
        buffer_id: int,
    ) -> Optional[MemoryBuffer]:
        """根据 ID 获取 buffer"""
        for buf in buffers:
            if buf.id == buffer_id:
                return buf
        return None

    # ========== 日志方法 ==========

    def _log_start(self, ctx: CompileContext, memory_limit: int) -> None:
        if not ctx.debug:
            return
        print(f"\n{'='*80}")
        print(f"Unified Memory Planning")
        print(f"{'='*80}")
        if memory_limit == float("inf"):
            print(f"Memory limit: unlimited")
        else:
            print(f"Memory limit: {memory_limit} bytes ({memory_limit / 1024:.2f} KB)")

    def _log_use_points(
        self,
        use_points: Dict[str, UsePointInfo],
        debug: bool = False,
    ) -> None:
        if not debug:
            return

        print(f"\n--- Use Points Analysis ---")
        print(f"{'Tensor':<20} {'Produce':>8} {'Uses':>20} {'Size':>10} {'Type':>10}")
        print(f"{'-'*70}")

        for name, info in sorted(use_points.items(), key=lambda x: x[1].produce_at):
            uses_str = ",".join(map(str, info.use_at)) if info.use_at else "-"
            size_str = f"{info.size / 1024:.2f}K" if info.size >= 1024 else f"{info.size}"
            type_str = []
            if info.is_input:
                type_str.append("input")
            if info.is_output:
                type_str.append("output")
            if info.is_constant:
                type_str.append("const")
            if not type_str:
                type_str.append("intermediate")

            print(f"{name:<20} {info.produce_at:>8} {uses_str:>20} {size_str:>10} {'/'.join(type_str):>10}")

    def _log_initial_residency(
        self,
        residency: Dict[str, ResidencyDecision],
        use_points: Dict[str, UsePointInfo],
        nodes: List,
        debug: bool = False,
    ) -> None:
        if not debug:
            return

        print(f"\n--- Initial Residency Decisions ---")
        print(f"{'Tensor':<20} {'Residency Ranges':>30} {'Size':>10}")
        print(f"{'-'*60}")

        for name, decision in sorted(residency.items()):
            info = use_points[name]
            size_str = f"{info.size / 1024:.2f}K" if info.size >= 1024 else f"{info.size}"
            ranges_str = str(decision.resident_ranges)
            print(f"{name:<20} {ranges_str:>30} {size_str:>10}")

    def _log_after_optimization(
        self,
        node_memory: List[NodeMemoryInfo],
        memory_limit: int,
        debug: bool = False,
    ) -> None:
        if not debug:
            return

        print(f"\n--- Node Memory Usage After Optimization ---")
        print(f"{'Node':<20} {'Idx':>4} {'Resident':>8} {'Produced':>8} {'Usage':>10}")
        print(f"{'-'*60}")

        for nm in node_memory:
            usage_str = f"{nm.fast_memory_usage / 1024:.2f}K" if nm.fast_memory_usage >= 1024 else f"{nm.fast_memory_usage}"
            print(f"{nm.node_name:<20} {nm.node_idx:>4} {nm.resident_count:>8} {nm.produced_count:>8} {usage_str:>10}")

        peak = max(nm.fast_memory_usage for nm in node_memory)
        peak_str = f"{peak / 1024:.2f}K" if peak >= 1024 else f"{peak}"
        limit_str = "unlimited" if memory_limit == float("inf") else f"{memory_limit / 1024:.2f}K"
        print(f"\nPeak memory: {peak_str}, Limit: {limit_str}")

    def _log_buffer_allocation(
        self,
        buffers: List[MemoryBuffer],
        tensor_to_buffer: Dict[str, int],
        use_points: Dict[str, UsePointInfo],
        debug: bool = False,
    ) -> None:
        if not debug:
            return

        print(f"\n--- Buffer Allocation ---")
        print(f"Total buffers: {len(buffers)}")

        for buf in buffers:
            size_str = f"{buf.size / 1024:.2f}K" if buf.size >= 1024 else f"{buf.size}"
            print(f"  Buffer #{buf.id}: offset={buf.offset}, size={size_str}, alignment={buf.alignment}")
            for tensor_name in buf.tensors:
                info = use_points.get(tensor_name)
                if info:
                    tsize_str = f"{info.size / 1024:.2f}K" if info.size >= 1024 else f"{info.size}"
                    print(f"    {tensor_name}: {tsize_str}")

    def _log_spill_reload_points(
        self,
        spill_points: List[SpillPoint],
        reload_points: List[ReloadPoint],
        debug: bool = False,
    ) -> None:
        if not debug:
            return

        print(f"\n--- Spill/Reload Points ---")
        print(f"Spill points: {len(spill_points)}")
        for sp in spill_points:
            size_str = f"{sp.size / 1024:.2f}K" if sp.size >= 1024 else f"{sp.size}"
            print(f"  {sp.tensor_name}: spill after {sp.after_node} (idx {sp.after_node_idx}), size={size_str}")

        print(f"Reload points: {len(reload_points)}")
        for rp in reload_points:
            size_str = f"{rp.size / 1024:.2f}K" if rp.size >= 1024 else f"{rp.size}"
            print(f"  {rp.tensor_name}: reload before {rp.before_node} (idx {rp.before_node_idx}), size={size_str}")

    def _log_summary(self, ctx: CompileContext, plan: UnifiedMemoryPlan) -> None:
        if not ctx.debug:
            return
        print(f"\n{'='*80}")
        print(f"Unified Memory Plan Summary")
        print(f"{'='*80}")
        print(f"Peak memory: {plan.peak_memory} bytes ({plan.peak_memory / 1024:.2f} KB)")
        if plan.fast_memory_limit != float("inf"):
            print(f"Fast memory limit: {plan.fast_memory_limit} bytes ({plan.fast_memory_limit / 1024:.2f} KB)")
        else:
            print(f"Fast memory limit: unlimited")
        print(f"Slow memory used: {plan.slow_memory_size} bytes ({plan.slow_memory_size / 1024:.2f} KB)")
        print(f"Buffers: {plan.buffer_count}")
        print(f"Spill points: {plan.spill_count}")
        print(f"Reload points: {plan.reload_count}")
        print(f"{'='*80}\n")


# ========== 辅助函数 ==========


def get_unified_memory_plan(ctx: CompileContext) -> UnifiedMemoryPlan:
    """获取统一内存计划

    Args:
        ctx: 编译上下文

    Returns:
        UnifiedMemoryPlan 对象

    Raises:
        RuntimeError: 如果 UnifiedMemoryPass 未运行
    """
    plan = ctx.metadata.get("unified_memory_plan")
    if plan is None:
        raise RuntimeError("UnifiedMemoryPass must be run first")
    return plan
