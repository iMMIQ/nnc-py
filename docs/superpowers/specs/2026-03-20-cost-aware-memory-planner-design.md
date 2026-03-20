# Cost-Aware Memory Planner Design

## Goal

为 `nnc-py` 设计一套新的 memory planner，在用户给定 `fast memory` 上限时，优先减少 `fast memory` 与 `slow memory` 之间的数据交换总字节数，而不是单纯最小化 fast pool 占用。

该设计保留当前保守分配算法作为 `O0` 基线，同时为 `O1/O2/O3` 引入新的 `cost-aware` 分配策略，服务于后续性能优化与 benchmark 对比。

## Problem Statement

当前工程的 `basic` allocator 使用顺序追加式分配：

- tensor 分配后不会因为生命周期结束而回收复用
- 在没有 `max_memory` 约束时，容易生成远大于 live-set 峰值的 fast pool
- 在有 `max_memory` 约束时，spill 策略是“内存不够就整体清空”，目标不是最小化交换代价

这与当前目标不一致。

用户关注的重点是：

- `fast memory` 是外部输入约束，不是 planner 自由压缩的目标
- planner 应尽量减少 slow-memory 交换
- 优化目标应以 `spill/reload` 总字节数为核心，而不是单纯减少分配峰值

## Scope

### In Scope

- 保留现有 `basic` allocator
- 新增 `cost_aware` allocator
- `O0` 默认使用 `basic`
- `O1/O2/O3` 默认使用 `cost_aware`
- 复用现有 `LivenessAnalysisPass`，并补充使用点信息
- 在统一 `MemoryAllocationPlan` 中表达：
  - fast buffer 布局
  - spill points
  - reload points
  - slow memory 占用
  - transfer 统计信息
- 优化目标聚焦：
  - `total_transfer_bytes = spill_bytes + reload_bytes`

### Out of Scope

- ILP / exact optimization
- profile-guided memory planning
- 多 fast pool / 多 slow pool / NUMA 建模
- 改写 codegen 总体接口
- 运行时异步搬运或 overlap 调度

## Design Principles

- `O0` 保持保守、稳定、易理解
- `O1+` 的 planner 应围绕真实性能目标优化，而不是为了“内存越小越好”
- planner 决策应建立在未来使用价值上，而不是仅看当前是否能塞下
- 输出结构保持统一，避免 codegen 与具体 allocator 强耦合
- 第一版优先可验证、可扩展，不追求全局最优

## Proposed Architecture

### Strategy Split By Optimization Level

- `O0`
  - 使用现有 `basic` allocator
  - 保持当前行为，作为保守基线
- `O1/O2/O3`
  - 默认使用新的 `cost_aware` allocator
  - 在给定 `max_memory` 下，优先减少 future spill/reload 总字节数

建议保留统一入口：

- `MemoryPlanningPassV2`
  - 根据 `ctx.optimization_level` 或显式 `memory_strategy` 选择 allocator

这样现有 pass 管线不需要拆分，只改变默认策略选择规则。

### Unified Plan Output

新的 allocator 仍输出 `MemoryAllocationPlan`，但补充以下统计字段：

- `spill_bytes`
- `reload_bytes`
- `total_transfer_bytes`

这些字段用于：

- 测试验证
- benchmark 对比
- 后续改进 eviction 策略时的回归基线

### Liveness Extension

当前 `TensorLiveness` 只包含：

- `live_start`
- `live_end`

新的 allocator 还需要：

- `use_positions`
  - tensor 被消费的 node index 列表
- `next_use_after(node_idx)`
  - 给定当前位置后，tensor 的下一次使用位置
- `remaining_uses`
  - 当前位置之后还有多少次使用

推荐做法：

- 在 `LivenessAnalysisPass` 中补充这些字段
- 或在 metadata 中额外写入 `tensor_use_info`

为了减少接口分裂，建议直接扩充 liveness 结果。

## Cost-Aware Allocation Model

### Planner View

第一版将 `fast memory` 视为受限缓存：

- fast memory：高性能、容量受限
- slow memory：容量更大、访问更慢

planner 目标不是最小化 fast memory 已分配字节，而是：

1. 在 `max_memory` 约束内为热点 tensor 保留 fast residency
2. 对必须换出的 tensor，尽量选择未来价值最低的一组
3. 降低 `spill_bytes + reload_bytes`

### Event-Driven Walk

allocator 按拓扑序扫描 node。

对每个 node：

1. 处理即将执行的输入
   - 若输入 tensor 已在 fast memory，直接使用
   - 若不在 fast memory，需要 reload
   - 若 reload 前空间不足，先执行 eviction 决策

2. 处理 node 产出
   - 若输出后续还会被使用，尝试驻留 fast memory
   - 若空间不足，则触发 eviction
   - 若输出不会再被使用，可跳过驻留或立即释放

3. 处理生命周期结束
   - 对于在当前 node 完成最后一次使用的 tensor，立即释放 fast slot

这样 planner 可以显式利用“未来还会不会再用”和“多久后再用”的信息。

## Eviction Heuristic

### Objective Approximation

第一版不做全局最优搜索，采用启发式 eviction。

核心偏好：

- 优先保留“很快还会再用”的 tensor
- 优先保留“体积大、搬运成本高”的 tensor
- 优先换出“距离下次使用很远”或“不再使用”的 tensor

建议第一版使用简单分数：

`evict_score = next_use_distance / size`

约定：

- `next_use_distance` 越大，越适合被换出
- `size` 越大，越不适合频繁搬运
- 若 tensor 未来不再使用，则直接视为最优先释放对象

当需要腾空间时：

- 从当前 fast-resident tensors 中按 `evict_score` 从高到低选取
- 直到释放空间满足当前 reload / allocation 需求

### Why This Heuristic

它不是理论最优，但与目标函数方向一致：

- 同样大小的 tensor，越晚再用越应该被换出
- 同样未来距离的 tensor，越大越应该尽量留在 fast

对当前项目阶段，这比“统一 spill all”更符合性能目标，也更容易验证。

## Memory Layout Semantics

### Fast Memory

对于 `cost_aware` allocator：

- fast memory 中的 slot 必须支持回收复用
- planner 不能继续使用 append-only offset 模型
- 需要显式维护 free regions 或 free slots

第一版建议：

- 采用 aligned first-fit free-list
- tensor 驻留 fast 时占据一个连续区间
- tensor 生命周期结束或被 spill 后，该区间归还 free-list

这能在不引入复杂颜色分配器的前提下提供稳定行为。

### Slow Memory

slow memory 第一版保持简单：

- spill 后写入 slow pool 的连续区域
- 不做 slow-memory compaction
- 以追加方式为 spilled tensor 分配 slow offset

这样可以把设计复杂度集中在“谁被换出”，而不是“slow pool 如何优化”。

## Codegen Contract

新 allocator 不应改变 codegen 的总体消费方式。

codegen 继续依赖统一的 `MemoryAllocationPlan`：

- `tensor_allocations`
- `buffers`
- `spill_points`
- `reload_points`

需要新增但不破坏兼容性的仅是统计字段。

这保证：

- 旧实现仍可工作
- `basic` 与 `cost_aware` 可共存
- benchmark 与测试可以在同一 schema 下对比

## Testing Strategy

### Unit Tests

新增测试应覆盖：

- `O0` 默认仍为 `basic`
- `O1/O2/O3` 默认切到 `cost_aware`
- 无 `max_memory` 时，新 allocator 至少保持正确性与稳定分配
- 有 `max_memory` 时，新 allocator 的 `total_transfer_bytes` 低于或不高于 `basic`
- 生命周期结束后 fast slot 被复用
- eviction 优先级符合“未来远 + size 小更容易换出”的预期

### Integration Tests

需要覆盖小图场景：

- 链式图
- 带分支与 join 的图
- 残差结构
- 多输入节点

重点断言：

- spill/reload 点的时序正确
- tensor 在被消费前一定在 fast memory
- graph 输出与参考结果一致

### Benchmark Validation

在 benchmark 上优先看：

- 在相同 `max_memory` 下，`total_transfer_bytes` 是否下降
- spill/reload 次数是否下降
- ResNet18 端到端 latency 是否改善

若未设置 `max_memory`：

- 关注 fast memory 是否仍保持正确且比保守算法更合理
- 但这不是第一目标

## Rollout Plan

1. 扩充 liveness/use info
2. 新增 `cost_aware` allocator
3. 扩展 `MemoryAllocationPlan` 统计字段
4. 切换 optimization-level 默认策略
5. 增补单测与集成测试
6. 使用 benchmark 验证 `total_transfer_bytes` 与 latency

## Risks

### Heuristic Suboptimality

启发式不保证全局最优，可能在某些图结构上做出局部次优决策。

缓解方式：

- 把 transfer 统计写入 plan
- 通过测试和 benchmark 做回归比较
- 后续允许替换 score 公式而不改 plan 接口

### Fragmentation

free-list 复用可能带来 fast memory 碎片。

缓解方式：

- 使用 aligned first-fit 作为第一版
- 在 plan 中保留 buffer/offset 明确表示
- 若后续有必要，再引入 best-fit 或区间合并优化

### Spill Correctness Complexity

当 tensor 被频繁换入换出时，spill/reload 时序错误会直接破坏结果正确性。

缓解方式：

- 先在小图上构造强约束测试
- 保持 codegen 仍只消费统一 plan
- 将 `total_transfer_bytes` 与 spill/reload 点显式暴露，便于调试

## Success Criteria

该设计完成后，应满足：

- `O0` 保持现有保守行为
- `O1/O2/O3` 默认启用新的 `cost_aware` 策略
- 在给定 `max_memory` 约束下，planner 明确以降低 `total_transfer_bytes` 为目标
- 统一 plan 接口不被破坏
- ResNet18 等 benchmark 可以用于比较新旧 allocator 的 slow-memory 交换与端到端性能差异
