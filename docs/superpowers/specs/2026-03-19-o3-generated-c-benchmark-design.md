# O3 Generated C Benchmark Design

## Goal

为 `nnc-py` 增加一套面向主机环境的 benchmark 框架，用于评估不同算法版本在 `O3` 下生成的 C 代码的运行性能和静态内存占用。

该框架不评估编译时间，也不以 pass 内部统计作为主输出。核心目标是为后续 O3 算法优化提供稳定、可重复、可对比的基准结果。

## Scope

### In Scope

- 主机环境下运行生成的 x86 C 代码
- 固定 `opt_level=3`
- 端到端 ONNX 模型 benchmark
- 第一版以 `ResNet18` 为主 benchmark case
- 指标包括：
  - `batch=1` 延迟
  - 多 batch 吞吐
  - 静态内存占用
  - 生成产物大小
- 支持比较不同算法版本或不同 commit 的 O3 结果

### Out of Scope

- O0/O1/O2 对比
- 编译时长分析
- 嵌入式设备或交叉编译环境
- `perf`、`valgrind`、RSS、cache miss 等系统级 profiling
- pass 内部细粒度 explainability 指标

## Existing Project Context

当前工程已经具备生成代码和运行正确性验证所需的大部分基础设施：

- `src/nnc_py/compiler.py` 提供编译主流程
- `src/nnc_py/codegen/x86_backend.py` 生成 x86 C artifacts
- `models/resnet18.onnx` 可作为首个端到端 benchmark 模型
- `tests/test_snapshots_resnet18.py` 已验证 ResNet18 的代码生成与运行正确性

当前缺失的是专门面向 benchmark 的：

- case 定义层
- benchmark runner
- 结果 schema
- 基线对比机制
- 可重复的结果落盘和报告输出

## Design Principles

- 黑盒评测生成代码，不把 benchmark 与编译器内部实现强耦合
- 指标稳定优先于信息极大化
- 第一期先覆盖最关键路径：单模型、固定 O3、主机环境
- benchmark 与测试解耦，不放入 `tests/`
- 结果必须结构化，方便后续比较不同算法版本

## Proposed Architecture

新增独立 `benchmarks/` 子系统，结构如下：

```text
benchmarks/
├── __init__.py
├── cases.py
├── harness.py
├── runner_gen.py
├── metrics.py
├── compare.py
└── results/
```

### `benchmarks/cases.py`

职责：

- 定义 benchmark case 配置
- 第一版提供 `resnet18` case
- 为每个 case 记录：
  - ONNX 路径
  - 默认 workload batch sizes
  - warmup 次数
  - measure 次数
  - 输入 tensor 生成策略

建议的数据结构：

```python
@dataclass
class BenchmarkCase:
    name: str
    model_path: Path
    workload_batch_sizes: list[int]
    warmup_iterations: int
    measure_iterations: int
```

### `benchmarks/harness.py`

职责：

- 命令行入口
- 编排 compile -> build -> run -> collect -> report 全流程
- 写出 benchmark 结果 JSON
- 在提供基线结果时触发结果对比

该模块不修改现有 `nnc` CLI，第一版采用独立脚本入口：

```bash
python -m benchmarks.harness --model resnet18 --batch-sizes 1,8,16,32
python -m benchmarks.harness --model resnet18 --baseline-result path/to/baseline.json
```

### `benchmarks/runner_gen.py`

职责：

- 生成专用 benchmark runner C 文件
- 不复用当前偏调试/示例性质的 `_test_runner`
- 为每个 workload batch size 生成：
  - 输入初始化
  - warmup 循环
  - 多轮计时
  - 延迟与吞吐统计输出

runner 行为要求：

- 使用稳定、单调递增时钟
- 在一次进程运行内完成 warmup 和 measure
- 输出机器可解析 JSON，避免 fragile 文本解析
- 输入初始化固定，可复现

第一版中的 `batch size` 定义为 workload batch，而不是运行时改变模型输入 shape 的真实张量 batch。
对于仓库中固定 shape 的 ONNX 模型（如 `resnet18.onnx`），runner 通过连续执行 `nnc_run()` N 次来表示一个 workload batch，并将吞吐统计为 `N / elapsed_time`。
其中：

- `batch=1` 表示单次推理延迟
- `batch=8/16/32` 表示单个测量窗口内顺序执行 8/16/32 次推理，用于观察吞吐变化

### `benchmarks/metrics.py`

职责：

- 提取静态内存与产物大小指标
- 解析生成目录中的：
  - `tensors.c`
  - `constants.bin`
  - benchmark executable

第一版内存定义：

- `fast_memory_bytes`:
  - dual pool：从 `tensors.c` 中的 `NNC_FAST_MEMORY_SIZE` 提取
  - single pool：从 `tensors.c` 中的 `NNC_MEMORY_SIZE` 提取（此时 `slow_memory_bytes=0`）
- `slow_memory_bytes`:
  - dual pool：从 `tensors.c` 中的 `NNC_SLOW_MEMORY_SIZE` 提取
  - single pool：为 `0`
- `constants_bytes`: `constants.bin` 文件大小
- `binary_size_bytes`: benchmark 可执行文件大小
- `total_static_bytes = fast + slow + constants`

这是“生成 C 代码静态占用”的定义，不包含进程级 RSS。

### `benchmarks/compare.py`

职责：

- 对比 baseline 与 candidate 两份 benchmark 结果
- 输出结构化 diff
- 给出每个 batch 下的性能变化和内存变化

对比入口形式（推荐使用 harness 生成 diff）：

```bash
python -m benchmarks.harness --model resnet18 --baseline-result baseline.json --output candidate.json
# 生成 candidate.diff.json
```

如需自定义对比流程，可在 Python 中直接调用 `benchmarks.compare.compare_results()`。

## Benchmark Flow

一次标准 benchmark 的执行流程如下：

1. 读取 benchmark case
2. 使用 `Compiler(target="x86", opt_level=3)` 编译目标 ONNX 模型
3. 生成 benchmark runner C 文件
4. 使用 `gcc` 构建生成代码与 runner
5. 针对每个 workload batch size：
   - 生成输入
   - 执行 warmup
   - 执行多轮测量，每轮连续调用 `nnc_run()` N 次
   - 记录 latency 和 throughput
6. 解析静态内存与产物大小
7. 写出结果 JSON
8. 若提供 baseline result，则生成对比报告

## Metrics Schema

每次 benchmark 输出一份 JSON，建议结构如下：

```json
{
  "model": "resnet18",
  "commit": "abcdef1",
  "benchmark_date": "2026-03-19T12:00:00Z",
  "compiler_config": {
    "target": "x86",
    "opt_level": 3
  },
  "build_config": {
    "cc": "gcc",
    "cflags": ["-O3", "-std=c11"]
  },
  "runs": [
    {
      "batch_size": 1,
      "warmup_iterations": 5,
      "measure_iterations": 20,
      "latency_ms_mean": 0.0,
      "latency_ms_p50": 0.0,
      "latency_ms_p95": 0.0,
      "throughput_samples_per_sec": 0.0
    }
  ],
  "memory": {
    "fast_memory_bytes": 0,
    "slow_memory_bytes": 0,
    "constants_bytes": 0,
    "binary_size_bytes": 0,
    "total_static_bytes": 0
  },
  "artifacts": {
    "output_dir": "benchmarks/build/.../resnet18-.../",
    "executable_path": "benchmarks/build/.../resnet18-.../resnet18_bench"
  }
}
```

对比输出在此基础上增加：

- `baseline_commit`
- `candidate_commit`
- `runs[].latency_delta_pct`
- `runs[].throughput_delta_pct`
- `memory.total_static_bytes_delta`

## Case Definition

第一版仅提供：

- `resnet18`

默认配置建议：

- workload batch sizes: `[1, 8, 16, 32]`
- warmup iterations: `5`
- measure iterations: `20`

选择理由：

- `batch=1` 对应单次推理延迟
- 多 workload batch 用于观察顺序推理场景下的吞吐变化
- `ResNet18` 已存在于仓库中，具备端到端代表性

## Error Handling

框架需要对以下错误给出明确失败信息：

- ONNX 模型不存在
- 编译失败
- C 构建失败
- runner 输出格式非法
- 内存指标解析失败
- baseline 结果 schema 不兼容

错误处理原则：

- 失败时保留中间产物目录，方便排查
- 错误消息必须指出阶段：`compile` / `build` / `run` / `parse`
- 结果不完整时不要静默回退为 0

## Verification Strategy

### Unit Tests

- `metrics.py` 中的静态内存提取逻辑
- 结果 schema 序列化
- compare 差异计算

### Integration Tests

- 跑一个缩小版 benchmark 流程，验证能产出完整 JSON
- 对 `resnet18` 至少执行单 batch smoke test

### Stability Measures

- runner 内置 warmup
- 多轮重复测量
- 报告 `p50` 和 `p95`，不只看均值
- 固定输入初始化方式

## Non-Goals for V1

以下内容明确推迟到后续版本：

- 多模型 benchmark suite 扩展
- 系统级 RSS 采集
- `perf stat` 集成
- 自动 checkout baseline commit 并在同一命令中完成双版本构建
- benchmark 结果可视化 dashboard
- 合入现有 `nnc` CLI

## Open Decisions Resolved

- 主目标：评估生成 C 代码的性能和静态内存占用
- 运行环境：当前主机环境
- 优化等级：固定 `O3`
- 评测对象：端到端模型
- 首个模型：`ResNet18`
- 比较方式：不同算法版本的 O3 结果对比
- 接口形式：独立 benchmark 脚本，不进入现有 CLI

## Implementation Readiness

该设计已经具备进入 implementation plan 的条件。

推荐先实现最小可用版本：

1. `cases.py` 中定义 `resnet18`
2. `runner_gen.py` 生成 JSON 输出 benchmark runner
3. `harness.py` 跑通 compile/build/run
4. `metrics.py` 提取静态内存与产物大小
5. `compare.py` 输出 baseline diff

完成后即可支持后续 O3 算法优化迭代中的回归与收益评估。
