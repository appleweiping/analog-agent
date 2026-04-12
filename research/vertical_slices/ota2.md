# OTA2 Vertical Slice v1

`ota2_v1` 是当前论文主路径的固定 vertical slice。它把 `two_stage_ota` 从“能跑的例子”收束成“可复现、可验收、可统计、可写论文”的标准路径。

## Version

- Frozen version: `ota2_v1`
- Freeze rule: 论文主路径默认保持 `v1` 不变；后续若要改变对象语义、测量定义、默认 fidelity 或模型绑定，必须创建 `v2`，或保证 `v1` 行为不变。

## Object Chain

完整对象链固定为：

`DesignTask -> CandidateRecord -> SimulationBundle -> VerificationResult -> CalibrationFeedback / PlannerFeedback -> EpisodeMemoryRecord`

标准入口：

- Acceptance: `libs.vertical_slices.ota2.run_ota_acceptance()`
- Experiment suite: `libs.vertical_slices.ota2.run_ota_experiment_suite()`

这两个入口都会走现有主干，而不是测试专用路径。

## Canonical Paths

- Benchmark config: `configs/benchmarks/ota2.yaml`
- Netlist template: `templates/netlist/ota2/v1/ota2_demonstrator_truth.spice.tpl`
- Quick truth testbench: `templates/testbench/ota2/v1/quick_truth.yaml`
- Focused truth testbench: `templates/testbench/ota2/v1/focused_truth.yaml`
- Measurement contract: `templates/testbench/ota2/v1/measurement_contract.yaml`

## Design Variables

OTA v1 固定六个变量，全部来自第二层正式 `DesignTask.design_space`，不允许隐式魔法参数：

- `w_in`: 输入差分对宽度，单位 `m`
- `l_in`: 输入差分对长度，单位 `m`
- `w_tail`: 第二级 / 尾电流代理器件宽度，单位 `m`
- `l_tail`: 第二级 / 尾电流代理器件长度，单位 `m`
- `ibias`: OTA 总偏置电流代理量，单位 `A`
- `cc`: Miller 补偿电容，单位 `F`

## Measurement Contract

OTA v1 固定四项核心测量目标，且必须走第五层正式 measurement schema：

- `dc_gain_db`
- `gbw_hz`
- `phase_margin_deg`
- `power_w`

其中：

- `dc_gain_db` / `gbw_hz` / `phase_margin_deg` 由 `ac` 提取
- `power_w` 由 `op` 的供电功耗聚合提取

任何提取失败都必须通过 `MeasurementResult.status` 和 `MeasurementFailureReason` 结构化表达，不允许用 `NaN`、空值或默默回退。

## Fidelity Policy

OTA v1 的默认 fidelity 制度固定如下：

- `quick_truth = op + ac`
- `focused_truth = op + ac + minimal tran`

默认 escalation 语义：

- 默认主循环使用 `quick_truth`
- nominal feasible 候选升级到 `focused_truth`
- near-feasible 边界候选在绝对最小 margin 不超过 `0.08` 时升级到 `focused_truth`
- trust / measurement 异常但高价值候选允许升级到 `focused_truth`
- `focused_truth` 默认 batch limit 为 `1`

## Model Binding

OTA v1 默认模型绑定为：

- `model_type = builtin`
- `truth_level = demonstrator_truth`
- `default_builtin_model = builtin_demo_ota2_small_signal_v1`

这意味着当前 OTA v1 的物理意义是“真实 SPICE 参与闭环的 demonstrator truth”，而不是工业级 PDK 可信结果。

外部模型卡升级方式：

- 在 `configs/simulator/ngspice.yaml` 中配置 `model_type = external`
- 提供 `external_model_card_path`
- 主路径会自动把 truth level 切换到 `configured_truth`

## Acceptance Usage

```python
from libs.vertical_slices.ota2 import run_ota_acceptance

result = run_ota_acceptance()
```

输出对象固定为 `SystemAcceptanceResult`，内部包含：

- `CrossLayerTrace`
- `ArtifactTrace`
- `StatsAggregationResult`

## Experiment Usage

```python
from libs.vertical_slices.ota2 import run_ota_experiment_suite

suite = run_ota_experiment_suite()
```

标准对照模式固定为：

- `full_simulation_baseline`
- `no_world_model_baseline`
- `full_system`

同时自动支持 stats 导出。

## Current Scope

OTA v1 当前明确只覆盖：

- `two_stage_ota`
- fixed-topology sizing
- `ngspice`
- `demonstrator_truth` 默认模型绑定
- `quick_truth / focused_truth`

它不覆盖：

- topology search
- LDO / bandgap / comparator 主路径
- full PVT
- Monte Carlo
- Xyce / Spectre consistency

这条路径的价值在于：它是当前系统中最稳定、最适合作为论文方法与实验主线的第一条真实物理闭环路径。
