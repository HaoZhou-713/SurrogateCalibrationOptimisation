# 自建仿真验证复现

权威来源是 `Surrogate_models_for_small-size_datasets/CaseStudy_battery_sim.ipynb`。本工作流保留其中最终三框架流程与原始参数，提供保存结果重绘和重新训练两个入口。原始算法没有重新设计；仅整理执行顺序、文件路径、导出和命令行入口。

## 运行

在新仓库目录执行：

```bash
python workflows/inhouse/run.py figures
python workflows/inhouse/run.py train --smoke
python workflows/inhouse/run.py train
```

也可用脚本的绝对路径从任意工作目录运行。`--output-dir PATH` 指定输出目录。默认输出彼此隔离：

- 保存结果重绘：`outputs/inhouse/figures/`
- 缩减训练验证：`outputs/inhouse/train/smoke/`
- 原参数训练：`outputs/inhouse/train/full/`

`figures` 使用最终保存的数值结果，生成 `simulation_three_pf_validation_from_csv.pdf`、同名 PNG，以及 `_pf.csv`、`_training.csv`、`_validation.csv`。所有输入均在新仓库内，不需要父目录。

`train` 执行原 notebook 的训练、评估、最终拟合、三组 Pareto 优化、验证点选择，并保存 `trained_pareto_fronts.npz`、预测对实际值图、`selected_validation/` 下的新验证点，以及三框架图。每次运行的参数、结果来源、选中模型和完成状态写入 `run_metadata.json`。

## 原始来源与执行顺序

以下 cell 索引从 **0** 开始，完整哈希记录在 `provenance/inhouse.json`。

| 阶段 | 原 notebook cell | 使用内容 |
|---|---|---|
| 导入及数据 | 2, 4 | 共享原始 GP/Pareto 模块和 Excel 数据 |
| 预测图函数 | 8 | 原 `plot_pred_vs_actual_from_auto` |
| 配置、拟合、评估 | 11–14 | 50 行抽样、模型选择和测试评估 |
| 最终拟合 | 15–16 | 完整 50 行拟合和预测图 |
| 原始与原始不确定性 PF | 17 | 两组 NSGA-II 调用 |
| 校准与 CFSC PF | 22–25 | notebook 内原校准函数及调用 |
| 三框架验证点 | 37, 38, 40, 41 | 选择与导出 |
| 最终验证图 | 42 | 原 `plot_three_pf_with_validation` 函数 |

`workflows/inhouse/source/training_cells.json` 保存这些训练 cell 的原始源码字符串和逐 cell SHA-256。执行器只修改 cell 4 的输入路径和 cell 41 的输出路径。`source/plotting.py` 是 cell 42 的原始导入和函数定义；依赖 notebook 当前目录的调用部分由可移植入口替代。原算法模块从仓库内 `legacy/` 导入。

旧双框架选择/画图（29–36）、未调用的模型存储辅助函数（6–7）、探索性检查（18–20、26）及后续历史 seed sweep（43–53）不属于最终三框架图的必要依赖，未作为默认执行路径保留。cell 19 的保存输出有五个特征，与当前四特征数据不一致，故不用于恢复当前结果。

## 数据与参考结果

- `data/Contraction Channel Data.xlsx`：原始 168 行工作簿。特征为 `Lambda (mm)`、`Amplitude (mm)`、`Pin_Height (um)`、`v_in`；目标为 `Twall_avg`、`ΔP(Pa)`。所有名称、单位标签和数值均保持原样。
- `data/simulation_selected_validation_points.csv`：外部仿真给定的 25 条记录，包括原批次预测值、`Real T` 和 `Real P`。Raw 10 条，Raw + uncertainty 5 条，CFSC 10 条；不同批次和重复仿真均保留。
- `reference/*_pf.csv`：最终三组 PF，各 128 点，共 384 点。
- `reference/*_training.csv`：50 行背景训练目标，与工作簿 `df.sample(n=50, random_state=42)` 完全一致。
- `reference/*_validation.csv`、PNG、PDF：最终图及已保存误差表。
- `reference/simulation_selected_validation_points*`：原始 15 候选点导出及去重设计记录，作为选择阶段参考；不以其覆盖 25 条外部真值记录。

复制的原始输入与参考结果均逐字节保存，输出不会写回这些文件。原始 pickle 仅作为参考归档，`figures` 不需要加载 pickle。

## 原参数与 smoke 参数

完整训练保留原 notebook 参数：CPU、float64；50 行抽样与配置 seed=42；test/validation 比例均为 0.15；20 bags；原 GP 核搜索网格；测试评估使用 frozen scaler；最终 full refit 使用 refit scaler。校准保留 K=5、alpha=0.10、per_target、seed=7 和原 `model_key="current"` 调用。NSGA-II 保留模块默认 population=128、generations=150、seed=0；两个目标均最小化，不使用 hull 过滤。Raw uncertainty 保留 beta=1 与 UCB，CFSC 保留原校准回调。

`--smoke` **仅验证执行链条**，显式缩减为 20 行、0 bags、RBF 核网格、K=2、population=12、generations=2，并使用一个 Torch CPU 线程。其数值不代表原研究结果。所有覆盖参数均写入运行元数据。

## 外部仿真边界

本仓库可以训练代理模型、生成优化设计和画图；`Real T`、`Real P` 是已有外部仿真输入。原项目的 COMSOL `.mph` 模型文件体积很大，未纳入这个最小 Python 复现仓库，Python 入口不会运行 COMSOL，也不会生成新的仿真真值。

`train` 的最终图遵循 cell 42：背景 PF 来自本次重新训练，而验证标记及误差线使用 CSV 中原批次的预测值和真值。因此它们不能被解读为本次新候选点已完成仿真验证。新候选设计写入 `selected_validation/`，其 observed 字段保持空白，等待外部仿真。

保存结果重绘可准确恢复原图的数值输入。本次还在原 `arbo` 环境完成了原参数训练，生成的三张数值 CSV、最终 PNG 和两张候选点 CSV 均与历史归档逐字节一致。包内保留当前算法实现与原参数，不用重新训练结果替换历史参考数据。

## 已执行验证

在原 notebook 的 `arbo` Python 环境执行了以下检查：

```bash
python workflows/inhouse/test_figures.py
python workflows/inhouse/run.py train --smoke
```

集成测试从无关临时目录运行真实重绘入口，确认全部 25 条真值记录、384 个 PF 点、PNG/PDF 输出和参考输入不变。缩减训练实际完成 GP 拟合、评估、校准、三框架优化、选点和导出。

随后执行 `python reproduce.py run inhouse train`，完成原参数完整训练（50 行、20 bags、K=5、population=128、generations=150），用时约 150 秒。最终 PF、training、validation 三张 CSV、最终 PNG、selected_long 和 simulation_unique 两张候选点 CSV 均与原归档逐字节一致。这项检查使用已有外部仿真真值，没有重新求解 COMSOL。
