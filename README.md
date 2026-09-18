# Surrogate models: final-result reproduction

本仓库整理现有项目最终采用的 benchmark、public datasets 和 in-house validation 流程。保留原有科学计算实现、实验设置与已归档结果，补齐独立运行入口、数据、环境版本和结果出处。运行不需要上级 `physics_test` 目录。

## 快速开始

建议使用 Python 3.10.15。原 notebook 的 kernel 为 `arbo`；`requirements.txt` 与 `requirements-lock.txt` 根据该环境当前安装的科学计算包及其依赖版本整理。

```bash
cd surrogate_reproducible
conda env create -f environment.yml
conda activate surrogate-reproducible
python reproduce.py verify --environment
python reproduce.py figures
```

如果继续使用本机原有环境，可直接 `conda activate arbo`，无需重复安装。也可以在 Python 3.10 虚拟环境中执行 `python -m pip install -r requirements.txt`。

`figures` 会从仓库内的最终数值结果重新绘图、生成表格，输出到 `outputs/`。它实际执行绘图程序，不是复制 PDF。原始归档留在各流程的 `reference/` 中。

```bash
python reproduce.py list
python reproduce.py figures --workflow benchmark
python reproduce.py figures --workflow public
python reproduce.py figures --workflow inhouse
```

## 三条结果链

| 流程 | 采用的出处 | 可运行内容 |
|---|---|---|
| Benchmark | `revision/results/For paper/generate_paper_artifacts.py`、最终 FON/DTLZ2 表及其对应 notebook cells | Fig. 5.1–5.4、论文表格、分阶段训练与原始 Pareto 流程 |
| Public datasets | 最新 robust nested repeated validation 的共同有效 cells；`CaseStudy_battery_Li.ipynb` / `CaseStudy_battery_Ver.ipynb`；`pareto_results/` | Li/Verma nested surrogate、nested calibration、最终多 seed Pareto、reliability 与 Pareto 绘图 |
| In-house | `CaseStudy_battery_sim.ipynb` 的最终三框架流程及 cell 42 | 训练、校准、三种 Pareto 前沿、候选点导出、已有仿真结果验证图 |

详细对应关系、参数及命令见 [benchmark](docs/benchmark.md)、[public datasets](docs/public.md)、[in-house](docs/inhouse.md)。

## 从原始数据重新计算

训练入口与结果重画入口分开。运行参数中的 `--smoke` 表示缩小规模的通路验证，会改变样本/重复次数等设置；不能用于替代论文数值。

```bash
# 查看 benchmark 各个原始实验阶段
python reproduce.py run benchmark train --help

# Public datasets：完整 nested repeated validation
python reproduce.py run public train --dataset li --stage nested-surrogate
python reproduce.py run public train --dataset li --stage nested-calibration
python reproduce.py run public train --dataset verma --stage nested-surrogate
python reproduce.py run public train --dataset verma --stage nested-calibration

# Public datasets：最终 CaseStudy 多 seed Pareto 流程
python reproduce.py run public train --dataset li --stage pareto
python reproduce.py run public train --dataset verma --stage pareto

# In-house：原始训练、校准、Pareto 和候选点选择流程
python reproduce.py run inhouse train
```

先确认环境与运行通路：

```bash
python reproduce.py run public train --dataset li --stage nested-surrogate --smoke
python reproduce.py run public train --dataset li --stage nested-calibration --smoke
python reproduce.py run public train --dataset li --stage pareto --smoke
python reproduce.py run inhouse train --smoke
```

完整重复 GP 实验计算量较大。每次运行的日志和退出状态保存在 `outputs/runs/`；各训练流程另存实际设置与 smoke 标记。可通过各流程的 `--output-dir` 指定独立输出目录。

## 复现范围

- **Benchmark 最终表是多个历史批次的合并结果。** 已保存的最终表和绘图来源完整保留。准确率、Pareto 与后续 CV 指标存在不同的样本量/实验批次；当前留存代码没有完整记录所有最终表手工合并步骤。重新训练的阶段输出不会被冒充为整张论文表的精确复现。详见 benchmark 来源说明。
- **Public 的 nested validation 与最终 Pareto 是不同的原始流程。** 二者分别保留。最新 nested notebook 后追加但未完整跑出的 Verma Pareto 实验未用来替换最终 `pareto_results`。
- **In-house 使用已提供的仿真真值。** 数据包包含 25 条验证记录；训练数据采用原 notebook 的 `Contraction Channel Data.xlsx`，并按原设置抽取 50 行。仓库可以训练代理模型、选择设计点并复现已有验证图。新增设计点的仿真求解仍需原外部求解器；数 GB 的 COMSOL `.mph` 文件不纳入这个代码仓库。
- **环境记录是当前可用的原 kernel 环境快照。** 未找到每个历史运行时的完整依赖锁。当前观测到的 PyTorch 为 `2.7.1+cu126`；通用依赖文件固定为 `2.7.1`，具体 CUDA/CPU wheel、硬件和字体可能导致数值或 PDF 字节差异。`verify --environment` 会列出实际差异。

已执行的验证和实际复现程度见 [验证记录](docs/validation.md)。

本次已跑完原参数的 in-house 完整训练：最终三张数值 CSV、PNG 和两张候选点导出 CSV 与原归档逐字节一致。Benchmark 与 public datasets 已验证绘图、来源汇总和缩减训练，尚未重跑全部多 seed / nested 实验。

## 目录

```text
reproduce.py                  统一运行入口与运行日志
legacy/                       原始共享科学实现，逐字节保留
workflows/benchmark/           benchmark 来源 cells、绘图脚本、结果快照
workflows/public/              原始数据、nested/Pareto cells、结果快照
workflows/inhouse/             原始数据、最终流程 cells、仿真验证快照
provenance/                    原路径、SHA-256、cell 索引和适配记录
docs/                         出处说明、运行说明与验证记录
tests/                        仓库完整性和运行入口检查
outputs/                      新运行结果；Git 忽略
```

源码 cell 保留原内容，由小型执行器提供正确执行顺序和仓库内路径。只有明确指定 `--smoke` 时才缩减实验设置。`provenance/manifest.json` 检查分发文件完整性，各流程来源清单记录原项目路径及 notebook cell 索引（从 0 开始）。

## 文件筛选原则

未采用 `Copy` notebook、早期重复实验、大量中间图片、缓存和独立旧版重建项目。`surrogate_conformal_rebuild/` 依赖原目录且目标并非本次指定的全部最终结果，因此不作为新仓库主体。共享模块在 `revision/` 和 `Surrogate_models_for_small-size_datasets/` 中内容一致，仓库中只保留一份。更细的筛选依据见各流程文档。
