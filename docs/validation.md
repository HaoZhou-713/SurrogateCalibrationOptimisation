# 验证记录

验证日期：2026-09-18（Australia/Brisbane）。实际运行环境为原 notebook 的 `arbo`，Python 3.10.15；版本见 `provenance/environment-observed.json`。未声称已在全新环境、其他操作系统或其他硬件上验证。

## 原参数完整训练

```bash
python reproduce.py run inhouse train
```

执行成功，约 150 秒。50 条训练数据、20 bags、K=5、NSGA-II population=128 / generations=150，全程保留原设置。以下六项与归档文件**逐字节一致**：

- `simulation_three_pf_validation_from_csv_pf.csv`：384 个 PF 点。
- `simulation_three_pf_validation_from_csv_training.csv`：50 行训练目标。
- `simulation_three_pf_validation_from_csv_validation.csv`：25 条外部验证记录。
- `simulation_three_pf_validation_from_csv.png`。
- `simulation_selected_validation_points_selected_long.csv`。
- `simulation_selected_validation_points_simulation_unique.csv`。

三个数值表最大绝对差为 0。PDF 正常生成；不要求包含生成时间等元数据的 PDF 文件字节相同。外部 COMSOL 仿真未重新求解，已有真值 CSV 是输入。

## 最终结果重绘

`python reproduce.py figures` 实际运行三条绘图流程，正常生成 PDF、PNG 和数值表。

| 检查 | 结果 |
|---|---|
| Benchmark 主表 numeric / formatted、calibration diagonal | 与归档表一致 |
| DTLZ2 sensitivity 与 benchmark cost 数值 | 除文件来源路径外，与归档值一致 |
| Benchmark cost formatted | 六个原列一致；当前原始脚本额外导出 `Runs` 列，保留该行为 |
| Benchmark 最终 Pareto 汇总来源 | 270 个汇总值与原始运行 CSV 匹配 |
| Verma 四张 PF 统计表 | 在 `rtol=atol=1e-12` 下与归档表一致；最大绝对差约 `1.04e-16` |
| Li/Verma 最终 PF 绘图 | 全部保存点保留；Verma 每投影保留 7,680 个点 |
| In-house 已保存结果重绘 | 25 条验证记录、384 个 PF 点、50 条训练目标均保留 |

抽查了生成的 Fig. 5.1 与 Verma 主图，标签、图例和面板显示完整。PNG/PDF 的跨版本字节一致性不作为通用判据。

## 缩减训练

以下通路完成了实际 GP 拟合和对应后续计算，退出码为 0。`--smoke` 的设置覆盖写入运行元数据，不能替代完整论文实验。

- Benchmark：FON/DTLZ2 true vanilla；DTLZ2 five-methods selected；FON five-methods CV；FON/DTLZ2 CV metrics；DTLZ2 legacy CV Pareto；DTLZ2 sensitivity N=30。
- Public：Li 与 Verma 均完成 nested-surrogate、nested-calibration、Pareto smoke。Verma 两项 nested 输出的全部数值指标有限。
- In-house：训练、校准、三个 PF、候选点选择和导出 smoke。

Benchmark 全部 20-seed 重训练、public 完整 nested repeated validation 和 10-seed Pareto 尚未重新执行。其可运行入口、原参数、数据及已归档结果均保留；历史 benchmark 最终表合并过程的缺口详见 `benchmark.md`。

## 仓库与运行封装

- 复制完整代码/数据包到另一层级目录，从外部工作目录运行统一 `figures` 入口，三条流程全部成功；在那里额外完成一次 FON vanilla smoke。
- 独立复制目录中不包含原项目根目录的数据文件或原 notebook，运行仅使用包内路径。
- 共享模块及抽出的 benchmark/public/in-house cell 内容已与原始文件核对。原始项目文件保持不变。
- 所有选定 Python 文件通过编译检查。
- 13 项测试通过：4 项仓库入口/完整性/日志检查、3 项 benchmark 执行封装检查、5 项已有 Verma 数值测试、1 项 in-house 真实绘图集成测试。
- 已修复并验证：同一时间戳的日志覆盖；benchmark 输出目录指向仓库根目录时覆盖 README；sensitivity 早期批次失败未被最后状态反映。均只修改运行封装。

复跑测试：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s workflows/benchmark -p "test*.py" -v
python -m unittest discover -s workflows/public/plots -p "test*.py" -v
python -m unittest discover -s workflows/inhouse -p "test*.py" -v
python reproduce.py verify --environment
```

Windows 迁移检查中，更深层的临时路径触发了旧版 Python 的长路径限制；换到较短路径后完整复制及运行成功。Windows 使用者应把仓库放在较短路径下，例如 `C:\work\surrogate_reproducible`。

`provenance/validation-results.json` 保存关键数值比较记录。运行日志和完整新生成文件留在 `outputs/`，不纳入 Git。
