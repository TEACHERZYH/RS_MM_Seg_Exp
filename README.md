# RS_MM_Seg_Exp

QALF 的论文复现实验代码。该项目面向 ISPRS Vaihingen 与 Potsdam 的光学影像/DSM 多模态语义分割，覆盖完整输入、DSM 缺失、输入退化及二者组合等条件。

## 论文主线

实验支持的结论边界为：

- robust training 是性能改善的主要来源；
- QALF 以较小结构开销提供显式模态可用性处理、质量分数、局部门控图和可观测融合接口；
- robust fixed-late fusion 是必须保留的强性能控制；
- 不主张 QALF 在所有数据集和退化条件下均优于该控制；
- 不主张质量分数是校准概率、门控图是因果解释，也不主张官方 ISPRS 测试服务器成绩或普遍 SOTA。

在冻结效率协议下，相对匹配的 robust fixed-late 控制，QALF 的部署路径增加约 `0.0387 M` 活跃参数（`1.73%`）和 `0.1897 GFLOPs`（`0.39%`）。

## 公开内容

本仓库仅公开复现论文实验通常需要的内容：

- `src/`：模型、数据加载、损失函数和训练/评估引擎；
- `train.py`、`evaluate.py`：单配置训练与评估入口；
- `configs/`：基础配置以及论文最终冻结实验矩阵；
- `scripts/`：数据准备、协议评估、效率统计、退化测试和定性结果导出；
- `requirements.txt`：Python 依赖。

以下内容不随仓库分发：ISPRS 原始数据、切片数据、模型权重、检查点、远程主机信息、Slurm 运维脚本、同步回执、计费记录、内部审查文档、论文源文件和历史实验归档。

## 环境

建议使用独立 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

完整训练应使用 CUDA GPU。CPU 适合执行导入检查、小样本测试和部分结果汇总。

冻结配置使用 torchvision 官方 MobileNetV3-Large 初始化权重，并固定到以下本地路径。首次运行前可执行：

```bash
python -c "from pathlib import Path; from torch.hub import download_url_to_file; from torchvision.models import MobileNet_V3_Large_Weights as W; p=Path('assets/pretrained/mobilenet_v3_large-5c1a4163.pth'); p.parent.mkdir(parents=True, exist_ok=True); download_url_to_file(W.DEFAULT.url, str(p), hash_prefix='5c1a4163')"
```

## 数据

Vaihingen 和 Potsdam 数据须按 ISPRS benchmark 的访问政策自行获取，本仓库不重新分发。先从带标注的训练图幅生成基础切片；`test.txt` 在此阶段保持为空：

```bash
python scripts/prepare_isprs_direct.py \
  --dataset vaihingen \
  --raw-zip data_raw/Vaihingen.zip \
  --output-root data/Vaihingen_prepared_v4 \
  --splits train,val

python scripts/prepare_isprs_direct.py \
  --dataset potsdam \
  --raw-zip data_raw/Potsdam.zip \
  --output-root data/Potsdam_prepared_v4 \
  --splits train,val
```

随后按不读取影像或标签内容的冻结规则生成 confirmatory 划分。默认 `symlink` 模式避免复制大体量切片：

```bash
python scripts/create_holdout_split_roots.py \
  --dataset vaihingen \
  --source-root data/Vaihingen_prepared_v4 \
  --output-root data/qalf_repro_fair_20261102/Vaihingen_prepared_v4_confirmatory_v1

python scripts/create_holdout_split_roots.py \
  --dataset potsdam \
  --source-root data/Potsdam_prepared_v4 \
  --output-root data/qalf_repro_fair_20261102/Potsdam_prepared_v4_confirmatory_v1
```

正式划分记录位于：

```text
configs/minimal_claim_m2_r2_20261102_fair/split_lock.json
```

其中记录了训练、验证和 internal hold-out 的数量、留出图幅及 split SHA-256。生成后应核对脚本输出的数量和哈希；若本地数据根目录不同，只修改配置中的 `dataset.root_dir`，不要改变图幅划分、随机种子或训练预算后仍将结果视为精确复现。Windows 若无法创建目录符号链接，可启用开发者模式，或用等价目录链接/只读副本替代。

## 冻结实验矩阵

新的公平复现矩阵位于：

```text
configs/minimal_claim_m2_r2_20261102_fair/
  main/                         QALF 与 fixed-late 的三随机种子主矩阵
  control/                      main-only、availability-masked fixed 和 quality-only 控制
  holdout_evaluation_manifest.csv 绑定 internal holdout/test_split 与 split SHA-256
  training_stage_manifest.csv   43 个训练阶段及配置哈希
  main_evaluation_manifest.csv
  control_evaluation_manifest.csv
  qualitative_selection_manifest.csv
  split_lock.json
```

主矩阵覆盖 Vaihingen、Potsdam，随机种子 `101/202/303`，以及 `base_clean`、`clean_continuation`、`robust` 三种训练阶段。Vaihingen 使用 `45+28+28` 个 epoch，Potsdam 使用 `35+13+13` 个 epoch。主 fixed-late 路由使用 `availability_masked_average`，缺失 DSM 时按可用性归一化；旧的 `minimal_claim_m2_r2_20260716` 矩阵保留为历史 fixed-average 版本，不用于新表格。

两个 continuation 均从同一 base-clean 末轮分叉，分别作为等预算对照，不能按三个阶段连续训练。单条完整路径的预算为 Vaihingen `45+28`、Potsdam `35+13`。新矩阵尚无正式重跑结果，不能把历史表格归属于新配置。

可先核对全部配置哈希、阶段依赖和 split 绑定，再导出命令清单：

```bash
python scripts/plan_reproduction.py --output outputs/reproduction_commands.json
```

此命令只生成清单，不执行训练或评估，也不读取影像和标签。清单中的 holdout 命令须在单独授权后运行。

## 训练

以 Vaihingen、seed 101 的 QALF 为例，先训练 clean base，再执行 robust continuation：

```bash
python train.py --config \
  configs/minimal_claim_m2_r2_20261102_fair/main/m3-010_vaihingen_qalf_base_clean.yaml

python train.py --config \
  configs/minimal_claim_m2_r2_20261102_fair/main/m3-012_vaihingen_qalf_robust.yaml
```

强静态控制采用同等数据、种子和训练预算，例如：

```bash
python train.py --config \
  configs/minimal_claim_m2_r2_20261102_fair/main/m3-001_vaihingen_fixed_late_base_clean.yaml

python train.py --config \
  configs/minimal_claim_m2_r2_20261102_fair/main/m3-003_vaihingen_fixed_late_robust.yaml
```

## 评估

单检查点评估：

```bash
python evaluate.py \
  --config configs/minimal_claim_m2_r2_20261102_fair/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/last_model.pt
```

统一场景协议评估：

```bash
python scripts/run_eval_protocol.py \
  --config configs/minimal_claim_m2_r2_20261102_fair/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/last_model.pt \
  --split val_split \
  --trials 3 --single-clean-trial \
  --output-dir outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/eval_val_split
```

同一检查点的 internal holdout 评估使用配置中的 `holdout_split`，并在输出元数据中保存 split 文件哈希：

```bash
python scripts/run_eval_protocol.py \
  --config configs/minimal_claim_m2_r2_20261102_fair/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/last_model.pt \
  --split holdout_split \
  --trials 3 --single-clean-trial \
  --output-dir outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/eval_holdout_split
```

严重度扫描需显式冻结 M2 协议：

```bash
python scripts/eval_degradation_severity_curve.py \
  --config configs/minimal_claim_m2_r2_20261102_fair/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_repro_fair_20261102/main/vaihingen/qalf/seed_101/robust/last_model.pt \
  --split val_split \
  --output-dir outputs/severity_m2_vaihingen_qalf_robust \
  --protocol m2 --dataset-key vaihingen --include-combined
```

当前主矩阵汇总入口为：

```bash
python scripts/summarize_minimal_claim_results.py \
  --manifest configs/minimal_claim_m2_r2_20261102_fair/main_evaluation_manifest.csv \
  --output-dir outputs/qalf_repro_fair_20261102/statistics \
  --allow-incomplete
```

`scripts/summarize_full_rebuild_results.py` 仅保留用于历史 `rebuild_20260714` 证据，不应再用于当前 24 条主评估记录。

新版汇总要求每个结果目录包含 `eval_protocol_metadata.json`，核对配置、检查点、split 和结果 CSV 的哈希；不符合身份绑定的结果列为无效，不混入统计。三种子汇总输出 `training_seed_summary.csv` 和 `paired_seed_contrasts.csv`，后者分别计算 robust 相对 clean-continuation、QALF 相对 fixed-late 的配对差值、95% Student-t 区间和正差种子数。零支持类别的 `not_estimable` 保持不可估计，不补零或静默换用其他指标。

公平矩阵的评估还会强制核对检查点内嵌配置、末轮 epoch 和 `training_complete.json`，避免错误种子或旧权重换名后被接收。汇总同时核对结果 JSON、逐类 CSV、混淆矩阵 JSON 的哈希及真实 trial 数量；主评估 full/missing 各 1 次，degraded/combined 各 3 次。不同 split 哈希不能合并为配对统计。上述验证不代表正式实验已经完成。

M2 的 blur 为平均池化核（非 Gaussian blur），noise/lowres/mask 严重度也独立冻结；论文中的 Gaussian blur 图文不能未经核对归属于 M2。默认 `legacy` 仅用于历史协议，正式扫描须明确写出 `--protocol`。

补充分析入口：

```bash
python scripts/eval_degradation_severity_curve.py --help
python scripts/eval_misalignment_stress.py --help
python scripts/measure_efficiency.py --help
python scripts/export_comparative_qualitative.py --help
```

## 最小检查

无需完整数据即可先检查代码连通性：

```bash
python -m compileall -q src scripts train.py evaluate.py
python scripts/create_dummy_isprs_data.py --help
python scripts/test_result_aggregation_contracts.py
python scripts/test_minimal_claim_repro_contracts.py
python scripts/test_seed_statistics.py
python scripts/test_reproduction_plan.py
python scripts/summarize_minimal_claim_results.py --help
```

## 复现边界

- 正式主结果使用固定预算的 `last_model.pt`，不在 internal hold-out 上选择检查点或调参。
- internal hold-out 与官方 ISPRS test server 不同；仓库不报告官方服务器结果。
- missing-primary 只作为压力测试，不属于主鲁棒性主张。
- 蒸馏、functional-entropy 和其他探索性升级不构成论文主贡献。
- 训练权重未包含在当前公开包中；复现实验需按冻结配置重新训练。
