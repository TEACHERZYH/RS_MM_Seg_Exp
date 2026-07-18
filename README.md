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
  --output-root data/qalf_minimal_m2_20260715/Vaihingen_prepared_v4_confirmatory_v1

python scripts/create_holdout_split_roots.py \
  --dataset potsdam \
  --source-root data/Potsdam_prepared_v4 \
  --output-root data/qalf_minimal_m2_20260715/Potsdam_prepared_v4_confirmatory_v1
```

正式划分记录位于：

```text
configs/minimal_claim_m2_r2_20260716/split_lock.json
```

其中记录了训练、验证和 internal hold-out 的数量、留出图幅及 split SHA-256。生成后应核对脚本输出的数量和哈希；若本地数据根目录不同，只修改配置中的 `dataset.root_dir`，不要改变图幅划分、随机种子或训练预算后仍将结果视为精确复现。Windows 若无法创建目录符号链接，可启用开发者模式，或用等价目录链接/只读副本替代。

## 冻结实验矩阵

论文主实验配置位于：

```text
configs/minimal_claim_m2_r2_20260716/
  main/                         QALF 与 fixed-late 的三随机种子主矩阵
  control/                      main-only、availability-masked fixed 和 quality-only 控制
  training_stage_manifest.csv   43 个训练阶段及配置哈希
  main_evaluation_manifest.csv
  control_evaluation_manifest.csv
  qualitative_selection_manifest.csv
  split_lock.json
```

主矩阵覆盖 Vaihingen、Potsdam，随机种子 `101/202/303`，以及 `base_clean`、`clean_continuation`、`robust` 三种训练阶段。配置中的 `resume` 路径固定了阶段依赖。

## 训练

以 Vaihingen、seed 101 的 QALF 为例，先训练 clean base，再执行 robust continuation：

```bash
python train.py --config \
  configs/minimal_claim_m2_r2_20260716/main/m3-010_vaihingen_qalf_base_clean.yaml

python train.py --config \
  configs/minimal_claim_m2_r2_20260716/main/m3-012_vaihingen_qalf_robust.yaml
```

强静态控制采用同等数据、种子和训练预算，例如：

```bash
python train.py --config \
  configs/minimal_claim_m2_r2_20260716/main/m3-001_vaihingen_fixed_late_base_clean.yaml

python train.py --config \
  configs/minimal_claim_m2_r2_20260716/main/m3-003_vaihingen_fixed_late_robust.yaml
```

## 评估

单检查点评估：

```bash
python evaluate.py \
  --config configs/minimal_claim_m2_r2_20260716/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_minimal_m2_20260715/main/vaihingen/qalf/seed_101/robust/last_model.pt
```

统一场景协议评估：

```bash
python scripts/run_eval_protocol.py \
  --config configs/minimal_claim_m2_r2_20260716/main/m3-012_vaihingen_qalf_robust.yaml \
  --checkpoint outputs/qalf_minimal_m2_20260715/main/vaihingen/qalf/seed_101/robust/last_model.pt \
  --split val_split \
  --output-dir outputs/eval_protocol_vaihingen_qalf_robust
```

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
```

## 复现边界

- 正式主结果使用固定预算的 `last_model.pt`，不在 internal hold-out 上选择检查点或调参。
- internal hold-out 与官方 ISPRS test server 不同；仓库不报告官方服务器结果。
- missing-primary 只作为压力测试，不属于主鲁棒性主张。
- 蒸馏、functional-entropy 和其他探索性升级不构成论文主贡献。
- 训练权重未包含在当前公开包中；复现实验需按冻结配置重新训练。
