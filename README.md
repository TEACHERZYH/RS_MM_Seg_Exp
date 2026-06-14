# RS_MM_Seg_Exp

面向论文《缺失与低质量模态场景下的轻量多模态遥感分割》的实验工程。

## 当前目标

- 支持 `ISPRS Vaihingen` 和 `ISPRS Potsdam`
- 支持 `RGB/IRRG + DSM` 双模态输入
- 支持完整模态、缺失模态和退化模态实验协议
- 提供可扩展的轻量多模态分割框架

## 当前实现

- 轻量 backbone：`MobileNetV3-Small` 风格编码器
- 质量估计模块：`ModalityQualityEstimator`
- 动态门控融合模块：`DynamicGatedFusion`
- 主模型：`QALFNet`
- 训练入口：`train.py`
- 评估入口：`evaluate.py`

## 目录结构

```text
RS_MM_Seg_Exp/
  configs/
    default.yaml
  src/
    datasets/
      isprs_dataset.py
    models/
      qalf_net.py
    engine.py
    losses.py
    utils.py
  outputs/
  train.py
  evaluate.py
  requirements.txt
```

## 数据组织建议

```text
data/
  Vaihingen/
    images/
    dsm/
    masks/
    splits/
      train.txt
      val.txt
      test.txt
  Potsdam/
    images/
    dsm/
    masks/
    splits/
      train.txt
      val.txt
      test.txt
```

其中：

- `images/` 保存 RGB 或 IRRG 图像
- `dsm/` 保存高度图
- `masks/` 保存单通道语义标签图
- `splits/*.txt` 每行一个样本 ID，不带扩展名

## 运行流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备真实数据

#### Potsdam

```bash
python scripts/prepare_isprs.py ^
  --dataset potsdam ^
  --raw-zip data_raw/Potsdam.zip ^
  --output-root data/Potsdam_prepared
```

#### Vaihingen

```bash
python scripts/prepare_isprs.py ^
  --dataset vaihingen ^
  --raw-zip data_raw/Vaihingen.zip ^
  --output-root data/Vaihingen_prepared
```

### 3. 训练

```bash
python train.py --config configs/default.yaml
```

或使用正式实验配置：

```bash
python train.py --config configs/potsdam_rgb_dsm.yaml
python train.py --config configs/vaihingen_irrg_dsm.yaml
```

### 4. 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

## 说明

- 当前版本先提供完整实验骨架，后续可继续补充：
- 更强 backbone
- 更细致的数据增强
- 边界辅助分支
- Optical-SAR 扩展协议
- 更完整的日志和可视化输出
