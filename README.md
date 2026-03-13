# SDFormer

**SDFormer: Multi-Granularity Time Series Modeling for Computing Power Network Supply-Demand Forecasting**

> 面向算力网络供需预测的多粒度时序建模方法

---

## 简介

算力网络资源供需序列同时包含**常态化低频周期成分**与**非常态化高频尖峰成分**，两者在时域完全混叠，给传统时序预测方法带来双重挑战：单一固定窗口无法同时适配两类行为，基于均值/方差的实例归一化对稀疏极端值高度敏感。

SDFormer 从行为解耦的视角出发，提出四个核心组件：

| 组件 | 功能 |
|------|------|
| **多粒度独立编码** | 三个并行 Transformer 编码器分别建模高频突变、过渡频段、低频趋势，消除跨粒度梯度干扰 |
| **ATSR**（自适应时序显著性重校准） | 在局部窗口投影前引入可学习逐点权重，自动放大非常态突变信号，防止尖峰被稀释 |
| **AGF**（自适应多粒度门控融合） | 依据全局上下文动态生成各粒度贡献权重，高突发性时偏向细粒度，平稳时偏向粗粒度 |
| **异常感知归一化** | 以峰度和尾概率为特征，自适应在中位数/MAD 与均值/标准差之间凸插值，从输入端保护特征空间 |

---

## 主要结果

在 ETTh1/ETTh2/ETTm1/ETTm2/Weather 五个基准数据集上与 PatchTST、iTransformer、Autoformer、FEDformer 对比：

- ETTh1 96步 MSE 较 PatchTST **降低 7.6%**，较 iTransformer **降低 7.5%**
- ETTh1 192步 MSE 较 PatchTST **降低 14.9%**
- ETTh2 全部预测步长最优
- ETTm/Weather 中长期预测（336步以上）保持竞争力

---

## 环境要求

```
Python   3.12.3
PyTorch  2.5.1
CUDA     12.1
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 数据准备

从 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 下载数据集，放置于 `./dataset/` 目录：

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
└── weather/
    └── weather.csv
```

---

## 快速开始

**训练**

```bash
python run.py \
  --model SDFormer \
  --data ETTh1 \
  --seq_len 512 \
  --pred_len 96 \
  --patch_sizes 16 32 64 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 1 \
  --d_ff 2048 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20
```

**复现论文结果**

```bash
bash scripts/run_all.sh
```

---

## 项目结构

```
SDFormer/
├── models/
│   └── SDFormer.py          # 模型主体
├── layers/
│   ├── anomaly_norm.py      # 异常感知归一化
│   ├── atsr.py              # 自适应时序显著性重校准
│   └── agf.py               # 自适应多粒度门控融合
├── data_provider/           # 数据加载
├── exp/                     # 训练/评估流程
├── scripts/                 # 复现脚本
├── dataset/                 # 数据集（需自行下载）
├── run.py
└── requirements.txt
```

---

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{sdformer2025,
  title     = {SDFormer: Multi-Granularity Time Series Modeling for Computing Power Network Supply-Demand Forecasting},
  year      = {2025}
}
```

---

## 致谢

本项目基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 框架实现，感谢 PatchTST 和 iTransformer 的开源工作。