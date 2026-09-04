# LookStep

[English](README.md) [MODEL](https://huggingface.co/Kunyang-YU/LookStep)

LookStep 是论文 **LookStep: Efficient Vision-Language Navigation with
Linguistic Foresight and Event Driven Memory** 的独立主实验复现包。下载论文
checkpoint，并按官方许可准备 Matterport3D 与 VLN-CE 数据后，可以只使用本目录评测
R2R-CE 与 RxR-CE Val-Unseen 主结果；也可以从 expert trajectories 重新构造监督数据并
训练模型。


## 三个模块

```text
LookStep/
├── data_construction/   # 1. R2R/RxR expert trajectory -> LFS/EDRM 监督 JSONL
├── model_training/      # 2. ms-swift 数据转换与 Qwen3-VL full fine-tuning
└── simulation/          # 3. Habitat R2R/RxR 在线主实验评测
```

完整流水线：

```text
R2R-CE + RxR-CE expert trajectories
  -> deterministic LookStep labels
  -> ms-swift multimodal JSONL
  -> Qwen3-VL-8B full fine-tuning
  -> Habitat R2R-CE / RxR-CE Val-Unseen evaluation
```

训练 target 的固定顺序为：

```text
progress -> event -> memory_write -> memory_role
         -> outcomes for four candidate actions -> action
```

## 下载模型后复现主结果

先按[中文复现指南](REPRODUCIBILITY_ZH.md)准备固定环境和受许可数据，然后运行：

```bash
bash LookStep/reproduce_paper.sh test
conda activate lookstep-simulation

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh check-sim

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh smoke-r2r

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh eval-all

bash LookStep/reproduce_paper.sh verify
```


## 主入口

| 模块 | 入口 | 用途 |
| --- | --- | --- |
| 数据构造 | `data_construction/build_short_label_dataset.py` | 从 R2R/RxR expert actions 派生主实验标签 |
| 模型训练 | `model_training/prepare_ms_swift_data.py` | 按 episode 划分并转换为 ms-swift JSONL |
| 模型训练 | `model_training/scripts/train_qwen3vl_full.sh` | 论文 checkpoint 的 full fine-tuning 配方 |
| 模拟仿真 | `simulation/evaluate_short_label_sim.py` | Habitat 在线 EventFIFO 导航 |
| 统一入口 | `reproduce_paper.sh` | 分阶段 check/build/prepare/train/eval/verify |

## 引用

```bibtex
@inproceedings{lookstep,
  title     = {LookStep: Efficient Vision-Language Navigation with Linguistic Foresight and Event Driven Memory},
  author    = {Kun-Yang Yu and Yingzhe Li and Hongyu Xu and Shi-Yu Tian and Zhi Zhou and Yang Chen and Ming Yang and Sheng Wang and Qing Yu and Lan-Zhe Guo and Yu-Feng Li},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

If you have any questions, please contact yuky@lamda.nju.edu.cn.
