# 脑部MRI分类深度学习模型

## 项目概览
本项目是一个基于3维MRI图像的分类任务，旨在帮助影像科医生区分脑膜疾病。具体而言，有三种病症：脑膜炎、脑炎和脑膜转移（转移瘤）。我们需要模型根据输入的MRI图像，给出分类结果。加上正常的案例，一共有四个类别。实际应用中，由于脑炎和脑膜炎同属炎症，区分起来很困难，且区分炎症和肿瘤更有医学上的意义，所以目前将数据中的脑膜炎和脑炎合并为炎症类，任务简化为三分类问题。

项目使用 T1WI、T2WI、FLAIR 三种对齐序列。早期设想是将 FLAIR 序列上的医生标注 mask 迁移到 seq1、seq2、seq3，统一使用分类+分割双头模型训练；但实验发现，seq3 的 mask 辅助对 seq1 和 seq2 的分类效果不是正向收益。因此当前最优方案是：seq1、seq2 复用老项目中不涉及 mask 的 `FoundationModel_ori` 及其权重，seq3 使用带分类头和分割头的 `FoundationModel`，最后执行异构晚期融合软投票。

## 项目结构
```text
.
├── AGENTS.md
├── README.md
├── environment.yml
├── configs
│   ├── global_config.py
│   └── config_utils.py
├── models
│   ├── cnn3d.py
│   ├── FoundationModel_ori.py
│   ├── FoundationModel.py
│   └── ResNet.py
├── scripts
│   ├── preprocess_data.py
│   ├── preprocess_mask.py
│   ├── build_dataset_kfold.py
│   ├── check_dataset_kfold.py
│   ├── check_preprocessed_data.py
│   ├── check_file_sizes.py
│   ├── delete_small_preprocessed_cases.py
│   ├── get_raw_data_info.py
│   ├── deprecated_build_dataset.py
│   └── deprecated_check_dataset.py
├── utils
│   ├── data_scan.py
│   ├── dataset.py
│   ├── intensity.py
│   ├── io.py
│   ├── resample.py
│   ├── sequences.py
│   ├── spatial.py
│   └── train_and_test.py
├── baseline_scripts
│   ├── lda.py
│   ├── svm.py
│   ├── mlp.py
│   └── MLPResNet.py
├── Helper
│   ├── check_pth.py
│   ├── eval_pt.py
│   ├── get_mask_num.py
│   ├── pick_cases.py
│   ├── pick_easy_cases.py
│   └── seg_pred.py
├── eval_kfold.py
├── eval.py
├── eval_vote_kfold.py
├── external_eval.py
├── infer_kfold.py
├── infer.py
├── read_kfold_pth.py
├── train_kfold.py
├── train_meta_fusion.py
├── deprecated_train.py
├── runtime_defaults.py        # 尚未完全参数化入口的迁移后默认实验
├── output                    # 新实验配置与产物，整体被 Git 忽略
│   ├── data-hdbet
│   │   ├── preprocessing_config.py
│   │   ├── dataset_config.py
│   │   ├── data
│   │   └── datasets
│   └── runs-cross-entropy
│       ├── train_config.py
│       ├── checkpoints
│       └── output_texts
```

项目主要子文件和子文件夹意义如下：
1. configs：`global_config.py` 只保存类别、序列、随机种子、K-Fold 和原始数据路径等跨阶段设置；`config_utils.py` 负责加载实验目录中的阶段配置并解析标准产物目录
2. models：模型的实现，其中`FoundationModel_ori.py`用于seq1/seq2纯分类模型，`FoundationModel.py`用于seq3带分割头的多任务模型
3. utils：一些工具的实现
4. scripts：必要的脚本实现，包含“数据预处理”“mask预处理”和“训练集、验证集和测试集的生成”
5. train_kfold.py：训练脚本，配合k折交叉验证使用
6. eval_kfold.py：单序列或多通道模型测试脚本，先计算每一折的结果，再综合评估
7. eval.py：当前最优异构晚期融合软投票测试脚本；eval_vote_kfold.py保留为同构模型软投票评估脚本
8. `train_meta_fusion.py`：使用跨折 OOF 概率选择三个非负模态权重和转移阈值；固定等权 normal gate，并对每个目标 fold 做无标签泄漏的 cross-fitted 评估
9. infer.py / infer_kfold.py：预测脚本，支持临床晚期融合推理和k折单例推理
10. `deprecated_train.py`：旧的非 K-Fold 训练入口，仅用于明确提示迁移到 `train_kfold.py`
11. 原 `version1` 产物已迁移到 `output/data-hdbet` 和 `output/runs-cross-entropy`，目录组织与新规范一致
12. environment.yml：项目用的conda虚拟环境

## 实验过程

### 新实验运行规范

配置和产物统一放在项目根目录的 `output/` 中，不再在根目录散落新的数据或训练目录。原 `version1/` 已按此规范迁移；所有后续实验继续遵守以下结构：

```text
output/
├── data-hdbet/
│   ├── preprocessing_config.py
│   ├── dataset_config.py
│   ├── data/                         # 预处理图像、mask 和索引
│   └── datasets/                     # K-Fold 划分
└── runs-cross-entropy/
    ├── train_config.py
    └── checkpoints/                  # 本次训练权重
```

命名和运行规则：

1. 数据实验根使用 `output/data-<描述>`，例如 `output/data-hdbet`、`output/data-no-skull-strip`。
2. 训练实验根使用 `output/runs-<描述>`，例如 `output/runs-cross-entropy`、`output/runs-focal-loss`；不再增加 `runs/xxx` 这一层。
3. 实验目录名使用小写英文、数字和连字符，名称应能说明关键差异。
4. 每个实验根只对应一套配置和产物。改变关键参数时新建目录，不在原实验根中混放多套配置或覆盖已有结果。
5. 新配置从一个已有实验的配置复制后再修改；运行时始终传目标实验根内的配置文件。`configs/` 不保存阶段配置模板。
6. `output/` 下所有内容（包括实验配置）均被 Git 忽略；需要长期保留或分享的配置应另行归档。

以迁移后的历史实验为模板，初始化新的数据实验和训练实验：

```bash
mkdir -p output/data-no-skull-strip output/runs-focal-loss
cp output/data-hdbet/preprocessing_config.py output/data-no-skull-strip/preprocessing_config.py
cp output/data-hdbet/dataset_config.py output/data-no-skull-strip/dataset_config.py
cp output/runs-cross-entropy/train_config.py output/runs-focal-loss/train_config.py
```

随后修改新目录中的配置并将同一实验根传给对应脚本。不要直接修改
`output/data-hdbet` 或 `output/runs-cross-entropy` 中的迁移快照，以免配置与历史产物不再对应。

目录参数的解析规则：

- `--data-root output/data-hdbet`：需要数据集时使用 `output/data-hdbet/datasets`，需要 NIfTI 时使用 `output/data-hdbet/data`。
- 也可以直接传 `output/data-hdbet/datasets` 或 `output/data-hdbet/data`，但日常运行推荐统一传实验根。
- `--checkpoint-root output/runs-cross-entropy`：使用 `output/runs-cross-entropy/checkpoints`。
- 也可以直接传 `output/runs-cross-entropy/checkpoints`。
- 程序不会递归搜索或自动选择多个实验。如果 `output/` 下有多个 `data-*` 或 `runs-*`，必须明确指定其中一个实验根。
- 一个 checkpoint 根内部包含 `seq1_T1`、`seq2_T2`、`seq3_FLAIR` 三个目录是正常的一套融合实验。

### scripts 工具清单

| 脚本 | 作用 | 用法示例 |
|---|---|---|
| `preprocess_data.py` | 预处理三序列 MRI，生成 `data/`、图像和 metadata | `python -m scripts.preprocess_data --config output/data-hdbet/preprocessing_config.py --output-root output/data-hdbet` |
| `preprocess_mask.py` | 根据预处理 metadata 对齐医生 mask，并写入已有 `data/` | `python -m scripts.preprocess_mask --config output/data-hdbet/preprocessing_config.py --data-root output/data-hdbet` |
| `build_dataset_kfold.py` | 对共有病例执行分层 K-Fold 划分，生成 `datasets/` | `python -m scripts.build_dataset_kfold --config output/data-hdbet/dataset_config.py --data-root output/data-hdbet --output-root output/data-hdbet` |
| `check_dataset_kfold.py` | 读取各 fold 的 `.pt`，统计 train/val/test 类别分布 | `python -m scripts.check_dataset_kfold --data-root output/data-hdbet` |
| `check_preprocessed_data.py` | 按预处理配置检查 shape、spacing、文件大小和序列完整性 | `python -m scripts.check_preprocessed_data --config output/data-hdbet/preprocessing_config.py --data-root output/data-hdbet` |
| `check_file_sizes.py` | 不读取体素，快速列出小于阈值的 NIfTI | `python -m scripts.check_file_sizes --data-root output/data-hdbet --threshold-mb 1` |
| `delete_small_preprocessed_cases.py` | 按 case 清理异常小文件；默认仅 dry-run | `python -m scripts.delete_small_preprocessed_cases --data-root output/data-hdbet --threshold-mb 1` |
| `get_raw_data_info.py` | 只读统计原始 NIfTI 的 size、spacing 和物理尺寸 | `python -m scripts.get_raw_data_info --raw-root /path/to/raw` |

`deprecated_build_dataset.py` 和 `deprecated_check_dataset.py` 是无 K-Fold 的历史脚本，仅供追溯，不再维护或推荐执行。

### 数据预处理
数据预处理有三个步骤：
1. 重采样resample：统一输入图像的spacing,使像素物理意义一致
2. 归一化normalize：统一输入图像的像素值区间，执行Z-score标准化，使平均值为0，标准差为1。这样能消除设备差异，减小不同机器这个因素对模型的干扰
3. 裁剪/填充center_crop_or_pad：固定中心点，将重采样后的数据填充或裁剪至统一大小

数据预处理结束后，会在指定实验根目录下生成 `data`，包含类别子目录、`case_index.json`、`mask_index.json`（执行 mask 预处理后）和每个序列的预处理 metadata。每项数据的命名方式是：`case_$(case_id)_$(seq_id).nii.gz`。结构如下：
```txt
output/data-hdbet/data
├── 0_normal
│   ├── 1   # 属于正常类型、序列为T1WI的数据
│   ├── 2
│   └── 3
├── 1_inflammation
│   ├── 1
│   ├── 2   # 属于炎症类型、序列为T2WI的数据
│   └── 3
├── 2_metastasis
│   ├── 1
│   ├── 2
│   └── 3   # 属于脑膜转移类型、序列为FLAIR的数据
├── case_index.json
└── mask_index.json
```
执行数据预处理的方式是：
```bash
python -m scripts.preprocess_data \
    --config output/data-hdbet/preprocessing_config.py \
    --output-root output/data-hdbet
```

如果需要处理医生标注的mask数据，执行：
```bash
python -m scripts.preprocess_mask \
    --config output/data-hdbet/preprocessing_config.py \
    --data-root output/data-hdbet
```

### 构造数据集
本项目数据集被划分成了三个部分：训练集、验证集和测试集。其中，(训练集+验证集)占全部数据的80%，验证集占(训练集+验证集)的15%。为了提升测试结果的置信度，采用五折交叉验证的方法，将数据集分成五等份，每一份轮流当测试集，剩下的按17:3划分训练集和验证集。

数据集存放格式如下：
```txt
output/data-hdbet/datasets
├── seq1_T1
│   ├── fold1
│   │   ├── split.json
│   │   ├── test.pt
│   │   ├── train.pt
│   │   └── val.pt
│   ├── fold2
│   │   ├── split.json
│   │   ├── test.pt
│   │   ├── train.pt
│   │   └── val.pt
│   ├── fold3
│   │   ├── ...
│   ├── fold4
│   │   ├── ...
│   └── fold5
│       ├── ...
├── seq2_T2
│   ├── ...
└── seq3_FLAIR
    ├── ...
```

执行构造数据集的方式是：
```bash
python -m scripts.build_dataset_kfold \
    --config output/data-hdbet/dataset_config.py \
    --data-root output/data-hdbet \
    --output-root output/data-hdbet
```

### 模型训练

#### 模型结构
项目当前采用官方预训练的3D ResNet（`torchvision.models.video.r3d_18`）作为主要骨干网络，并针对3D MRI小样本场景做了以下改进：
1. 小 batch 场景下的归一化重设计（BN→IN）：针对 3D MRI 显存受限导致的极小 batch 训练不稳定，提出实例级归一化方案以提升收敛稳定性与泛化。在 BasicBlock 和 shortcut 中统一使用 InstanceNorm3d(affine=True)，减少小批量统计噪声问题。
```python
def _replace_bn3d_with_in3d(self, module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            inorm = nn.InstanceNorm3d(
                child.num_features,
                eps=child.eps,
                affine=True,
                track_running_stats=False,
            )
            # 可选：拷贝 affine 参数，减少突变
            with torch.no_grad():
                if child.weight is not None:
                    inorm.weight.copy_(child.weight)
                if child.bias is not None:
                    inorm.bias.copy_(child.bias)
            setattr(module, name, inorm)
        else:
            self._replace_bn3d_with_in3d(child)
```
2. 采用官方提供的预训练模型，预训练模型的结构是r3d_18，权重是R3D_18_Weights。迁移学习具有如下优势：
  - 提高模型性能：预训练模型已经学到了：“边缘、纹理、结构”和“空间 + 时间特征”。
  - 加快训练速度：一开始就在“比较合理的参数空间”。
  - 提高泛化能力：预训练模型是在大规模数据上训练的，相当于给模型加了“先验知识”。
```python
weights = R3D_18_Weights.DEFAULT if use_pretrained else None
model = r3d_18(weights=weights)
```
3. 实现异构晚期融合的多模态模型：seq1(T1)和seq2(T2)使用`FoundationModel_ori`纯分类模型，seq3(FLAIR)使用`FoundationModel`分类+分割双头模型。三个模型分别输出prob后取平均，实现软投票（decision-level soft voting）机制。这个设计来自实验结论：FLAIR mask 对 seq3 的局部病灶关注有帮助，但迁移到 seq1/seq2 后会干扰分类特征，因此 seq1/seq2 保持无 mask 训练流程。

整体流程如下：
```text
seq1 ──▶ FoundationModel_ori ──▶ prob1 ──────┐
                                               │
seq2 ──▶ FoundationModel_ori ──▶ prob2 ──────┼──▶ 软投票 ──▶ 最终分类结果
                                               │
seq3 ──▶ FoundationModel(双头) ──┬──▶ prob3 ──┘
                                  │
                                  └──▶ seg(分割结果)
```

#### 训练过程
训练过程引入patience机制，根据模型在验证集上的表现，决定是否早停。为了避免训练起始阶段收敛不稳定，从而异常早停，我们引入了最小训练轮数，在保护期内不触发早停。

由于本项目数据不均衡性明显（炎症数量远多于正常和脑膜转移），计算loss时采用了不平衡样本的“温和重加权损失”设计。
核心思想：少数类更高权重，缓解“多数类主导梯度”。
关键改造：对反比权重做 sqrt（幂指数 0.5）而不是直接用原始反比权重，避免少数类权重过大导致训练震荡。
效果定位：在不改模型结构的前提下，提高宏平均指标（macro-F1）与少数类召回。

训练相关命令如下：
```bash
python train_kfold.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --output-root output/runs-cross-entropy \
    --seq 1 --model FoundationModel_ori --fold 1

python train_kfold.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --output-root output/runs-cross-entropy \
    --seq 3 --model FoundationModel --fold 1
...
```

`CLASSIFICATION_LOSS` 可设为 `weighted_cross_entropy`（当前默认逻辑）、`cross_entropy` 或 `class_balanced_focal`。迁移后的历史 `.pt` 仍保存了原 `version1/data/...` 路径，但加载器会根据 `--data-root output/data-hdbet` 自动重定位到新的 `data/`。

### 模型测试
模型测试的核心评估指标有这些：

1. Test Loss：CrossEntropyLoss 在测试集上的平均值（avg_loss）。
2. Accuracy：整体分类正确率。
3. Precision (macro)：宏平均精确率，各类别等权。
4. Recall (macro)：宏平均召回率，各类别等权。
5. F1-score (macro)：宏平均 F1，兼顾 precision 与 recall。
6. Confusion Matrix：混淆矩阵，查看各类别之间的误分布。
7. Classification Report：每个类别的 precision/recall/F1/support 明细（sklearn）。

另外在 K-Fold 多折评估 时，还会统计：各指标的 Mean ± Std（均值和标准差），用于衡量跨折稳定性。

执行测试：
```bash
python eval_kfold.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-cross-entropy \
    --seq 1 --model FoundationModel_ori --fold 1

# 不指定fold，评估所有fold的平均性能
python eval_kfold.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-cross-entropy \
    --seq 1 --model FoundationModel_ori

# eval_kfold 省略这三个参数时，默认使用迁移后的 data-hdbet 和 runs-cross-entropy
python eval_kfold.py --seq 1 --model FoundationModel_ori

# 当前最优异构晚期融合软投票评估
python eval.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-cross-entropy

# 训练受约束的 cross-fitted OOF 晚期融合器。
# 每个目标 fold 使用另外四折的 OOF-test 概率选择三个模态权重和转移阈值；
# normal gate 固定为等权融合。结果写入 reports/constrained_oof。
python train_meta_fusion.py \
    --config output/runs-meta-linear/meta_fusion_config.py \
    --data-root output/data-hdbet \
    --checkpoint-roots \
        output/runs-cross-entropy \
        output/runs-cross-entropy \
        output/runs-cross-entropy \
    --output-root output/runs-meta-linear

# 只报告 fold 1；由于训练来源是另外四折，仍会生成所有缺失的 test 概率缓存。
python train_meta_fusion.py \
    --config output/runs-meta-linear/meta_fusion_config.py \
    --data-root output/data-hdbet \
    --checkpoint-roots \
        output/runs-cross-entropy \
        output/runs-cross-entropy \
        output/runs-cross-entropy \
    --output-root output/runs-meta-linear \
    --fold 1
```

## 使用的数据和最佳的结果

### 数据情况
本项目的数据集取自3728例T1WI、T2WI、FLAIR对齐的脑部MRI图像。数据集的划分以及各类别的情况如下：
```text

--- fold1 ---
  Split    | normal(0)       | meningitis(1)   | encephalitis(2) | metastasis(3)   | Total   
  ---------------------------------------------------------------------------------------------
  train    | 228             | 226             | 1606            | 474             | 2534    
  val      | 48              | 48              | 268             | 84              | 448     
  test     | 89              | 71              | 463             | 123             | 746     

--- fold2 ---
  Split    | normal(0)       | meningitis(1)   | encephalitis(2) | metastasis(3)   | Total   
  ---------------------------------------------------------------------------------------------
  train    | 251             | 221             | 1596            | 466             | 2534    
  val      | 52              | 48              | 267             | 81              | 448     
  test     | 62              | 76              | 474             | 134             | 746     

--- fold3 ---
  Split    | normal(0)       | meningitis(1)   | encephalitis(2) | metastasis(3)   | Total   
  ---------------------------------------------------------------------------------------------
  train    | 244             | 232             | 1594            | 464             | 2534    
  val      | 50              | 50              | 268             | 80              | 448     
  test     | 71              | 63              | 475             | 137             | 746     

--- fold4 ---
  Split    | normal(0)       | meningitis(1)   | encephalitis(2) | metastasis(3)   | Total   
  ---------------------------------------------------------------------------------------------
  train    | 232             | 231             | 1595            | 477             | 2535    
  val      | 48              | 49              | 275             | 76              | 448     
  test     | 85              | 65              | 467             | 128             | 745     

--- fold5 ---
  Split    | normal(0)       | meningitis(1)   | encephalitis(2) | metastasis(3)   | Total   
  ---------------------------------------------------------------------------------------------
  train    | 255             | 226             | 1602            | 452             | 2535    
  val      | 52              | 49              | 277             | 70              | 448     
  test     | 58              | 70              | 458             | 159             | 745 
```

### 最优模型

目前在三个序列上分别用“迁移学习”的方式训练单模态模型，其中seq1/seq2采用`FoundationModel_ori`，seq3采用带mask辅助分割头的`FoundationModel`。最优模型是将三个模型的输出软投票，得到的异构多模态晚期融合模型。以下结果已随历史产物迁移到 `output/runs-cross-entropy/output_texts/`：

以下命令行是重构前生成该历史结果时的执行记录；当前 `eval.py` 需要显式配置和路径参数。

```bash
(BrainMRIClassification) ailab@ailab:~/projects/brain_mri_classification_with_segmentation$ python eval.py
```

| Fold | Test samples | Accuracy | Precision(macro) | Recall(macro) | F1-score(macro) | Metastasis recall |
|------|--------------|----------|------------------|---------------|-----------------|-------------------|
| 1 | 746 | 0.9008 | 0.9080 | 0.8291 | 0.8616 | 0.6412 |
| 2 | 746 | 0.9263 | 0.9042 | 0.8886 | 0.8922 | 0.7143 |
| 3 | 745 | 0.9168 | 0.9093 | 0.8778 | 0.8916 | 0.7254 |
| 4 | 745 | 0.9235 | 0.9121 | 0.8698 | 0.8894 | 0.7724 |
| 5 | 745 | 0.8886 | 0.9041 | 0.8180 | 0.8538 | 0.6415 |

5折平均结果如下：

```text
Method        : Late Fusion Soft Voting (Heterogeneous Ensemble)
Models        : Seq1/Seq2=FoundationModel_ori, Seq3=FoundationModel
----------------------------------------
Metric          | Mean       | Std
----------------------------------------
Accuracy        | 0.9112     | ±0.0143
Precision       | 0.9076     | ±0.0031
Recall          | 0.8567     | ±0.0279
F1-Score        | 0.8777     | ±0.0166
----------------------------------------
```

各fold混淆矩阵如下，类别顺序为`normal`、`inflammation`、`metastasis`：

```text
Fold 1:
[[ 78  11   0]
 [  4 510  12]
 [  0  47  84]]

Fold 2:
[[ 61   1   0]
 [  7 540  11]
 [  1  35  90]]

Fold 3:
[[ 67   4   0]
 [  2 513  17]
 [  2  37 103]]

Fold 4:
[[ 74  10   1]
 [  6 519  12]
 [  0  28  95]]

Fold 5:
[[ 49   9   0]
 [  1 511  16]
 [  1  56 102]]
```

从内部测试看，当前异构晚期融合模型的整体准确率和宏平均F1较高，但metastasis recall仍然是瓶颈。五个fold中，脑膜转移的召回率分别是0.6412、0.7143、0.7254、0.7724、0.6415；主要错误仍然是将脑膜转移误判为炎症。

### 外部验证结果与泛化分析

外部验证使用`external_eval.py`。脚本现在直接复用内部预处理入口及指定的
`preprocessing_config.py`，包括 HD-BET、前景强度归一化、前景中心裁剪/填充和相同的
`[Z, Y, X]` 模型轴顺序，再使用当前异构模型做三序列软投票。输出诊断 CSV 包含每个病例、
每个 fold、每个序列模型的三类概率，以及 fold 内投票和跨 fold 平均后的预测结果。

```bash
python external_eval.py \
    --preprocess-config output/data-hdbet/preprocessing_config.py \
    --checkpoint-root output/runs-cross-entropy \
    --output-root output/runs-cross-entropy \
    --data-root /path/to/external_cases \
    --label 2
```

下面两行是修复前旧脚本产生的历史记录。旧脚本没有完整复用 HD-BET 预处理，而且输入轴顺序
与内部数据加载器不一致，因此这些数值不能作为修复后流程的外部基线，也不能据此判断真实域偏移；
在相同外部数据上重新运行上述命令后再更新结论。

| 外部数据类别 | 样本数 | Accuracy / Recall | 预测为normal | 预测为inflammation | 预测为metastasis |
|-------------|--------|-------------------|--------------|--------------------|------------------|
| metastasis | 33 | 39.39% | 0 | 20 | 13 |
| inflammation | 45 | 93.33% | 0 | 42 | 3 |

待修复后的外部结果产生后，可以依次检查以下可能因素：

1. 内部K-fold验证来自同一数据源和同一采集/标注分布，训练集和测试集虽然按case划分，但仍属于同分布验证。外部数据来自另一批病例，扫描协议、设备、层厚、FOV、增强时相、病灶形态和诊断边界都可能发生变化，因此内部K-fold结果不能直接等价于外部泛化能力。
2. 训练数据中炎症样本显著多于转移瘤样本。虽然训练时使用了sqrt反比类别权重，并用macro-F1作为早停指标，但内部测试已经显示metastasis recall明显低于normal和inflammation。外部数据只要稍微偏离内部转移瘤分布，模型就更容易回退到数量更多、形态覆盖更广的炎症类。
3. 即使预处理流程已经一致，也不能消除真实域差异；仍需要结合修复后的 per-case/per-fold/per-sequence 概率判断偏差来源。
4. 当前mask辅助只作用于seq3/FLAIR，seq1和seq2采用不带mask的旧分类模型。外部域偏移下，三个序列等权soft voting不一定仍是最优组合；如果seq1/seq2在外部数据上更偏炎症，等权平均可能抵消seq3分割辅助带来的转移瘤证据。
5. 外部标签只用于独立评估时，不应直接在同一批外部测试数据上训练融合权重或选择阈值；如需外部域适配，应另外划分校准/微调集。

后续可以优先做以下分析和改进：

1. 优先查看诊断CSV，确认外部metastasis错例是所有分支都偏炎症，还是某个序列/某个fold主导了偏差。
2. 单独统计外部metastasis病例的metastasis概率分布，判断这些病例是完全被炎症压制，还是接近分类阈值。
3. 在独立校准集上尝试温度校准、类别阈值调整、加权 soft voting，或以 metastasis recall 为优先目标选择融合权重。
4. 如果能获得少量外部标注数据，优先做小规模域适配或微调，而不是只依赖内部K-fold结果。

## 新特性：基于掩码（Mask）特征辅助的多任务学习网络

在最新的迭代中，网络已由单分类任务演进为**分类与分割并行的多任务学习（Multi-task Learning）架构**。通过引入病灶区域的像素级标注（Mask），网络得以在训练期间学习到更聚焦的局部结构与轮廓特征，从机制上来讲，这能有效提升特征提取的质量并反哺主分类任务表现。

### 双头网络设计 (Dual-head Architecture)
目前的模型 `FoundationModel` 主要包含三个模块：
1. **共享主干（Backbone）**：以3D ResNet（`official_r3d18` 等）作为骨干网络提取包含三维空间维度的深层特征图 `[B, C, D, H, W]`。
2. **分类主分支（Classification Head）**：对共享特征图进行 3D 全局平均池化，随后展平传入全连接层进行全局的三分类（正常、炎症、转移瘤）。
3. **分割辅助分支（Segmentation Head）**：由轻量级的 3D 卷积层和实例归一化（InstanceNorm3d）组建。该分支进一步提取病灶轮廓特征，并利用三线性插值（Trilinear Interpolation）将深层特征图等比放大回输入图像的原始尺寸，实现像素级输出。

### 混合掩码数据训练策略
在现实的医学影像场景下，获取所有数据的精细像素级 Mask 成本极高。因此，我们的训练机制进行了特别设计，以**兼容含 Mask 与无 Mask 的混合数据集**：
- 对于**有 Mask 的案例**：计算分类损失（如 CrossEntropyLoss）的同时计算分割损失（如 Dice Loss 或 Pixel-wise CE），联合更新参数。
- 对于**无 Mask 的案例**：仅执行分类分支的损失计算与反向传播。

在模型执行推理或常规测试时，可通过控制参数 `return_seg=False` 动态关闭分割头的前向传播计算，从而保证分类场景下的推理速度与原来保持一致，无带来额外的显存/推理时间开销。
