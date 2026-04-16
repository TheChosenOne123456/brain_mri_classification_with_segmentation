# 脑部MRI分类深度学习模型

## 项目概览
本项目是一个基于3维MRI图像的分类任务，旨在帮助影像科医生区分脑膜疾病。具体而言，有三种病症：脑膜炎、脑炎和脑膜转移（转移瘤）。我们需要模型根据输入的MRI图像，给出分类结果。加上正常的案例，一共有四个类别。实际应用中，由于脑炎和脑膜炎同属炎症，区分起来很困难，且区分炎症和肿瘤更有医学上的意义，所以目前将数据中的脑膜炎和脑炎合并为炎症类，任务简化为三分类问题。

## 项目结构
```text
.
├── configs
│   ├── global_config.py
│   ├── __init__.py
│   ├── __pycache__
│   └── train_config.py
├── environment.yml
├── eval_command.sh
├── eval_kfold.py
├── eval.py
├── eval_vote_kfold.py
├── Helper
│   ├── eval_pt.py
│   ├── pick_cases.py
│   ├── pick_easy_cases.py
│   └── __pycache__
├── infer_kfold.py
├── infer.py
├── models
│   ├── cnn3d.py
│   ├── __pycache__
│   └── ResNet.py
├── quick_command.sh
├── read_kfold_pth.py
├── README.md
├── scripts
│   ├── build_dataset_kfold.py
│   ├── build_dataset.py
│   ├── check_dataset_kfold.py
│   ├── check_dataset.py
│   ├── preprocess_data.py
│   └── __pycache__
├── train_command.sh
├── train_kfold.py
├── train.py
├── utils
│   ├── data_scan.py
│   ├── dataset.py
│   ├── __init__.py
│   ├── intensity.py
│   ├── io.py
│   ├── __pycache__
│   ├── resample.py
│   ├── sequences.py
│   ├── spatial.py
│   └── train_and_test.py
├── version1
│   ├── checkpoints
│   ├── data
│   ├── datasets
│   └── output_texts
└── version2
    ├── checkpoints
    ├── data
    ├── datasets
    └── output_texts
```

项目主要子文件和子文件夹意义如下：
1. configs：包含总体配置，如类别数量和名称（normal、inflammation、metastasis）、序列数量和名称（T1WI、T2WI、FLAIR）等全局信息；和训练配置，包含训练有关的超参数
2. models：模型的实现
3. utils：一些工具的实现
4. scripts：必要的脚本实现，包含“数据预处理”和“训练集、验证集和测试集的生成”
5. train_kfold.py：训练脚本，配合k折交叉验证使用
6. eval_kfold.py：测试脚本，先计算每一折的结果，再综合评估
7. eval_vote_kfold.py：晚期多通道融合模型的测试
8. infer_kfold.py：预测脚本，支持单个图像的输入。实际投入使用的时候再实现
9. versionx：存放当前版本的data（预处理后的数据）、dataset（训练集、验证集、测试集）、checkpoints（模型参数）和output_texts（记录测试输出）
10. environment.yml：项目用的conda虚拟环境

## 实验过程

### 数据预处理
数据预处理有三个步骤：
1. 重采样resample：统一输入图像的spacing,使像素物理意义一致
2. 归一化normalize：统一输入图像的像素值区间，执行Z-score标准化，使平均值为0，标准差为1。这样能消除设备差异，减小不同机器这个因素对模型的干扰
3. 裁剪/填充center_crop_or_pad：固定中心点，将重采样后的数据填充或裁剪至统一大小

数据预处理结束后，会在对应版本文件夹（如version1）下生产data目录，该目录下有3个类别子目录，以及一个图像编号索引文件case_index.json。每个类别子目录下，分别有所有序列的子目录。本项目一共有3个序列，它们是严格对齐的。每项数据的命名方式是：case_$(case_id)_$(seq_id).nii.gz。data的结构如下：
```txt
version1/data
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
└── case_index.json
```
执行数据预处理的方式是：
```python
python -m scripts.preprocess_data
```

### 构造数据集
本项目数据集被划分成了三个部分：训练集、验证集和测试集。其中，(训练集+验证集)占全部数据的80%，验证集占(训练集+验证集)的15%。为了提升测试结果的置信度，采用五折交叉验证的方法，将数据集分成五等份，每一份轮流当测试集，剩下的按17:3划分训练集和验证集。

数据集存放格式如下：
```txt
version1/datasets
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
```python
python -m scripts.build_dataset_kfold
```

### 模型训练

#### 模型结构
项目采用的模型总体是ResNet10，但是在ResNet10的基础上有一些改进：
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
3. 实现晚期融合的多模态模型：先单独训练三个序列的分类模型，每个模型最后会输出一个prob，将这个prob取平均值，实现软投票（decision-level soft voting）机制。由于不同序列容易提取的特征有所不同，晚期融合使每个分支模型重点学习对应序列特征（专家模型），避免单模型里多模态通道竞争导致的特征稀释。该策略在保留模态特异性判别能力的同时，利用跨模态互补信息提升了分类鲁棒性与泛化性能。

完整结构如下：
```python
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


# 固定在模型文件内的配置（非构造参数）
FOUNDATION_BACKBONE_NAME = "official_r3d18"
FOUNDATION_USE_PRETRAINED = True
FOUNDATION_DROPOUT = 0.0


class FoundationModel(nn.Module):
    """
    可扩展“壳子”：
    - backbone: 当前仅支持 official_r3d18（官方预训练）
    - neck: 预留（默认 Identity）
    - head: 分类头（可替换）
    - aux_heads: 预留多任务/蒸馏/对比学习头
    """
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
    ):
        super().__init__()
        self.backbone_name = FOUNDATION_BACKBONE_NAME
        self.in_channels = in_channels

        self.backbone, feat_dim = self._build_backbone(
            FOUNDATION_BACKBONE_NAME,
            FOUNDATION_USE_PRETRAINED,
            in_channels
        )
        self.neck = self._build_neck(feat_dim)   # 预留创新点1
        self.head = self._build_head(feat_dim, num_classes, FOUNDATION_DROPOUT)
        self.aux_heads = nn.ModuleDict()         # 预留创新点2

    # ===== 可扩展点：backbone =====
    def _build_backbone(self, backbone_name: str, use_pretrained: bool, in_channels: int):
        if backbone_name == "official_r3d18":
            weights = R3D_18_Weights.DEFAULT if use_pretrained else None
            model = r3d_18(weights=weights)

            # [新增] 仅做 BN -> IN
            self._replace_bn3d_with_in3d(model)

            # 适配输入通道（MRI 常见 1 通道）
            if in_channels == 1:
                model.stem[0] = self._adapt_first_conv_to_1ch(model.stem[0])
            elif in_channels != 3:
                raise ValueError("official_r3d18 仅支持 in_channels=1 或 3")

            feat_dim = model.fc.in_features
            model.fc = nn.Identity()  # 把分类头拆掉，交给壳子自己的 head
            return model, feat_dim

        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
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

    # ===== 可扩展点：neck =====
    def _build_neck(self, feat_dim: int):
        # 后续可替换为 attention / adapter / projector
        return nn.Identity()

    # ===== 可扩展点：head =====
    def _build_head(self, feat_dim: int, num_classes: int, dropout: float):
        if dropout > 0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        return nn.Linear(feat_dim, num_classes)

    def _adapt_first_conv_to_1ch(self, conv3d: nn.Conv3d):
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            bias=(conv3d.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv3d.weight.mean(dim=1, keepdim=True))
            if conv3d.bias is not None:
                new_conv.bias.copy_(conv3d.bias)
        return new_conv

    def forward_features(self, x):
        feat = self.backbone(x)   # [B, feat_dim]
        feat = self.neck(feat)
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits
```

#### 训练过程
训练过程引入patience机制，根据模型在验证集上的表现，决定是否早停。为了避免训练起始阶段收敛不稳定，从而异常早停，我们引入了最小训练轮数，在保护期内不触发早停。

由于本项目数据不均衡性明显（炎症数量远多于正常和脑膜转移），计算loss时采用了不平衡样本的“温和重加权损失”设计。
核心思想：少数类更高权重，缓解“多数类主导梯度”。
关键改造：对反比权重做 sqrt（幂指数 0.5）而不是直接用原始反比权重，避免少数类权重过大导致训练震荡。
效果定位：在不改模型结构的前提下，提高宏平均指标（macro-F1）与少数类召回。

训练相关命令如下：
```python
python train_kfold.py --seq 1 --model ResNet --fold 1
python train_kfold.py --seq 1 --model ResNet --fold 2
python train_kfold.py --seq 1 --model ResNet --fold 3
python train_kfold.py --seq 1 --model ResNet --fold 4
python train_kfold.py --seq 1 --model ResNet --fold 5
...
```

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
```python
python eval_kfold.py --seq 1 --model ResNet --fold 1
# 不指定fold，评估所有fold的平均性能
python eval_kfold.py --seq 1 --model ResNet
python eval_kfold.py --seq 2 --model ResNet
python eval_kfold.py --seq 3 --model ResNet
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

目前在三个序列上分别用“迁移学习”的方式训练了单模态预训练ResNet10模型（FoundationModel），最优模型是将三个模型的输出软投票，得到的多模态晚期融合模型。模型在测试集上的表现如下：
```bash
(BrainMRIClassification) ailab@ailab:~/projects/brain_mri_classification$ python eval_vote_kfold.py --model FoundationMode
l

>>> Starting K-Fold Evaluation for: Late Fusion Soft Voting <<<
Mode: All 5 Folds Average

==================== Evaluating Fold 1 ====================
Mode: Late Fusion (Soft Voting) | Model: FoundationModel
  -> Loading test sets for all 3 sequences... Done in 95.4s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 1)
Test samples  : 746
Accuracy      : 0.9048
Precision     : 0.9063
Recall        : 0.8424
F1-score      : 0.8706

Confusion Matrix:
[[ 78  11   0]
 [  3 507  16]
 [  0  41  90]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9630    0.8764    0.9176        89
inflammation     0.9070    0.9639    0.9346       526
  metastasis     0.8491    0.6870    0.7595       131

    accuracy                         0.9048       746
   macro avg     0.9063    0.8424    0.8706       746
weighted avg     0.9035    0.9048    0.9018       746


===== Misclassified Cases =====
Total misclassified: 71
CaseID: 0015 | GT: 0 | Pred: 1
CaseID: 0062 | GT: 0 | Pred: 1
CaseID: 0219 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 2 ====================
Mode: Late Fusion (Soft Voting) | Model: FoundationModel
  -> Loading test sets for all 3 sequences... Done in 137.9s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 2)
Test samples  : 746
Accuracy      : 0.9196
Precision     : 0.8969
Recall        : 0.8768
F1-score      : 0.8834

Confusion Matrix:
[[ 60   2   0]
 [  6 538  14]
 [  1  37  88]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.8955    0.9677    0.9302        62
inflammation     0.9324    0.9642    0.9480       558
  metastasis     0.8627    0.6984    0.7719       126

    accuracy                         0.9196       746
   macro avg     0.8969    0.8768    0.8834       746
weighted avg     0.9176    0.9196    0.9168       746


===== Misclassified Cases =====
Total misclassified: 60
CaseID: 0136 | GT: 0 | Pred: 1
CaseID: 0164 | GT: 0 | Pred: 1
CaseID: 0676 | GT: 1 | Pred: 2
...

==================== Evaluating Fold 3 ====================
Mode: Late Fusion (Soft Voting) | Model: FoundationModel
  -> Loading test sets for all 3 sequences... Done in 157.8s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 3)
Test samples  : 745
Accuracy      : 0.9074
Precision     : 0.9048
Recall        : 0.8590
F1-score      : 0.8788

Confusion Matrix:
[[ 66   5   0]
 [  1 513  18]
 [  2  43  97]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9565    0.9296    0.9429        71
inflammation     0.9144    0.9643    0.9387       532
  metastasis     0.8435    0.6831    0.7549       142

    accuracy                         0.9074       745
   macro avg     0.9048    0.8590    0.8788       745
weighted avg     0.9049    0.9074    0.9041       745


===== Misclassified Cases =====
Total misclassified: 69
CaseID: 0007 | GT: 0 | Pred: 1
CaseID: 0084 | GT: 0 | Pred: 1
CaseID: 0129 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 4 ====================
Mode: Late Fusion (Soft Voting) | Model: FoundationModel
  -> Loading test sets for all 3 sequences... Done in 144.4s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 4)
Test samples  : 745
Accuracy      : 0.9195
Precision     : 0.9208
Recall        : 0.8521
F1-score      : 0.8827

Confusion Matrix:
[[ 73  11   1]
 [  3 523  11]
 [  0  34  89]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9605    0.8588    0.9068        85
inflammation     0.9208    0.9739    0.9466       537
  metastasis     0.8812    0.7236    0.7946       123

    accuracy                         0.9195       745
   macro avg     0.9208    0.8521    0.8827       745
weighted avg     0.9188    0.9195    0.9170       745


===== Misclassified Cases =====
Total misclassified: 60
CaseID: 0028 | GT: 0 | Pred: 2
CaseID: 0078 | GT: 0 | Pred: 1
CaseID: 0118 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 5 ====================
Mode: Late Fusion (Soft Voting) | Model: FoundationModel
  -> Loading test sets for all 3 sequences... Done in 155.8s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 5)
Test samples  : 745
Accuracy      : 0.8832
Precision     : 0.9134
Recall        : 0.8023
F1-score      : 0.8450

Confusion Matrix:
[[ 49   9   0]
 [  0 516  12]
 [  1  65  93]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9800    0.8448    0.9074        58
inflammation     0.8746    0.9773    0.9231       528
  metastasis     0.8857    0.5849    0.7045       159

    accuracy                         0.8832       745
   macro avg     0.9134    0.8023    0.8450       745
weighted avg     0.8852    0.8832    0.8752       745


===== Misclassified Cases =====
Total misclassified: 87
CaseID: 0017 | GT: 0 | Pred: 1
CaseID: 0051 | GT: 0 | Pred: 1
CaseID: 0123 | GT: 0 | Pred: 1
...

==================================================
   K-FOLDS AVERAGE REPORT (5 folds)   
==================================================
Method        : Late Fusion Soft Voting
Model         : FoundationModel
----------------------------------------
Metric          | Mean       | Std       
----------------------------------------
Accuracy        | 0.9069     | ±0.0133
Precision       | 0.9085     | ±0.0081
Recall          | 0.8465     | ±0.0248
F1-Score        | 0.8721     | ±0.0143
----------------------------------------
```

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