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
self.bn1 = nn.InstanceNorm3d(planes, affine=True)
self.bn2 = nn.InstanceNorm3d(planes, affine=True)
...
nn.InstanceNorm3d(self.expansion * planes, affine=True)
```
2. 保细节的输入 stem 设计（3×3×3, stride=1）：为避免早期下采样破坏小病灶/薄层结构，采用高分辨率 stem 保留解剖细节。把标准 ResNet 的激进下采样 stem 改成了不降采样的 3D 卷积，且没有 early maxpool。
```python
self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
```
3. 实现晚期融合的多模态模型：先单独训练三个序列的分类模型，每个模型最后会输出一个prob，将这个prob取平均值，实现软投票（decision-level soft voting）机制。由于不同序列容易提取的特征有所不同，晚期融合使每个分支模型重点学习对应序列特征（专家模型），避免单模型里多模态通道竞争导致的特征稀释。该策略在保留模态特异性判别能力的同时，利用跨模态互补信息提升了分类鲁棒性与泛化性能。

完整结构如下：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet 的基本残差块 (Basic Residual Block)
    包含两个 3x3x3 卷积层和一条 Shortcut 连接
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一层卷积：如果 stride > 1，在这里进行下采样
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 使用 InstanceNorm3d 代替 BatchNorm3d
        # 原因：你的 Batch Size = 2，BN 的统计量会非常不稳定，导致训练困难。
        # IN 对 Batch Size 不敏感，是 3D 医学图像任务的首选。
        # affine=True 让 IN 层拥有可学习的参数 (gamma, beta)
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        
        # 第二层卷积：保持尺寸不变
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.InstanceNorm3d(planes, affine=True)

        # Shortcut (跳跃连接)
        # 如果输入输出维度不一致（stride!=1 或通道数改变），需要用 1x1 卷积调整 x 的形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.InstanceNorm3d(self.expansion * planes, affine=True)
            )

    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 残差连接：F(x) + x
        # 这让梯度可以直接流向浅层，解决了深层网络难以训练的问题
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, in_channels=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # ================== 初始层 ==================
        # 标准 ResNet 使用 7x7 stride=2，但对于 MRI（特别是 Z 轴层数较少时），
        # 过早的下采样会丢失细节。这里改用 3x3 stride=1 保留分辨率。
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.InstanceNorm3d(64, affine=True)
        
        # ================== 残差层 (Layer 1-4) ==================
        # Layer 1: 64通道，不降采样
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # Layer 2: 128通道，降采样 (stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # Layer 3: 256通道，降采样 (stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # Layer 4: 512通道，降采样 (stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ================== 分类头 ==================
        # 全局平均池化：无论输入尺寸多大，都压缩成 1x1x1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out) # [B, 64, D, H, W]
        out = self.layer2(out) # [B, 128, D/2, H/2, W/2]
        out = self.layer3(out) # [B, 256, D/4, H/4, W/4]
        out = self.layer4(out) # [B, 512, D/8, H/8, W/8]

        out = self.avgpool(out) # [B, 512, 1, 1, 1]
        out = out.flatten(1)    # [B, 512]
        
        return self.fc(out)


# ================== 快捷入口 ==================

def ResNet10(num_classes=2, in_channels=1):
    """
    ResNet-10: 较浅的网络，适合数据量较少的情况
    结构: [1, 1, 1, 1] 个 Block
    """
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, in_channels=in_channels)

def ResNet18(num_classes=2, in_channels=1):
    """
    ResNet-18: 标准轻量级 ResNet
    结构: [2, 2, 2, 2] 个 Block
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

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

目前在三个序列上分别训练了单模态改良后ResNet10模型，最优模型是将三个模型的输出软投票，得到的多模态晚期融合模型。模型在测试集上的表现如下：
```bash
>>> Starting K-Fold Evaluation for: Late Fusion Soft Voting <<<
Mode: All 5 Folds Average

==================== Evaluating Fold 1 ====================
Mode: Late Fusion (Soft Voting) | Model: ResNet
  -> Loading test sets for all 3 sequences... Done in 72.3s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 1)
Test samples  : 746
Accuracy      : 0.8566
Precision     : 0.8255
Recall        : 0.7745
F1-score      : 0.7892

Confusion Matrix:
[[ 77  10   2]
 [ 17 494  15]
 [  2  61  68]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.8021    0.8652    0.8324        89
inflammation     0.8743    0.9392    0.9056       526
  metastasis     0.8000    0.5191    0.6296       131

    accuracy                         0.8566       746
   macro avg     0.8255    0.7745    0.7892       746
weighted avg     0.8527    0.8566    0.8484       746


===== Misclassified Cases =====
Total misclassified: 107
CaseID: 0015 | GT: 0 | Pred: 1
CaseID: 0033 | GT: 0 | Pred: 2
CaseID: 0215 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 2 ====================
Mode: Late Fusion (Soft Voting) | Model: ResNet
  -> Loading test sets for all 3 sequences... Done in 91.2s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 2)
Test samples  : 746
Accuracy      : 0.8794
Precision     : 0.8313
Recall        : 0.8206
F1-score      : 0.8225

Confusion Matrix:
[[ 58   4   0]
 [  7 524  27]
 [  3  49  74]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.8529    0.9355    0.8923        62
inflammation     0.9081    0.9391    0.9233       558
  metastasis     0.7327    0.5873    0.6520       126

    accuracy                         0.8794       746
   macro avg     0.8313    0.8206    0.8225       746
weighted avg     0.8739    0.8794    0.8749       746


===== Misclassified Cases =====
Total misclassified: 90
CaseID: 0136 | GT: 0 | Pred: 1
CaseID: 0164 | GT: 0 | Pred: 1
CaseID: 0169 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 3 ====================
Mode: Late Fusion (Soft Voting) | Model: ResNet
  -> Loading test sets for all 3 sequences... Done in 85.8s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 3)
Test samples  : 745
Accuracy      : 0.8564
Precision     : 0.8163
Recall        : 0.8220
F1-score      : 0.8184

Confusion Matrix:
[[ 67   3   1]
 [  6 484  42]
 [  3  52  87]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.8816    0.9437    0.9116        71
inflammation     0.8980    0.9098    0.9038       532
  metastasis     0.6692    0.6127    0.6397       142

    accuracy                         0.8564       745
   macro avg     0.8163    0.8220    0.8184       745
weighted avg     0.8528    0.8564    0.8542       745


===== Misclassified Cases =====
Total misclassified: 107
CaseID: 0007 | GT: 0 | Pred: 1
CaseID: 0129 | GT: 0 | Pred: 2
CaseID: 0143 | GT: 0 | Pred: 1
...

==================== Evaluating Fold 4 ====================
Mode: Late Fusion (Soft Voting) | Model: ResNet
  -> Loading test sets for all 3 sequences... Done in 56.3s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 4)
Test samples  : 745
Accuracy      : 0.8779
Precision     : 0.8422
Recall        : 0.8153
F1-score      : 0.8279

Confusion Matrix:
[[ 74   8   3]
 [  8 501  28]
 [  0  44  79]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9024    0.8706    0.8862        85
inflammation     0.9060    0.9330    0.9193       537
  metastasis     0.7182    0.6423    0.6781       123

    accuracy                         0.8779       745
   macro avg     0.8422    0.8153    0.8279       745
weighted avg     0.8746    0.8779    0.8757       745


===== Misclassified Cases =====
Total misclassified: 91
CaseID: 0028 | GT: 0 | Pred: 2
CaseID: 0056 | GT: 0 | Pred: 1
CaseID: 0078 | GT: 0 | Pred: 2
...

==================== Evaluating Fold 5 ====================
Mode: Late Fusion (Soft Voting) | Model: ResNet
  -> Loading test sets for all 3 sequences... Done in 44.7s
  -> Successfully loaded 3 models (T1, T2, FLAIR).

===== Test Results =====
Sequence      : ALL (Soft Voting) (Fold 5)
Test samples  : 745
Accuracy      : 0.8456
Precision     : 0.8419
Recall        : 0.7781
F1-score      : 0.8059

Confusion Matrix:
[[ 48  10   0]
 [  2 490  36]
 [  1  66  92]]

Classification Report:
              precision    recall  f1-score   support

      normal     0.9412    0.8276    0.8807        58
inflammation     0.8657    0.9280    0.8958       528
  metastasis     0.7188    0.5786    0.6411       159

    accuracy                         0.8456       745
   macro avg     0.8419    0.7781    0.8059       745
weighted avg     0.8402    0.8456    0.8403       745


===== Misclassified Cases =====
Total misclassified: 115
CaseID: 0017 | GT: 0 | Pred: 1
CaseID: 0051 | GT: 0 | Pred: 1
CaseID: 0123 | GT: 0 | Pred: 1
...

==================================================
   K-FOLDS AVERAGE REPORT (5 folds)   
==================================================
Method        : Late Fusion Soft Voting
Model         : ResNet
----------------------------------------
Metric          | Mean       | Std       
----------------------------------------
Accuracy        | 0.8632     | ±0.0132
Precision       | 0.8314     | ±0.0099
Recall          | 0.8021     | ±0.0212
F1-Score        | 0.8128     | ±0.0138
----------------------------------------
```
