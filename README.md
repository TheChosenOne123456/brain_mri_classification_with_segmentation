# 脑部MRI分类深度学习模型

## 项目概览
本项目是一个基于3维MRI图像的分类任务，旨在帮助影像科医生区分脑膜疾病。具体而言，有三种病症：脑膜炎、脑炎和脑膜转移（转移瘤）。我们需要模型根据输入的MRI图像，给出分类结果。加上正常的案例，一共有四个类别。实际应用中，由于脑炎和脑膜炎同属炎症，区分起来很困难，且区分炎症和肿瘤更有医学上的意义，所以目前将数据中的脑膜炎和脑炎合并为炎症类，任务简化为三分类问题。

项目使用 T1WI、T2WI、FLAIR 三种对齐序列。早期设想是将 FLAIR 序列上的医生标注 mask 迁移到 seq1、seq2、seq3，统一使用分类+分割双头模型训练；但实验发现，seq3 的 mask 辅助对 seq1 和 seq2 的分类效果不是正向收益。历史固定参照仍是 HD-BET 数据上的异构晚期融合；当前内部测试最佳工作点已经转为 SynthStrip 预处理管线：seq1、seq2 使用 `FoundationModel_ori`，seq3 在 `FoundationModel` 分类概率上叠加 OOF nnU-Net 病灶引导，再执行三序列等权软投票。最终病灶分割使用原生 nnU-Net，而不是 Foundation 的轻量分割头。

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
│   ├── FoundationModelHierarchical.py
│   ├── FoundationModelLesionAwareHierarchical.py
│   ├── model_factory.py
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
│   ├── runs-cross-entropy
│       ├── train_config.py
│       ├── checkpoints
│       └── output_texts
│   └── runs-hierarchical
│       └── train_config.py
```

项目主要子文件和子文件夹意义如下：
1. configs：`global_config.py` 只保存类别、序列、随机种子、K-Fold 和原始数据路径等跨阶段设置；`config_utils.py` 负责加载实验目录中的阶段配置并解析标准产物目录
2. models：模型的实现，其中`FoundationModel_ori.py`用于seq1/seq2纯分类模型，`FoundationModel.py`用于seq3带分割头的多任务模型；`FoundationModelHierarchical.py`是主三分类头+异常子类头+可选分割头的层级实验模型，`FoundationModelLesionAwareHierarchical.py`是使用分割 soft attention 汇聚病灶空间特征的 FLAIR 专用层级模型，`FLAIRUNet3D.py`是各向异性 3D U-Net 分割与全局/病灶双路分类模型，`model_factory.py`统一模型注册、能力判断与前向输出
3. utils：一些工具的实现
4. scripts：必要的脚本实现，包含“数据预处理”“mask预处理”和“训练集、验证集和测试集的生成”
5. train_kfold.py：训练脚本，配合k折交叉验证使用
6. eval_kfold.py：单序列或多通道模型测试脚本，先计算每一折的结果，再综合评估
7. eval.py：基础异构晚期融合软投票测试脚本；`eval_foundation_nnunet_guided.py` 用于当前 SynthStrip guided 工作点的锁定评估，eval_vote_kfold.py保留为同构模型软投票评估脚本
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
| `setup_synthstrip.py` | 下载官方 SynthStrip 命令脚本和模型，并校验模型 SHA-256 | `python -m scripts.setup_synthstrip --output-root output/data-synthstrip` |
| `preprocess_data.py` | 预处理三序列 MRI，生成 `data/`、图像和 metadata | `python -m scripts.preprocess_data --config output/data-hdbet/preprocessing_config.py --output-root output/data-hdbet` |
| `preprocess_mask.py` | 根据预处理 metadata 对齐医生 mask，并写入已有 `data/` | `python -m scripts.preprocess_mask --config output/data-hdbet/preprocessing_config.py --data-root output/data-hdbet` |
| `build_dataset_kfold.py` | 对共有病例执行分层 K-Fold 划分，生成 `datasets/` | `python -m scripts.build_dataset_kfold --config output/data-hdbet/dataset_config.py --data-root output/data-hdbet --output-root output/data-hdbet` |
| `check_dataset_kfold.py` | 读取各 fold 的 `.pt`，统计 train/val/test 类别分布 | `python -m scripts.check_dataset_kfold --data-root output/data-hdbet` |
| `check_preprocessed_data.py` | 按预处理配置检查 shape、spacing、文件大小和序列完整性 | `python -m scripts.check_preprocessed_data --config output/data-hdbet/preprocessing_config.py --data-root output/data-hdbet` |
| `check_brain_extraction_quality.py` | 检查脑提取前景体积、连通性、边界、跨序列一致性和病灶保留率，并生成原图叠加图 | `python -m scripts.check_brain_extraction_quality --config output/data-synthstrip/preprocessing_config.py --data-root output/data-synthstrip` |
| `check_file_sizes.py` | 不读取体素，快速列出小于阈值的 NIfTI | `python -m scripts.check_file_sizes --data-root output/data-hdbet --threshold-mb 1` |
| `delete_small_preprocessed_cases.py` | 按 case 清理异常小文件；默认仅 dry-run | `python -m scripts.delete_small_preprocessed_cases --data-root output/data-hdbet --threshold-mb 1` |
| `get_raw_data_info.py` | 只读统计原始 NIfTI 的 size、spacing 和物理尺寸 | `python -m scripts.get_raw_data_info --raw-root /path/to/raw` |

`deprecated_build_dataset.py` 和 `deprecated_check_dataset.py` 是无 K-Fold 的历史脚本，仅供追溯，不再维护或推荐执行。

### 数据预处理
数据预处理有四个步骤：
1. 重采样resample：统一输入图像的spacing,使像素物理意义一致
2. 脑区提取brain extraction：生成前景 mask，并按配置膨胀以减少脑膜邻近信息被切除的风险
3. 归一化normalize：统一输入图像的像素值区间，执行Z-score标准化，使平均值为0，标准差为1。这样能消除设备差异，减小不同机器这个因素对模型的干扰
4. 裁剪/填充crop_or_pad：以前景 bbox 中心为中心，将重采样后的数据填充或裁剪至统一大小

脑区提取方法记录：

- `HD-BET fast`：`output/data-hdbet` 当前历史基线，每个序列独立提取，关闭 TTA，随后统一膨胀 5 mm。
- `SynthStrip`：`output/data-synthstrip` 对照实验，使用官方成人含 CSF 模型；其余 spacing、shape、归一化、外部清零、膨胀和裁剪参数与 HD-BET 基线保持一致。
- `HD-BET accurate`：尚未执行，保留为以后可能的五模型集成/TTA 方向；必须使用新的 `output/data-*`，不能覆盖 `data-hdbet`。

SynthStrip 调用前会检查非有限值、恒定强度、第 99 百分位无对比度及近乎全零的输入。此类源 NIfTI 无法安全归一化，会跳过整个病例并以 `brain_extraction_input_qc` 阶段写入 `data/preprocess_errors.csv`，不会把 NaN mask 保存为预处理结果。

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

首次运行 SynthStrip 前，安装其轻量 Python 依赖并下载官方脚本和模型。下载器会把文件放在被 Git 忽略的实验目录中，并校验模型 SHA-256：

```bash
conda activate BrainMRIClassification
python -m pip install surfa==0.6.3
python -m scripts.setup_synthstrip \
    --output-root output/data-synthstrip
```

如果从默认 PyPI 遇到 `TLS/SSL connection has been closed (EOF)`，本服务器可改用已经验证过的清华镜像：

```bash
python -m pip install surfa==0.6.3 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果当前终端设置了不可用的 HTTP(S) 代理，模型下载命令可追加 `--no-proxy`。随后运行独立的 SynthStrip 数据实验：

```bash
python -m scripts.preprocess_data \
    --config output/data-synthstrip/preprocessing_config.py \
    --output-root output/data-synthstrip

python -m scripts.preprocess_mask \
    --config output/data-synthstrip/preprocessing_config.py \
    --data-root output/data-synthstrip
```

脑提取质量检查会把 CSV、JSON 摘要和默认 24 个可疑病例三平面叠加图写入
`output/data-synthstrip/reports/brain_extraction_qc/`。红线是膨胀、裁剪后的前景边界，存在医生病灶 mask 时黄线表示病灶；建议先做少量 smoke check，再检查全部数据：

```bash
python -m scripts.check_brain_extraction_quality \
    --config output/data-synthstrip/preprocessing_config.py \
    --data-root output/data-synthstrip \
    --max-cases 20 \
    --num-overlays 12

python -m scripts.check_brain_extraction_quality \
    --config output/data-synthstrip/preprocessing_config.py \
    --data-root output/data-synthstrip
```

该检查中的跨序列 Dice 只用于筛查明显差异，不等价于配准质量；厚层数据经常在 Z 轴两端仍有前景，因此脚本报告 Z 边界接触但默认不把它单独判为失败。

确认 SynthStrip 预处理质量后，再构造与该实验绑定的 K-Fold 数据集：

```bash
python -m scripts.build_dataset_kfold \
    --config output/data-synthstrip/dataset_config.py \
    --data-root output/data-synthstrip \
    --output-root output/data-synthstrip
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

#### 层级分类实验

`FoundationModelHierarchical` 使用一个共享 r3d_18 backbone，同时训练：

1. 主头：`normal / inflammation / metastasis` 三分类，保持正常类判别能力。
2. 子类头：`inflammation / metastasis` 二分类，只在真实异常样本上计算损失，直接强化当前最主要的炎症/转移混淆。
3. 可选分割头：相同模型代码分别训练三个序列，但只有 `--seq 3` 创建和训练分割头；seq1/seq2 不创建分割参数。

层级最终判定先由三个主头的平均概率判断 normal/abnormal；若判为 abnormal，再由三个子类头的平均概率决定 inflammation/metastasis。子类头不会单独把主头判为 normal 的病例强行改成转移，因此比全局降低转移阈值更有利于保持整体准确率。

所有模型必须显式声明 `has_classification_head`、`has_subtype_head`、`has_segmentation_head`。训练和评估通过 `models/model_factory.py` 查询能力，不再使用 `model_name == "FoundationModel"` 推断输出。旧模型的默认 `forward(x)` 和 checkpoint state_dict 键保持兼容。

多卡运行时，结构化前向的控制标志由 `forward_model` 统一以位置参数传递。不要把这些布尔标志改成传给 `DataParallel` 的关键字参数：最后一个 batch 小于 GPU 数量时，这会产生没有影像输入的空 replica，并触发 `forward() missing ... 'x'`。该问题不能用 `drop_last=True` 回避，否则会丢训练样本。

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

SynthStrip 数据对应的同超参数基线位于 `output/runs-cross-entropy-synthstrip`。脚本默认训练 seq1/Fold 1；也可指定单序列五折或顺序执行全部 15 个任务。已有 checkpoint 默认跳过：

```bash
bash output/runs-cross-entropy-synthstrip/train.sh
bash output/runs-cross-entropy-synthstrip/train.sh 3 all
bash output/runs-cross-entropy-synthstrip/train.sh all all

# 单独评估一个序列的全部五折；末尾可加 1～5 只评估指定 fold
bash output/runs-cross-entropy-synthstrip/eval.sh seq1
bash output/runs-cross-entropy-synthstrip/eval.sh seq2
bash output/runs-cross-entropy-synthstrip/eval.sh seq3

# checkpoint 齐全后评估全部五折三序列软投票
bash output/runs-cross-entropy-synthstrip/eval.sh fusion all
```

该实验使用 SynthStrip 预处理后通过 QC 的全部 3606 例；历史 HD-BET 基线使用 3588 例，两者共有 3585 例，且共有病例只有约 31% 落在相同 test fold。因此当前结果适合作为“整套预处理管线”的对照，不是严格同病例、同 fold 的配对消融；若要归因到脑提取方法本身，应另建两套共有病例且共享 split 的数据实验并重新训练双方。

#### SynthStrip 同配置基线的结果与定位

| 数据与实验 | Accuracy | Macro-F1 | Metastasis P / R / F1 | 定位 |
| --- | ---: | ---: | ---: | --- |
| HD-BET：`output/runs-cross-entropy` | 0.9044 | 0.8683 | 0.8727 / 0.6946 / 0.7735 | 历史可靠基线，继续作为固定对照 |
| SynthStrip：`output/runs-cross-entropy-synthstrip` | 0.9146 | 0.8805 | 0.9101 / 0.6990 / 0.7907 | 成功的预处理管线基线，作为后续模型优化的主要起点 |

SynthStrip 管线的 pooled 融合结果在 Accuracy、Macro-F1 及转移 precision/F1 上均明确高于
历史基线，转移 recall 则基本持平。其主要收益来自三序列互补性和更少的
inflammation->metastasis 假阳性；三个单序列并非都稳定提高。结合病例集合与 fold 不完全
一致，当前结论应表述为“新预处理、重新划分和重训组成的端到端管线是一次成功尝试”，
不能单独证明 SynthStrip 是全部增益的原因。后续成功方向可优先以该数据管线为起点组合，
但每次只改变一个主要因素，并继续用历史 HD-BET 结果作固定参照。

已知数据治理待办：当前 `case_index` 只按原始末尾数字形成的 `case_key` 防止重复处理，
不能识别编号不同但体素内容相同的检查。内容指纹检查发现 SynthStrip 数据中有 22 对（44
个 case ID）至少一个序列完全相同，其中 19 对跨外层 fold；去除这些病例后的只读比较并
不能解释本次性能提升，但只有重新分组训练才能严格排除影响。后续应为选中的 T1/T2/FLAIR
计算体素内容指纹，任一序列完全相同的病例归入同一 duplicate group，并在 K-fold 中保证
同组只进入一个外层 fold；三个序列均相同时可考虑只保留一个，标签冲突时必须人工核对，
不在预处理阶段静默删除。

#### SynthStrip + nnU-Net 引导的叠加结果

SynthStrip 应视为一个**预处理与数据管线层面的优化大类**，而不是某个分类头的小改动。
在此基础上，Dataset502 使用相同 SynthStrip FLAIR 和项目五折划分重新训练原生 nnU-Net；
`output/runs-foundation-nnunet-guided-synthstrip` 冻结 SynthStrip Foundation 与 nnU-Net，
仅训练 OOF soft-mask 特征上的 subtype expert。这样先固定新的数据管线，再验证模型级改动
能否继续叠加，避免把脑提取、病例集合、fold 和模型结构的影响混在一次实验中。

以下两行来自相同的 3606 例 SynthStrip 五折 locked test。`F2` 是更偏重召回率的 F-score：

| 模型 | Accuracy | Macro-F1 | 转移 Precision | 转移 Recall | 转移 F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation fusion | 0.9146 | 0.8805 | 0.9101 | 0.6990 | 0.7330 |
| nnU-Net-guided fusion | **0.9157** | **0.8826** | 0.8985 | **0.7151** | **0.7456** |

guided fusion 在转移 precision 下降 `1.16` 个百分点的同时，Accuracy、Macro-F1、转移
Recall 和 F2 均提高；3606 例中只有 19 例发生最终融合改判，其中纠正 11 例、损害 7 例，
净增加 4 例正确分类。五折 validation 选择的引导混合权重分别为
`0.02/0.02/0/0.28/0.44`，Fold 3 因未满足保护约束自动回退 Foundation。因此该结果不是
对全部病例大幅改写，而是在保留强 Foundation 基线的前提下小幅推动炎症/转移边界。按照
“Accuracy 保持 0.90 以上并提高转移召回”的既定目标，它是当前内部测试最佳分类工作点；
Foundation fusion 继续保留为转移 precision 更高的保守工作点。

Dataset502 原生 nnU-Net 五折结果位于 `output/nnunet-flair-synthstrip`。SynthStrip guided
五折训练与锁定评估命令为：

```bash
bash output/runs-foundation-nnunet-guided-synthstrip/train.sh
bash output/runs-foundation-nnunet-guided-synthstrip/eval.sh
```

后续实验按两个层次组织：第一层固定 SynthStrip 数据、split 和 Foundation 基线；第二层再
分别复现此前较成功的模型级方向，例如 lesion-aware/hierarchical subtype head、受约束
温度缩放或 nnU-Net guidance。先做单因素复现，再研究 `SynthStrip + guidance + 层级边界`
等组合，不假定旧 HD-BET checkpoint 或各项收益能够直接相加。所有混合系数、温度与约束
继续只在 validation/OOF 上选择；本次 locked test 已用于最终开发记录，不再据此调参。

层级模型使用现有 `output/data-hdbet`，不需要重新预处理或重建数据集。首轮配置为
`SUBTYPE_ALPHA=0.5`、`SUBTYPE_CLASS_WEIGHT_POWER=0.5`、
`HIERARCHICAL_MIN_VAL_ACCURACY=0.85`。这里的 0.85 是单序列验证准确率下限；旧基准的
单序列验证准确率约为 0.87～0.90，最终三序列融合仍以 0.90+ 为目标：

```bash
# 把 --seq 和 --fold 分别替换为 1~3、1~5，合计训练 15 个模型
python train_kfold.py \
    --config output/runs-hierarchical/train_config.py \
    --data-root output/data-hdbet \
    --output-root output/runs-hierarchical \
    --seq 1 \
    --model FoundationModelHierarchical \
    --fold 1
```

层级模型优先在验证集层级准确率达到 `HIERARCHICAL_MIN_VAL_ACCURACY` 的 epoch 中，以
`val_hierarchical_f1` 保存最佳 checkpoint；若整折始终未达下限，才回退到层级 F1
最高的 epoch。测试集不参与训练、早停、阈值或 checkpoint 选择。

`CLASSIFICATION_LOSS` 可设为 `weighted_cross_entropy`（当前默认逻辑）、`cross_entropy` 或 `class_balanced_focal`。迁移后的历史 `.pt` 仍保存了原 `version1/data/...` 路径，但加载器会根据 `--data-root output/data-hdbet` 自动重定位到新的 `data/`。

### FLAIR 3D U-Net 分阶段实验

`FLAIRUNet3D` 用于验证“先学会定位，再让定位辅助分类”的路线。它与历史
`FoundationModel` 的主要差别是：

1. 使用四级 U-Net decoder，每一级都通过转置卷积上采样并连接对应 encoder skip；
2. 根据当前 `(3.0, 0.75, 0.75) mm` spacing，前两级使用 `(1,3,3)` 卷积和
   `(1,2,2)` 下采样，平面 spacing 接近层厚后才沿 Z 轴下采样；
3. decoder 提供三个由低到高分辨率的 deep-supervision 输出；
4. 分割目标是单通道二值 lesion mask。医生 mask 实际只表达病灶区域，不再用病例级
   inflammation/metastasis 标签人为制造三类分割目标；
5. 分类头拼接 bottleneck 全局池化特征与预测 soft lesion mask 引导的最高分辨率
   decoder 特征，分割不完美时仍保留全脑上下文。

现有 `train_kfold.py` 的单优化器、三类 voxel CE 和按分类分布采样不适合这个实验，
因此使用独立入口 `train_flair_unet.py`。三个阶段都只使用对应 fold 的 train/validation；
test 不参与 checkpoint、阈值或超参数选择。阶段一会自动排除“异常但无可靠 mask”的病例，
保留所有有医生 mask 的异常病例和正常全零 mask 病例。当前 Fold 1 train 中二者是
543:248，约为 2.19:1，无需为了得到 2:1 再重复采样。

当前保留的相关实验根：

| 实验根 | 模型与阶段 | 状态 |
| --- | --- | --- |
| `output/runs-flair-unet-segmentation` | 原始 `FLAIRUNet3D`，阶段一：从头训练分割 | 5 折完整，作为结构/训练基线 |
| `output/runs-flair-unet-nnunet-segmentation` | `FLAIRUNet3DNNUNet`，阶段一：导入 nnU-Net Fold 1 epoch-877 best | 当前有效分割初始化，含正确 validation 报告 |
| `output/runs-foundation-nnunet-guided` | 冻结 Foundation 分类器 + OOF nnU-Net soft-mask subtype expert | 五折完成；锁定 test 显示为召回优先的小幅有效改进 |
| `output/nnunet-flair-synthstrip` | SynthStrip Dataset502 原生 nnU-Net | 五折完成；当前病灶分割首选 |
| `output/runs-foundation-nnunet-guided-synthstrip` | SynthStrip Foundation + Dataset502 OOF guidance | 五折完成；当前内部测试最佳分类工作点 |

先只跑 Fold 1 的阶段一，确认 positive-case Dice、病灶体素 recall 和正常病例假阳性体素
比例是否合理：

```bash
python train_flair_unet.py \
    --config output/runs-flair-unet-segmentation/train_config.py \
    --data-root output/data-hdbet \
    --output-root output/runs-flair-unet-segmentation \
    --stage segmentation \
    --fold 1
```

这是全体积 `(48,320,320)` 3D U-Net 训练，默认 batch size 为 1，属于长任务，单折通常
需要数小时而不是几分钟。开始正式五折前应先观察 Fold 1 的显存和单 epoch 时间；如果首个
epoch 15～20 分钟仍没有任何 batch 进度，再检查数据读取或显存问题。不要直接改回历史
batch size 8。

原始全体积 U-Net 的阶段二/阶段三从未实际训练，且不适用于当前 patch-based nnU-Net
骨干，相关空配置已清理。当前先完成下文 nnU-Net 专用阶段二；阶段三联合微调需要在
阶段二 validation 结果确认后再单独适配。

阶段一五折完成后，使用 validation-only 诊断脚本扫描分割概率阈值，并输出逐病例
Dice/precision/recall、正常病例假阳性体积、完全漏检率、病灶体积分层以及最佳/最差病例
叠加图。该脚本不会读取 test：

```bash
python analyze_flair_unet_segmentation.py \
    --config output/runs-flair-unet-segmentation/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-flair-unet-segmentation \
    --device cuda:0 --batch-size 1 --num-workers 8
```

默认报告写入
`output/runs-flair-unet-segmentation/reports/validation_segmentation_analysis/`。阈值推荐以
0.5 为基线，在“不增加正常病例平均假阳性体积”的候选中最大化 positive-case mean
Dice；任何阈值或后续超参数判断都只使用 validation 结果。

Hard-negative BCE + Dice 已完成失败判定并清理：Fold 1/2 的 positive Dice 分别只有
`0.1710/0.0736`，pooled Dice `0.1185`，明显低于标准 U-Net pooled `0.3527`；该设置
表现为高 recall、极低 precision，不应按原参数重复训练。

#### nnU-Net 风格 FLAIR U-Net

独立 nnU-Net Dataset501 的 Fold 1 final checkpoint 在相同 132 个 validation 病例上，
项目口径的 positive-case mean Dice/precision/recall 为
`0.3682/0.4421/0.4036`，完全漏检率为 `7.95%`；44 个正常病例的 mean/P95 FP
体积为 `1.40/4.56 mL`。nnU-Net 终端的 `Mean Validation Dice=0.3411` 还会把产生
假阳性的正常病例按 Dice 0 纳入均值，因此不对应项目一直报告的
`positive_dice`；新训练日志将这个官方混合口径单列为
`nnunet_foreground_dice`，并继续用前述拆分指标解释模型表现。

`FLAIRUNet3DNNUNet` 在保留全局/soft-mask 局部双路分类接口的前提下复现该成功设置：

1. 六级 `32/64/128/256/320/320` PlainConvUNet、strided-conv 下采样、两通道
   segmentation head；对外返回 lesion/background logit 差，因此仍兼容现有单通道
   二值分割接口；
2. `(40,224,192)` patch、global batch 2、每 epoch 250 次更新和约 50% 的实际强制
   前景 patch（batch 2 下由 nnU-Net 的 0.33 规则取整得到）；
3. dense CE 等价 BCE + foreground soft Dice、nnU-Net deep-supervision 权重；
4. SGD Nesterov、初始学习率 `0.01`、momentum `0.99`、poly schedule、梯度裁剪 12；
5. 与各向异性计划一致的平面旋转/缩放、噪声、模糊、亮度、对比度、低分辨率、
   gamma 和三轴镜像增强；整例非零脑区 Z-score 保持训练/验证一致，validation 使用
   非零区域裁剪、Gaussian sliding window 和 mirroring。

当前有效的阶段一位于 `output/runs-flair-unet-nnunet-segmentation`：它把独立 nnU-Net
重新训练得到的 Fold 1 epoch-877 best 逐层转换为项目模型，只新增随机初始化的分类分支，
不会冒充分割权重。decoder 通道顺序修复后，项目 validation 口径的 positive
Dice/precision/recall 为 `0.3715/0.4484/0.4036`，完全漏检率 `4.55%`，正常病例
mean/P95 FP 为 `2.50/9.23 mL`。原生 best 仍备份在 `output/nnunet-flair`；旧
epoch-1000 final 的项目导入副本已经被这个召回更优的 best 取代并清理。

冻结 nnU-Net encoder/decoder 后直接训练三分类头的两轮实验都明显低于 Foundation
分类基线，相关 warmup 目录已经清理。当前路线改为保留完整 Foundation 分类器，只把
nnU-Net soft mask 用作多尺度空间先验：对 Foundation layer2/3/4 分别计算全脑池化、
病灶加权池化和病灶-周边对比池化，再由独立线性 subtype expert 区分 inflammation 与
metastasis；Foundation 仍负责 normal 门控。目标 fold 的 train 病例按其外层 test fold
选择从未见过该病例的 nnU-Net，validation/test 使用目标 nnU-Net fold，避免分割特征泄漏。

Fold 1 的 validation 选择在第 1 个 expert epoch 完成；后续离散指标不变而连续 validation
BCE 恶化，早停在第 5 轮并保留 epoch 1，属于线性 probe 的正常决策平台期。运行五折时，
脚本会跳过已有 checkpoint，并把 Fold 2–5 分配到四组双卡；影像特征缓存是长任务，线性
expert 训练本身只需几分钟：

```bash
bash output/runs-foundation-nnunet-guided/train.sh
```

Fold 1–4 的 validation 均选择了满足保护约束的正 alpha；Fold 5 的病灶 expert 在提高转移
召回前已经造成过多 inflammation->metastasis 假阳性，因此自动保存 alpha=0 的
`foundation_fallback`，严格复现该折 Foundation。`best_unconstrained` 只用于说明高召回
代价，不进入正式评估。以后若其他 fold 也没有安全增益，训练入口会执行相同回退并正常
结束，而不是令整个五折任务失败。

五折全部完成后，以下命令只使用 validation 保存的 epoch 和混合系数做一次锁定 test，
同时比较 FLAIR 单模态和 T1/T2/FLAIR 等权融合的 Foundation 基线与引导版本：

```bash
bash output/runs-foundation-nnunet-guided/eval.sh
```

其中 baseline fusion 严格复用历史 FP32 对照概率：seq1/T1 与 seq2/T2 均为
`FoundationModel_ori`，seq3/FLAIR 为 `FoundationModel`，三者等权平均；guided fusion
保持 seq1/seq2 不变，只替换 seq3 的炎症/转移条件概率。不要为 T1/T2 额外启用 FP16
autocast，它曾导致正常类概率失真并产生约 0.786 的错误融合准确率。修复后的 pooled test
结果如下。`F2` 是更重视召回率的 F-score（召回权重是精确率的 4 倍）：

| 输出 | Accuracy | Macro-F1 | Metastasis P / R / F1 / F2 |
| --- | ---: | ---: | ---: |
| Foundation FLAIR | 0.8877 | 0.8491 | 0.7916 / 0.6916 / 0.7382 / 0.7096 |
| Guided FLAIR | 0.8871 | 0.8506 | 0.7701 / 0.7181 / 0.7432 / 0.7279 |
| Baseline fusion | 0.9044 | 0.8683 | 0.8727 / 0.6946 / 0.7735 / 0.7241 |
| Guided fusion | 0.9038 | 0.8683 | 0.8628 / 0.7019 / 0.7741 / 0.7291 |

因此这次实验可列为**成功但幅度有限的召回优先尝试**，不能替代
`output/runs-cross-entropy` 作为当前可靠综合基线：Guided FLAIR 的转移召回提高
`2.65` 个百分点，进入三序列融合后仍提高 `0.73` 个百分点，且融合 accuracy 保持在
预设的 `0.90` 以上。Fold 1–4 的融合转移召回均提高，Fold 5 因 validation 保护约束未
满足而以 `alpha=0` 回退，结果不变，方向具有跨折一致性。代价是 pooled fusion 多找回
5 个转移病例的同时，新增 7 个 inflammation->metastasis 假阳性，最终少判对 2/3588
例，转移 precision 下降 `0.99` 个百分点。它证明 nnU-Net 病灶先验可以把既有分类器推向
更偏转移召回的工作点，但增量较小，尚不能宣称具有统计或临床显著优势。本次 test 已作为
锁定开发记录，不再用来修改 epoch、混合系数或选择约束。

阶段一的训练配置已与有效 checkpoint 合并在同一个实验根。不要在该根直接重新启动
fresh training，以免触发覆盖保护；它主要用于复现实验参数和执行 validation 分析：

```bash
CUDA_VISIBLE_DEVICES=0 python analyze_flair_unet_segmentation.py --config output/runs-flair-unet-nnunet-segmentation/train_config.py --data-root output/data-hdbet --checkpoint-root output/runs-flair-unet-nnunet-segmentation --folds 1 --device cuda:0 --batch-size 1 --num-workers 8
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

# 历史 HD-BET 异构晚期融合固定对照
python eval.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-cross-entropy

# 当前 SynthStrip nnU-Net-guided 最佳工作点（五折 checkpoint 完成后）
bash output/runs-foundation-nnunet-guided-synthstrip/eval.sh

# 层级模型单序列评估：同时报告主头、层级输出和异常子类头指标
python eval_kfold.py \
    --config output/runs-hierarchical/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-hierarchical \
    --seq 3 \
    --model FoundationModelHierarchical

# 三序列层级融合评估
python eval.py \
    --config output/runs-hierarchical/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-hierarchical \
    --model-set hierarchical

# 临床层级融合推理
python infer.py \
    --id 0001 \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-hierarchical \
    --output-root output/runs-hierarchical \
    --model-set hierarchical

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

### 历史可靠基线与当前最佳工作点

历史固定对照位于 `output/runs-cross-entropy`。seq1/T1、seq2/T2 使用
`FoundationModel_ori`，seq3/FLAIR 使用带 mask 辅助分割头的 `FoundationModel`，三个
模型的 FP32 三分类概率等权平均。下面使用当前 `output/data-hdbet` 划分和当前评估入口；
每个病例只出现在一个 outer test fold 中，五折 pooled 共 3588 例：

```bash
python eval.py \
    --config output/runs-cross-entropy/train_config.py \
    --data-root output/data-hdbet \
    --checkpoint-root output/runs-cross-entropy
```

| Fold | Test samples | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Metastasis recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 718 | 0.8969 | 0.8934 | 0.8253 | 0.8548 | 0.6912 |
| 2 | 718 | 0.9123 | 0.9228 | 0.8445 | 0.8766 | 0.6765 |
| 3 | 718 | 0.9081 | 0.9163 | 0.8433 | 0.8737 | 0.6788 |
| 4 | 717 | 0.9038 | 0.8892 | 0.8495 | 0.8671 | 0.7132 |
| 5 | 717 | 0.9010 | 0.8940 | 0.8482 | 0.8691 | 0.7132 |

五折指标的 mean ± standard deviation 为：Accuracy `0.9044 ± 0.0053`、Macro-Precision
`0.9032 ± 0.0137`、Macro-Recall `0.8422 ± 0.0088`、Macro-F1 `0.8683 ± 0.0075`。
将五个互斥 test fold 的预测合并后，pooled Accuracy/Macro-F1 为 `0.9044/0.8683`，
metastasis precision/recall/F1/F2 为 `0.8727/0.6946/0.7735/0.7241`。pooled 混淆矩阵
如下，类别顺序为 `normal`、`inflammation`、`metastasis`：

```text
[[ 316,   48,   1],
 [  18, 2456,  68],
 [   7,  201, 473]]
```

旧 README 曾记录 `0.9112 ± 0.0143` accuracy，但对应每折 745–746 例、总计 3727 例，
与当前每折 717–718 例、pooled 3588 例的病例集合并不一致，也不是当前
`runs-cross-entropy/output_texts/fusion.txt` 的结果。由于旧记录缺少足够的数据版本
provenance，不能把两个数值当作同一测试集上的模型升降；本节已经用当前产物和统一评估
口径替换旧表。该基线的主要瓶颈仍是将 metastasis 判为 inflammation（201 例）。它继续
承担跨历史实验的固定参照作用，但不再是当前点估计最高的工作点。当前分类首选是上一节的
SynthStrip nnU-Net-guided fusion（Accuracy/Macro-F1/转移 Recall/F2 为
`0.9157/0.8826/0.7151/0.7456`）；当前分割首选是
`output/nnunet-flair-synthstrip` 的 Dataset502 原生 nnU-Net。由于两套预处理管线的病例
集合和 fold 不完全一致，HD-BET 与 SynthStrip 之间只作端到端管线比较，不能解释为严格的
单变量脑提取消融。

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
