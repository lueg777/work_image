# AIGC 频域取证：残差算子族消融

期末大作业项目（图像处理与分析）。基于 NPR（CVPR 2024, `chuangchuangtan/NPR-DeepfakeDetection`）
做**残差算子族的系统消融**，研究"通用 AIGC 检测器到底靠什么"。

---

## 1. 一句话结论式的问题设定

NPR 之所以能跨生成器泛化，官方解释是"上采样算子会产生**频域之外**的伪影"。
但读它的源码会发现一件更简单的事：**NPR 本身就是一个固定高通滤波器**。

```python
# networks/resnet.py, 官方实现
def interpolate(self, img, factor):
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', ...),
                         scale_factor=1/factor, mode='nearest', ...)

def forward(self, x):
    NPR = x - self.interpolate(x, 0.5)      # 高通残差
    x = self.conv1(NPR * 2.0 / 3.0)          # 再送进 ResNet
```

也就是说，网络的输入端不是图像，而是 `x - 最近邻下采样再上采样(x)`。
这就留下一个**尚未被明确回答、且纯图像处理就能回答**的问题：

> NPR 的泛化能力来自"提取了高频"，
> 还是来自"它的算子恰好匹配了生成器的上采样核"？

本项目把这个算子从一个写死的常量，变成一个**可替换的算子族**，用同一套训练/评测流程
扫出性能差，从而把上面二选一变成可测的数字。

### 可证伪的预测

| 假设 | 预测 | 若观察到的结果 |
| --- | --- | --- |
| **算子匹配说** | 只有 `nearest` 往返最好；bilinear / bicubic / box / Laplacian / DCT / FFT 全部明显更差 | 支持"匹配"解释 → 频域只是表象 |
| **纯高通说** | 所有高通算子（含 Laplacian、DCT、FFT）性能接近，只有 `identity` 差 | 支持"高频"解释 → 算子形式无关 |
| **两者都不是** | 算子之间无稳定差异，但 `identity` 也接近 | 说明残差算子本身贡献很小，泛化另有来源 |

三种结果都能写成报告，**没有"实验失败"这种结局** —— 这是选题时就定下的标准。

---

## 2. 算子族

| 名称 | 定义 | 归属 | 作用 |
| --- | --- | --- | --- |
| `npr_nearest` | `x - up_nearest(down_nearest(x))`，×2/3 | match | 官方基线 |
| `npr_bilinear` | 同式，bilinear | match | 换插值核，检验"核是否必须匹配" |
| `npr_bicubic` | 同式，bicubic | match | 同上 |
| `npr_box` | `x - up_nearest(avgpool2(x))` | match | 分离"点采样"与"抗混叠低通" |
| `laplacian` | `x - GaussianBlur(x)`, σ=1 | highpass | 教科书式高通 |
| `dct_lowpass` | `x - IDCT(8×8 块 DCT 径向低频)` | highpass | 频域高通（JPEG 同源） |
| `fft_radial` | `x - IFFT(DFT 径向低频)` | highpass | 频域高通（全局） |
| `identity` | 原图，不做高通 | control | 归因残差算子本身的贡献 |

所有算子返回与输入同形状的张量；奇数边长按官方 README 的约定裁掉 1 像素。
算子无参数、无持久化 buffer，因此 **`state_dict` 与官方 `NPR.pth` 完全兼容**（已由 smoke test 断言）。

---

## 3. 目录结构

```
aigc-freq-forensics/
├── configs/
│   ├── default.yaml            # 路径/预处理/训练超参
│   └── operators.yaml          # 算子族声明、分组依据、量纲告警
├── docs/
│   └── plan.md                 # 实验矩阵、四周计划、决策日志、风险
├── scripts/
│   ├── smoke_test.py           # 无数据集自检（70 项）
│   ├── prepare_data.py         # organize / verify / manifest（torch-free）
│   ├── download_data.py        # 下载 → 解压 → 摆位 → 体检，一条命令
│   └── build_report.py         # 生成交作业的 1_报告.docx（幂等，可重跑）
└── src/
    ├── ops/residual_ops.py     # ★ 核心：算子族
    ├── models/npr_resnet.py    # NPR 主干（layers=(3,4) → 1.44M 参数）
    ├── layout.py               # real/fake 配对规则（loader 与准备脚本共用）
    ├── data/
    │   ├── transforms.py       # 与官方逐行一致的预处理
    │   └── datasets.py         # real/fake 标签钉死，不依赖排序
    ├── eval/
    │   ├── metrics.py          # ACC / AUC / AP / TPR@FPR（纯 numpy）
    │   ├── spectrum.py         # 径向功率谱、高频能量比（训练无关基线）
    │   ├── engine.py           # ckpt 加载 + 打分开采
    │   ├── cross_generator.py  # ★ 跨生成器扫描
    │   └── robustness.py       # ★ JPEG / 缩放鲁棒性矩阵
    ├── train.py                # 训练（Adam 2e-4, 50ep, 每 10ep ÷10）
    ├── test.py                 # 评测入口
    ├── sweep.py                # ★ 算子消融驱动
    └── build.py                # 配置 + DataLoader
```

---

## 4. 预处理：为什么它值得单独一个文件

`src/data/transforms.py` 是逐行对齐官方 `data/datasets.py` 写的，因为**预处理不一致会
让所有对比数字失去意义**。三个容易踩的点：

1. 官方是 `RandomCrop(224)`，**不是** `RandomResizedCrop` —— 裁剪不改变分辨率尺度。
2. 顺序固定为 `Resize(256,256) → crop → flip → ToTensor → Normalize`。
3. 官方 `testmodel()` 里是 `no_resize=False, no_crop=True`，即**在整张 256×256 上评**，
   而验证集用的是 `CenterCrop(224)`。两者不同，配置里留了 `eval_no_crop` 开关。

归一化固定为 ImageNet 均值方差 `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`。
GenImage 场景的 `translate_duplicate`（整数平铺复制，不插值）也实现了，标记为 EXPERIMENTAL。

---

## 5. 快速开始

### 5.1 自检（不需要数据、不需要 GPU）

```bash
python scripts/smoke_test.py            # 无 torch 时自动跳过 torch 段
python scripts/smoke_test.py --require-torch   # 有 torch 时要求全过
```

已实测：**70 项全过**（2026-11-03，CPU torch 2.15.0+cpu）：

| 段 | 项数 | 覆盖内容 |
|---|---|---|
| metrics | 11 | AUC/AP 与暴力枚举逐一比对、混淆矩阵、tpr@fpr |
| spectrum | 14 | DC 剔除、径向谱平坦性、距离对称性、形状不匹配报错 |
| data prep | 21 | real/fake 配对规则、organize 三次重跑、硬链接别名、哈希泄漏、manifest |
| transforms | 2 | JPEG 往返、上下采样还原尺寸 |
| 算子 + 模型 | 22 | NPR 官方公式逐元素一致（最大偏差 `5.96e-08`）、常数图归零、参数量 `1,437,761`、`state_dict` 无 `op.` 键、Adam 三步降 loss |

其中 data prep 段的 `LEAKAGE` / `Broken: real side empty` 两行 `FAIL` 是**被断言的预期输出**
（脚本先证明 verify 能报错，再断言它确实返回 1），不是自检失败。

### 5.2 装环境

```bash
pip install -r requirements.txt
```

### 5.3 数据

目录约定（每个生成器一个文件夹，内含 `real` 与 `fake`）：

```
datasets/
├── ForenSynths_train_val/
│   ├── train/{car,cat,chair,horse}/{real,fake}/
│   └── val/{car,cat,chair,horse}/{real,fake}/
└── Generalization_Test/
    ├── ProGAN/{real,fake}/
    ├── StyleGAN/{real,fake}/
    ├── ADM/{real,fake}/
    ├── Glide/{real,fake}/
    ├── Midjourney/{real,fake}/
    ├── SDv1.4/{real,fake}/
    └── ...
```

数据来源（2026-11 核对自 NPR / CNNDetection 官方 README）：

| 用途 | 数据集 | 下载 |
| --- | --- | --- |
| 训练 + 验证 | ForenSynths（20 类，取 car/cat/chair/horse） | [CNNDetection](https://github.com/PeterWang512/CNNDetection) → [百度盘](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)（提取码 ft8b） |
| **跨生成器测试（16 个，出表 4/表 5）** | AIGCDetectBenchmark | [Google Drive](https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw) |
| 备用测试集：GAN | GANGen-Detection（FreqNet） | [Google Drive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj) |
| 备用测试集：扩散 | DiffusionForensics（DIRE） | [Google Drive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf) |
| 备用测试集：扩散 | UniversalFakeDetect | [Google Drive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-) |
| 备用测试集：扩散 | Diffusion1kStep | [Google Drive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq) |

一条命令搞定下载 → 解压 → 摆成 loader 要的形状 → 体检：

```bash
pip install gdown==4.7.1          # 只有 Google Drive 那条路需要
python scripts/download_data.py --check                    # 先看网络和工具
python scripts/download_data.py --dataset benchmark
```

下载根目录默认 `~/aigc_data`（Windows 即 `C:\Users\<你>\aigc_data`），不存在会自动建，
需要 **30 GB 空闲**。想换盘就加 `--dest <目录>`；**保持它和 `--into`（默认 `<repo>/datasets`）
在同一个盘**，摆位走硬链接，不会再多占一份 30 GB。

`download_data.py` 做四件事，每一步都能单独跑：

| 步骤 | 说明 |
| --- | --- |
| 探测 | `--check` 只测 `drive.google.com` / `pan.baidu.com` 的 TLS 连通性 + gdown/7z 是否存在，不下载 |
| 下载 | gdown 拉整个 Drive 文件夹（`--remaining-ok` 处理 >50 文件）；已有内容则跳过 |
| 解压 | `.zip`/`.tar.gz` 用标准库，`.7z`/`.rar` 用 7-Zip，下完自动展开 |
| 摆位 + 体检 | 按 `0_real`/`1_fake` 配对、**合并 ProGAN/StyleGAN 这类按物体类拆开的多类生成器**、`guided`→`ADM` 等官方别名归一，最后跑 `verify`（计数/不平衡/重复哈希） |

Drive 连不上（校园网常见）或已经手动下好了，就跳过下载这一步：

```bash
# 已解压的目录
python scripts/download_data.py --dataset benchmark --local-dir <下载目录>/AIGCDetectBenchmark
# 或者压缩包
python scripts/download_data.py --dataset benchmark --local-archive <下载目录>/aigcdetect.zip
# ForenSynths 训练集（百度盘手动下，脚本只做摆位；--split 从目录名自动推断）
python scripts/download_data.py --dataset forensynths --local-dir <下载目录>/ForenSynths/train
```

> 只跑零样本复现（表 5 的 `npr_nearest` 一行 + 报告表 4 的对照）只需要 **AIGCDetectBenchmark**
> 一个包；`ForenSynths` 只有做 `--mode scratch` 训练时才需要。GenImage 是百万级，
> 本项目不用（见 `docs/plan.md` D2）。

需要更细的控制（自定义改名、只拷某几个生成器）时才直接用 `prepare_data.py`，**不要手工 mv**：

```bash
# 转成 <generator>/{real,fake}；hardlink 优先，30GB 数据集不多占空间
python scripts/prepare_data.py organize --src <下载目录>/Generalization_Test \
    --dst datasets/Generalization_Test --layout flat

# ForenSynths 训练集：car/cat/chair/horse 是同一生成器 ProGAN 的四个物体类别，
# 必须显式 --generator，否则模型学的是"哪个物体"
python scripts/prepare_data.py organize --src <下载目录>/ForenSynths/train \
    --dst datasets/ForenSynths_train_val/train --generator ProGAN

# 体检：计数 / 不平衡 / 重复 / 与训练集的哈希泄漏
python scripts/prepare_data.py verify --root datasets/Generalization_Test \
    --leakage-with datasets/ForenSynths_train_val/train --strict

# 出清单 CSV，报告第 3 节"数据描述"直接引用
python scripts/prepare_data.py manifest --root datasets/Generalization_Test \
    --out manifests/generalization_test.csv --image-sizes
```

设计要点：

- **配对规则只有一份**（`src/layout.py:pair_class_dirs`），loader 与准备脚本共用。谁都不靠
  "目录名里有没有 real" 猜，而是要求"恰好一个 real-like + 一个 fake-like"；两个生成器名
  （`ProGAN`/`StyleGAN` 都长得像 fake）会被判为歧义并拒绝，而不是悄悄把一对生成器切成 real/fake。
- `organize` **绝不覆盖**：目标已存在且内容相同→`skipped`；内容不同→改名 `__dup2`、计入
  `renamed`。拷贝中断可直接重跑。
- `--mode auto` 在同一个卷上优先建硬链接（省 30GB），代价是**暂存文件就是原文件**，就地重写会
  改到下载的原始数据；两棵树必须独立时用 `--mode copy`。
- 布局自动识别：`<root>/<benchmark>/<gen>/{real,fake}`（官方形状）也能读，显示名写成
  `benchmark/generator`，避免两个基准都有 `ADM` 时混淆。

### 5.4 三个实验

```bash
# ① 零样本复现：官方权重，nearest 算子，跨生成器
python -m src.test --ckpt NPR.pth --op npr_nearest --cross

# ② 算子消融（推荐先跑这个，便宜）：同一份权重换算子
python -m src.sweep --mode pretrained --ckpt NPR.pth --max-batches 20

# ③ 算子消融（干净版，贵）：每个算子从头训练
python -m src.sweep --mode scratch --ops npr_nearest,laplacian --epochs 50

# ④ 鲁棒性矩阵：JPEG 95/75/50 + 0.5× 缩放
python -m src.test --ckpt NPR.pth --cross --robustness --out-dir runs/eval_npr
```

> **②和③回答的不是同一个问题。** ② 是在冻结的 `conv1` 权重下换输入算子，它测的是
> "学到的滤波器有多依赖它训练时那个残差"，混杂了"算子更差"和"这些权重没见过这种输入分布"。
> 干净结论要靠 ③。报告里必须写清是哪一种。

### 5.5 参数量自检

`layers=(3,4)` 时 `conv+BN+fc ≈ 1.44M`，与论文所述一致（smoke test 断言落在 1.3M–1.6M）。
若你的实现给出 ~0.81M，说明 layers 用了 `(2,2)`。

### 5.6 生成作业报告

```bash
python scripts/build_report.py                  # 写 【图像处理】-01234567-姓名/1_报告.docx
python scripts/build_report.py --dry-run        # 只打印路径，不落盘
```

报告由模板复制而来（保留老师给的前四个段落），正文、五张表全部由脚本决定，所以：

- **幂等**：重复运行会先清空正文与表格再重写，不会叠出第二份；
- **数据驱动**：读 `runs/sweep/summary.json` / `cross_all_ops.csv` 填表 5 的实测两列，
  没有结果时写 `待补`，绝不填论文值或合成数据；
- **预注册**：表 5 的"预期 / 机制依据"两列来自脚本里的 `EXPECTED` 字典，写在跑实验之前，
  避免"看完结果再解释"。

---

## 6. 参考数字（论文的，不是本项目的）

评测时用来判断"我复现得对不对"。以下均为 Acc%，同一个 AIGCDetectBenchmark：

| 方法 | 平均 | SDv1.4 | Midjourney | ADM | Glide |
| --- | --- | --- | --- | --- | --- |
| NPR (CVPR'24) | **91.7** | 97.4 | 92.6 | 84.9 | 96.7 |
| CNNSpot (CVPR'20) | 70.78 | **50.57** | 51.39 | 60.39 | 58.07 |
| LGrad (CVPR'23) | — | — | — | — | — |

CNNSpot 那一行是本项目最有力的动机证据：**在扩散模型上几乎随机（50.5%）**。
"频域/CNN 指纹在扩散模型上失效"不是假设，是已发表数字。

> ⚠️ 这些数字**不是**本项目的测量结果，任何报告里都必须标注引用来源。

---

## 7. 诚实边界

| 项 | 状态 |
| --- | --- |
| metrics / spectrum（纯 numpy） | ✅ 实测，25 项断言全过 |
| 数据工具 `layout.py` / `prepare_data.py` | ✅ 实测，21 项断言全过（含硬链接别名、冲突改名、跨集哈希泄漏） |
| 算子族 / 模型 / 训练循环（torch） | ✅ 实测（CPU torch 2.15.0+cpu）：`smoke_test.py` 共 **70 项**全过 |
| 预处理与官方一致性 | ✅ 已对照官方 `data/datasets.py` 源码逐条落实 |
| `train.py` / `test.py` / `sweep.py` 端到端 | ✅ 已在**合成数据**（随机噪声 vs 2× 最近邻上采样纹理）上跑通：训练 1 epoch 得 val_acc 66.67 / AUC 82.64；`--cross --robustness` 出 CSV 与矩阵；`sweep --mode pretrained` 过全部 8 个算子 |
| 真实数据集上的数字 | ❌ 未跑。本仓库不含任何 ForenSynths / Diffusion1kStep 的实测结果 |
| 官方 `Generalization_Test` 内部层级 | ⚠️ 与官方不同（见 5.3），需整理后使用 |
| 论文表格数字 | ✅ 摘自论文/README，已标注为引用 |
| `translate_duplicate` 精确实现 | ⚠️ EXPERIMENTAL，按 README 描述重建，未与原实现逐像素比对 |

> 合成数据上的"跑通"只证明**代码路径正确**（能读写、能算、能出表），
> 不证明方法有效。方法有效性必须等真实数据。

torch 段建议在你的机器上第一条命令就跑：

```bash
python scripts/smoke_test.py --require-torch
```

它会把算子恒等性（常数图 → 残差 0）、NPR 官方公式逐元素一致性、参数数量级、
`state_dict` 不含算子 buffer、Adam 三步 loss 下降全部验证一遍。

---

## 8. 与作业报告五节的对应

作业模板要求 Github 链接 / 任务描述 / 数据描述 / **代码修改描述** / **结果对比**：

| 报告节 | 本项目对应物 |
| --- | --- |
| 1. Github 链接 | 本仓库 |
| 2. 任务描述 | README §1（问题设定 + 三条可证伪预测） |
| 3. 数据描述 | README §5.3 + `docs/plan.md` §2（数据源、规模、划分） |
| 4. 代码修改描述 | `src/ops/residual_ops.py` 算子族 + `sweep.py` 驱动 + 预处理对齐；改动理由见 `docs/plan.md` §3 决策日志 |
| 5. 结果对比 | `runs/sweep/cross_all_ops.csv` / `summary.json` 自动产出；`scripts/build_report.py` 直接写进 `1_报告.docx` 的表 5；表格模板见 `docs/plan.md` §4 |

第 4 节能写满 300 字不是靠堆词，而是因为"把写死的算子族化 + 分离两种消融语义"本身就是
一个需要解释的设计决策。
