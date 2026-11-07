# 项目计划与决策日志

> 状态：骨架完成并**已实机验证**。torch 路径（算子族、模型、训练、评测、消融驱动）已在
> CPU torch 2.15.0+cpu 上跑通：`smoke_test.py` 49/49 通过，`train.py` / `test.py --cross
> --robustness` / `sweep.py --mode pretrained` 在合成数据上端到端出表。
> **未做的**：真实数据集上的任何实验（W1 起）。

## 1. 研究问题

NPR（CVPR 2024）在输入端放了一个固定高通残差算子：

```python
NPR = x - nearest_upsample(nearest_downsample(x))
x = conv1(NPR * 2.0 / 3.0)
```

论文把泛化归因于"上采样算子产生的伪影"，但没做**算子族的系统消融**：换成别的高通会怎样？
本项目只做这一件事，把它测成一张表。

**为什么这个改动值得做**（而不是随便加个模块）：
原计划"给 NPR 并联一个 DCT/FFT 分支再融合"是错的，因为 NPR 已经是高通，
再加一路高通大概率零增益甚至负增益。把一个既有假设从"没被验证"推到"可测量"，
比堆一个新模块更接近研究，也更好写报告。

## 2. 数据

| 用途 | 数据集 | 来源 | 规模 |
| --- | --- | --- | --- |
| 训练/验证 | ForenSynths（4 类：car/cat/chair/horse） | CNNDetection 百度盘<br>`https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b` | 20 类 × 36k 图，取 4 类子集 |
| 泛化测试 | ForenSynths_test + GANGen-Detection + DiffusionForensics + UniversalFakeDetect + Diffusion1kStep | 同上前者；Diffusion1kStep 在<br>`https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq` | 16+ 生成器 |
| 不推荐 | GenImage | arXiv 2306.08571 | 百万级，8GB 显存 + 消费级带宽不划算 |

训练类别与测试生成器**不重叠**（ProGAN 系列训练 → StyleGAN/ADM/SD/Glide 测试），
避免数据泄漏：ForenSynths 的 20 类共用同一批真实源图，所以"真实图"在 train/test 间必须
按官方划分走，不能自己重新随机切。

官方压缩包的目录约定与本项目要求不一致（`0_real`/`1_fake`、按 benchmark 嵌套、
ForenSynths 用物体类别当生成器名），统一用 `scripts/prepare_data.py` 转换并体检：

```bash
python scripts/prepare_data.py organize --src <下载目录> --dst datasets/... [--generator ProGAN]
python scripts/prepare_data.py verify  --root datasets/... --leakage-with <训练集>
python scripts/prepare_data.py manifest --root datasets/... --out manifests/xxx.csv
```

`verify --leakage-with` 用哈希查 train/test 重叠，这一条正是报告第 3 节能写"我们确认过无泄漏"
的依据；`manifest` 出的 CSV 可直接当附录。

## 3. 决策日志

| # | 决策 | 备选与被否原因 |
| --- | --- | --- |
| D1 | 选方向 B（AIGC 频域取证） | A（VLM-IQA）需要大模型 API 或 7B 权重，8GB 显存跑不动评测；C/D/E 缺"公开 baseline 数字"或代码改不动。B 有 NPR 现成权重 + 公开对照表 + 1.44M 参数可训。 |
| D2 | fork NPR 作主干，而非 CNNDetection | CNNSpot 在扩散模型上 50.5%（≈随机），拿它当 baseline 起点太低；NPR 91.7% 才有"往下掉多少"的意义。 |
| D3 | **推翻**"加频域分支 + 门控融合" | 读源码后发现 NPR 本身就是高通，并联第二路高通预期零增益。改为算子族消融。理由：原来的设计建立在"NPR 缺频域信息"这个错误前提上。 |
| D4 | 自实现 ResNet 而非 import torchvision | 官方是裁剪版（只有 layer1/layer2，分类头叫 `fc1`，输入 512）。用 torchvision 的 `resnet18(layers=[2,2])` 对不上权重键名，也拿不到 1.44M 参数。 |
| D5 | 预处理单独成文件并逐行对齐官方 | 预处理不一致是最常见的"复现不出来"原因；官方 `RandomCrop` ≠ `RandomResizedCrop` 这种细节会直接毁掉可比性。 |
| D6 | 标签钉死 real=0/fake=1，不用 ImageFolder 默认排序 | `ImageFolder` 按字母序给出 `fake=0, real=1`，会让 AUC 反向、AP 全错，且不报错。 |
| D7 | 区分 `pretrained` 与 `scratch` 两种消融语义 | 冻结 `conv1` 换算子是"探针"，混杂输入分布偏移；干净结论必须重训。两者都保留，但报告里必须说明是哪一种。 |
| D8 | metrics 用纯 numpy 而非 sklearn | 环境里没有 sklearn，而 AUC/AP 各 20 行就能写对，且可以用暴力枚举交叉验证（已在 smoke test 里做）。 |
| D9 | smoke test 不依赖数据集与 torch | 否则在无 GPU 环境无法做任何验证。无 torch 时明确 SKIP 而非假装 PASS。 |
| D10 | 只依赖 torch + Pillow + torchvision.transforms | 去掉 `torchvision.datasets` 的使用（改成 3 行 PIL 读图），因为标签必须钉死，用不上 ImageFolder。少一个依赖就少一个坏点。 |
| D11 | real/fake 配对规则抽到 `src/layout.py`，loader 与 `scripts/prepare_data.py` 共用 | 两处各写一份判定，迟早会让"准备出来的目录"和"loader 认识的目录"错开；而且"像 fake 的名字"有两类（`fake` 是类目录，`ProGAN` 是生成器名），分散实现必然有一边把生成器当类目录切。 |
| D12 | `organize` 默认 `--mode auto`＝同卷优先硬链接 | 数据集 30GB+，复制一份就是纯浪费；代价是暂存文件与原文件同一 inode（就地重写会改到原文件），已在文档与 smoke test 里显式断言这件事，需要独立副本时用 `--mode copy`。 |
| D13 | `organize` 遇冲突改名 `__dup2`，绝不覆盖 | 拷贝中断后重跑必须安全，且"内容不同"意味着两次下载不一致——覆盖会静默丢数据，改名让用户能看见。 |
| D14 | 作业报告由 `scripts/build_report.py` 生成，而非手写 docx | 表 5 的五列要在跑完 sweep 后填数，手抄必错且不可复现；生成器还能保证"重跑一次报告＝同一次结果"。代价是要处理 docx 细节（模板只有 `Normal Table` 样式、正文要把 `w:sectPr` 之外的 body 全清再写、表格不跨页要点 `cantSplit`）。 |
| D15 | 表 5 的"预期/机制依据"两列预注册在 `EXPECTED` 字典里 | 结果出来之后再解释机制，任何假设都能自圆其说；先写下"换成 bilinear 应该掉点"再跑，才构成可证伪的检验。没有实测时这两列照常显示，实测两列写 `待补`。 |
| D16 | 新增 `scripts/download_data.py` 把"下载→解压→摆位→体检"串成一条命令，而非只给 README 里的 gdown 手工步骤 | 官方三个包的形状各不相同（`0_real`/`1_fake`、嵌套 benchmark、生成器内部再按物体类别分子目录、`guided` 之类别名），手工 mv 一遍要 20 分钟且极易把 ProGAN 的 car/cat 当成两个生成器；脚本把这段固定下来，并复用 `prepare_data.py` 的配对/去重/体检逻辑而不是重写一份。下载本身仍可能在校园网失败，故保留 `--local-dir`/`--local-archive` 两条离线入口。 |
| D17 | 拿不到训练集时，消融只做 `--mode pretrained` 探针，并**在报告里明写它是什么** | 备选是"先在 AIGI_test 的某个子集上自训一个模型再消融"，被否：那会把"评测集"当训练集用，直接污染后面所有跨生成器结论，而且与官方权重的数字不可比。探针能回答的问题是"同一个分类头遇到不同前置算子会怎样"，这恰好是假设检验需要的；它的局限（`conv1` 没见过新算子分布）写进报告局限一节，而不是藏起来。 |
| D18 | 评测返回**原始 logit**，阈值默认 0.0，并一律同时报 Acc/AUC/AP | 官方权重分数被 sigmoid 压饱和（见 §3.1 第 11 条）后，AUC 会假性 0.5；只报 Acc 又会让"阈值没校准"和"排序能力差"混为一谈。三者同报，读者才能分清是判别力问题还是校准问题。 |

### 3.1 实测中修掉的错误（都被测试逮住了）

自检和执行的价值就在这里——以下 8 个 bug 全都"看起来是对的"：

1. `build.py` 的 `REPO_ROOT` 多向上取了一层，指向项目**父目录**，导致 `configs/default.yaml`
   找不到、所有相对数据路径静默偏移。修：两级 dirname。
2. `build.py` 读 `data.num_workers`，而配置和 CLI 用的是 `workers` → 用户在配置里改 workers
   完全无效，永远走默认 4。修：`workers` 为准、`num_workers` 作别名。
3. `_gaussian_blur` 用零填充 → 常数图在四边"变暗"，Laplacian 残差在图像边缘凭空造出一圈
   假边（max 残差 0.255）。对一个取证用高通算子来说，这等于自己伪造了伪影。修：replicate 填充。
4. `DCTLowpassOp` 的逆变换第一个 einsum 写成 `"iu,..."`，等价于用了 `B·X` 而不是 `Bᵀ·X`；
   DCT 矩阵**不对称**，所以常数图重建不回去（残差 0.462）。修：`"ui,..."` 并补注释说明原因。
5. `pair_class_dirs` 的兜底分支原写作"第二个名字像 fake 就猜" → `["ProGAN","StyleGAN"]`
   两个生成器会被切成 real/fake 对，整份测试集的标签翻转且不报错。修：只允许
   "恰好一侧可识别 + 另一侧不可识别"时猜测，两侧同向即判歧义返回 `None`。
6. `--mode auto` 硬链接下，smoke test 自己的"改坏目标文件"操作同时改坏了源文件
   （同一 inode），于是"该改名"用例永远走"跳过"。修测试：先 `os.remove` 断开链接再写；
   同时把"暂存文件就是源文件"这条断言加进去，让这个取舍始终可见。
7. `download_data.py` 的 `stage()` 里，内层 `for src in files:` 的循环变量和外层生成器名
   `name` 重名（Python 是函数作用域），生成器名被覆盖成最后一个文件名——`real` 侧写到正确
   目录，`fake` 侧写成 `Generalization_Test/img_2.png/fake/`，体检只报"没找到生成器"。
   修：生成器名 `out_name`、文件名 `base`、循环变量 `path`，三者不共用名字。
8. `plan_generators` 的"兄弟目录都是 real/fake 对就合并到父目录"这条规则，在
   `AIGCDetectBenchmark/{ADM,Glide}/{real,fake}` 上会把两个**不同生成器**合成一个名叫
   `AIGCDetectBenchmark` 的假生成器（而 `ProGAN/{car,cat}/...` 又确实该合并）。结构上两者
   同形，只能靠名字判：父目录名是已知生成器名 → 合并（其子目录是物体类别）；子目录名是已知
   生成器名 → 不合并（父目录只是 benchmark 容器）。同时 `collect_pairs` 补一条结构守卫：
   两侧都不含图片却有子目录 → 不是类别对，别在这层猜。

教训：这类错误不会抛异常，只会让所有后续数字悄悄偏掉。所以 smoke test 里专门断言
"常数图必须被每个高通算子归零"——它同时抓住了第 3 和第 4 个 bug；第 5 个则靠
"两个生成器名必须判歧义"这条反向断言。

9. **（用法而非代码，但同样静默）**`src.test` 的 `--test-root` 与 `--config` 是分开的：
   只给 `--test-root` 时，crop/load/normalise 仍取自 `configs/default.yaml`（224/256）。
   用 64×64 的合成数据跑，图像被插值放大到 256 再裁 224，最近邻网格被抹平，
   `auc` 从该有的 100 掉到 8–19，而**既不报错也不警告**——一度被误判成 `metrics.auc`
   在并列值上出错（单独调用 `metrics.auc` 与暴力实现都是 1.0，才排除）。
   修：`test.py` 在"给了 `--test-root` 却没给 `--config`"时打印醒目警告并回显当前
   crop/resize 取值。

10. **acc@0.5 在饱和分数上是废指标。**合成数据上 4 个生成器 AUC 全是 100、AP 全是 100，
    但 `acc` 一律 50.00：所有分数都挤在 0.99 以上，固定阈值 0.5 把每张图都判成 fake。
    所以主指标取 **AUC / AP**，`acc` 只作参考，另报 `acc_best`（阈值搜索）与
    `tpr@fpr5`。报告第 5 节引用 acc 时必须同时说明阈值。

以上 10 条都在合成数据阶段暴露。接上真实数据（`AIGI_test`，152,598 张）和官方权重
（`NPR.pth`）后又出现 3 条，全都是"只有真数据 + 真权重才会触发"的：

11. **logit 饱和把信号全抹掉。** 官方权重在真实图上输出的 logit 量级高达数百，
    `sigmoid` 在 |logit| > 17 之后在 float32 里就饱和成 1.0，于是所有样本得到同一个分数：
    AUC 假性 ≈ 0.5、Acc 假性 50%，而程序一句警告都没有（这正是更早那次合成数据上
    "acc 一律 50.00"的真因，当时误判成并列值）。修：`src/eval/engine.py` 的
    `collect_scores` 改为返回**原始 logit**（两个输出时取 `logits[:,1]-logits[:,0]`），
    新增 `DEFAULT_THRESHOLD = 0.0`；所有调用方（`evaluate_loader` / `cross_generator` /
    `robustness`）的默认阈值同步改成 0.0。教训：**评测层不要过 sigmoid**，单调变换
    对 AUC/AP 无意义，却会因浮点饱和丢掉全部排序信息。

12. **`--max-batches` 会只截到一类。** 数据集里 real 文件排在 fake 前面，按 batch 截断时
    前若干个 batch 全是实类 → 只有一类样本，`auc` 直接是 `nan`，而指标函数照样返回。
    修：数据集新增 `limit_per_class`，**每类各截 N 张**（`--limit-per-class`），
    `build_eval_loader` 读 `cfg["data"]["eval_limit_per_class"]`；
    `cross_generator` 在某生成器只有一类样本时打印 `[warn] ... use --limit-per-class`。

13. **一条坏 JPEG 能崩掉整轮 34 GB 的评测。** 真实测试集里有尾部缺字节的文件：
    `OSError: image file is truncated (18 bytes not processed)`，一个文件让整次 18 分钟的
    评测白跑。修：`src/data/datasets.py` 顶上显式 `ImageFile.LOAD_TRUNCATED_IMAGES = True`
    （公开数据集常见缺陷，PIL 补齐即可，业内常规处理），同时把读取包进 try/except 并
    `raise RuntimeError(f"unreadable image: {path} ({exc})")` —— 万一真要失败，**报错里必须
    带文件名**，否则只能靠二分猜。

14. **`_family()` 把扩散模型判成 GAN。** 名字表只列了 `sdv4`/`adm` 这类短名，
    而真实目录叫 `stable_diffusion_v_1_4`、`coco_sdxl_nw` → 被归进 GAN 组，
    "GAN vs 扩散"的分组均值直接错掉（这是报告里最重要的一个对照）。
    修：改为字母数字归一化后匹配集合 + `("diffusion","sdxl","dalle","midjourney")` 子串兜底。

## 4. 实验矩阵与报告表格模板

### 4.1 主表：算子 × 生成器（Acc%）

```bash
python -m src.sweep --mode scratch --ops all --epochs 50 --out-dir runs/sweep
```

| 算子 | 平均 | GAN 均值 | 扩散均值 | ProGAN | StyleGAN | ADM | Glide | Midjourney | SDv1.4 | DALLE2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `npr_nearest` | | | | | | | | | | |
| `npr_bilinear` | | | | | | | | | | |
| `npr_bicubic` | | | | | | | | | | |
| `npr_box` | | | | | | | | | | |
| `laplacian` | | | | | | | | | | |
| `dct_lowpass` | | | | | | | | | | |
| `fft_radial` | | | | | | | | | | |
| `identity` | | | | | | | | | | |

**读法**：若 `laplacian`/`dct`/`fft` 与 `npr_nearest` 差距 < 2pp → 支持"纯高通说"，
整个"频域指纹"叙事要被削弱；若只有 `nearest` 撑着、其余全线掉 > 10pp → 支持"算子匹配说"，
NPR 的卖点其实是一个上采样核的巧合，这是更有意思的结论。

### 4.2 副表：鲁棒性（Acc%，JPEG / 缩放）

```bash
python -m src.test --ckpt runs/sweep/train/<op>/best.pth --robustness --out-dir runs/robust/<op>
```

| 算子 | clean | JPEG95 | JPEG75 | JPEG50 | 0.5× resize |
| --- | --- | --- | --- | --- | --- |
| `npr_nearest` | | | | | |
| `laplacian` | | | | | |

**预测**：JPEG 直接量化 DCT 系数，正好打在频域指纹所在的频段。若某算子的优势在 JPEG50 下
完全消失，说明它靠的就是那一小段频域信息，而不是"理解了什么"。

### 4.3 训练无关基线（可选，但很加分）

`src/eval/spectrum.py` 的 `hf_energy_ratio` 是一个**不训练**的一维特征，直接阈值化就能做检测。
把它和 1.44M 参数的 NPR 放在同一张表里，就有了"大模型 vs 一个数字"的对照——
这正好呼应作业"与大模型相关"的要求，也是最能讲故事的一行。

```python
from src.eval.spectrum import hf_energy_ratio
score = hf_energy_ratio(img_array)   # 单张图，无需 GPU
```

## 5. 四周计划

| 周 | 目标 | 验收 |
| --- | --- | --- |
| W1 | 装环境、下数据、`smoke_test --require-torch` 全过、官方 `NPR.pth` 零样本复现 | nearest 的跨生成器均值落在 85–95% 区间（论文 91.7）。偏离 > 5pp 先查预处理 |
| W2 | `pretrained` 消融跑完 8 个算子（便宜） | 出一张 8×16 表 + 初步结论；确认是否值得进 W3 |
| W3 | 选 3–4 个最有信息量的算子做 `scratch` 重训；跑鲁棒性矩阵 | 主表 + 副表齐；每个数字都能追到 CSV |
| W4 | 写报告五节 + 图（径向功率谱 real vs fake、算子对比柱状图） | 第 4 节能说清"改了什么、为什么这么改"；第 5 节每个数字有出处 |

**成本估算**：单次 50 epoch / 4 类 / batch 32 在 8GB 笔记本 3060–4060 上约 6–10 小时。
所以 W3 只能选 3–4 个算子，不能全训。W2 的 `pretrained` 模式每个算子只要一次前向扫描
（约 10–20 分钟），是性价比最高的第一步。

## 6. 风险与对策

| 风险 | 症状 | 对策 |
| --- | --- | --- |
| 预处理不一致 | 复现 nearest 只有 70% 而不是 90% | 对照官方 `data/datasets.py`；先只跑一个生成器（ProGAN）确认 |
| 算子里程碑量纲不匹配 | `laplacian`/`dct` 一上来就崩到 50% | 先测各算子在真实 batch 上的残差 std；差 > 2× 就做归一化并记录（`configs/operators.yaml` 末尾有告警） |
| 数据泄漏 | 测试集真实图混进训练 | 严格用官方 train/val/test 划分，不自己重切 |
| 显存不足 | OOM at batch 32 | 降到 16 + 梯度累积；224 输入本身不大，1.44M 参数模型很轻 |
| torch 依赖漂移 | 官方 requirements 对应 torch 1.8 | 本项目只用到 `interpolate/conv2d/fft/einsum`，torch ≥ 1.10 即可，未锁死旧版 |
| 结论不显著 | 算子间差异落在噪声内 | 这本身是有效结论（支持"两者都不是"），报告里如实写；用多个随机种子给出波动范围 |

## 7. 待验证项（TO VERIFY）

- [ ] `translate_duplicate` 与原实现的逐像素一致性（当前按 README 文字重建）
- [ ] 官方论文表格用的是 `testmodel()`（no_crop=True）还是 test 脚本的 CenterCrop(224)
- [x] ~~各算子残差 L2 范数的实际比值，是否需要归一化后再比较~~ —— smoke test 已确认
      每个算子的 `scale` 取值与常数图归零行为；量纲差异仍需在真实 batch 上量，见 `configs/operators.yaml`
- [ ] 官方 `Generalization_Test` 的原始层级结构（当前要求整理成 `real/fake` 两层）
- [x] ~~在 8GB 显存上 batch 32 / 224 输入是否 OOM~~ —— 模型 1,437,761 参数，CPU 上前向已跑通；
      GPU 显存不是瓶颈（输入 224×224×3、batch 32 ≈ 19MB）
- [ ] 真实数据集上的复现：官方 `NPR.pth` 零样本跨生成器是否落在论文 91.7% 附近（W1 的验收项）
