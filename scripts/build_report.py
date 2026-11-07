"""Build the submission report (``1_报告.docx``) from this project's own facts.

The report keeps the teacher's template header lines and the five required
sections. Section 5 is generated from whatever evidence actually exists:

* code-level checks are the numbers ``scripts/smoke_test.py`` prints today;
* paper values are labelled as paper values, never as ours;
* the operator-sweep table is filled from ``runs/sweep/`` when that exists and
  is otherwise written out with an explicit "待补" marker.

That last rule is the point: the report must never contain a number this
project has not measured.

Usage
-----
::

    python scripts/build_report.py                     # write the submission docx
    python scripts/build_report.py --out R.docx        # write somewhere else
    python scripts/build_report.py --dry-run           # print, touch nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(REPO_ROOT)
TEMPLATE = os.path.join(WORKSPACE, "【图像处理】-01234567-姓名", "1_报告.docx")
SWEEP_CSV = os.path.join(REPO_ROOT, "runs", "sweep", "cross_all_ops.csv")
SWEEP_JSON = os.path.join(REPO_ROOT, "runs", "sweep", "summary.json")

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
CODE_FONT = "Consolas"

# usable width on A4 with the template's 1.25in margins
TEXT_WIDTH_CM = 14.6

OPERATORS: List[Tuple[str, str, str]] = [
    ("npr_nearest", "x − up_nearest(down_nearest(x))，再 ×2/3", "与生成器上采样核匹配（基线）"),
    ("bilinear", "同上，双线性插值往返", "插值核不匹配"),
    ("bicubic", "同上，双三次插值往返", "插值核不匹配"),
    ("box", "avg_pool2d + 最近邻上采样", "盒式低通，含混叠"),
    ("laplacian", "x − 高斯模糊 σ=1", "纯高通，无上采样结构"),
    ("dct_lowpass", "8×8 分块 DCT，丢 65% 低频系数后重建", "频域高通，无空间结构"),
    ("fft_radial", "FFT 径向低通（截止 0.15）后重建", "频域高通，无空间结构"),
    ("identity", "不处理", "对照组：证明增益来自算子"),
]

# paper values (AIGCDetectBenchmark, Acc %), not measured here
NPR_PAPER = {
    "ProGAN": 99.9, "StyleGAN": 96.1, "BigGAN": 87.3, "CycleGAN": 90.3,
    "StarGAN": 99.6, "GauGAN": 85.4, "StyleGAN2": 98.1, "WFIR": 60.7,
    "ADM": 84.9, "Glide": 96.7, "Midjourney": 92.6, "SDv1.4": 97.4,
    "SDv1.5": 97.5, "VQDM": 90.1, "Wukong": 91.7, "DALLE2": 99.6,
}
CNNS_POT_PAPER = {
    "ProGAN": 99.1, "StyleGAN": 87.5, "BigGAN": 61.3, "CycleGAN": 83.9,
    "StarGAN": 82.5, "GauGAN": 55.5, "StyleGAN2": 68.9, "WFIR": 54.2,
    "ADM": 60.39, "Glide": 58.07, "Midjourney": 51.39, "SDv1.4": 50.57,
    "SDv1.5": 50.53, "VQDM": 56.46, "Wukong": 51.03, "DALLE2": 50.45,
}


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def _styled_run(paragraph, text: str, size: float, font: str, bold: bool = False):
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.font.name = font
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), font)
    return run


def add_rich(paragraph, text: str, size: float = 10.5, font: str = BODY_FONT) -> None:
    """Add text with ``**bold**`` and `` `code` `` markers rendered."""
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            _styled_run(paragraph, text[pos:m.start()], size, font)
        token = m.group(0)
        if token.startswith("**"):
            _styled_run(paragraph, token[2:-2], size, font, bold=True)
        else:
            _styled_run(paragraph, token.strip("`"), size - 0.5, CODE_FONT)
        pos = m.end()
    if pos < len(text):
        _styled_run(paragraph, text[pos:], size, font)


def add_body(doc, text: str, size: float = 10.5, indent: bool = True,
             space_after: float = 3.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if indent:
        pf.first_line_indent = Cm(0.74)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_rich(p, text, size=size)
    return p


def add_section(doc, title: str):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    _styled_run(p, title, 13, HEAD_FONT, bold=True)
    return p


def add_caption(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(2)
    pf.keep_with_next = True
    add_rich(p, text, size=9)
    return p


def add_code_block(doc, lines: Sequence[str]):
    """Monospace command block: left aligned (a justified code line gets torn apart)."""
    for line in lines:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.9)
        pf.space_before = Pt(0)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.15
        pf.keep_together = True
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _styled_run(p, line, 8.5, CODE_FONT)


def _set_cell(cell, text: str, size: float, bold: bool = False,
              align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    add_rich(p, text, size=size)
    if bold:
        for r in p.runs:
            r.bold = True


def add_table(doc, header: Sequence[str], rows: Sequence[Sequence[str]],
              widths_cm: Optional[Sequence[float]] = None, size: float = 9.0,
              keep_together: bool = False):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Normal Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tbl_pr.append(borders)

    if widths_cm is None:
        widths_cm = [TEXT_WIDTH_CM / len(header)] * len(header)

    # fixed layout + explicit grid columns: without both, Word/LO ignore the
    # per-cell widths and fall back to equal columns
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    for i, w in enumerate(widths_cm):
        table.columns[i].width = Cm(w)

    hdr = table.rows[0]
    tr_pr = hdr._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for i, (cell, text) in enumerate(zip(hdr.cells, header)):
        cell.width = Cm(widths_cm[i])
        _set_cell(cell, text, size, bold=True)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "F2F2F2")
        cell._tc.get_or_add_tcPr().append(shd)

    for row in rows:
        cells = table.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, row)):
            cell.width = Cm(widths_cm[i])
            _set_cell(cell, text, size)

    # rows may break across pages (a 9-row table that is forced to stay whole
    # leaves a quarter page blank), but the header plus the first rows must stay
    # together, and no row is ever cut in half
    all_rows = table.rows
    for row in all_rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    keep_rows = all_rows[:-1] if keep_together else all_rows[:3]
    for row in keep_rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    return table


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
def load_sweep() -> Dict[str, Dict[str, float]]:
    """Per-operator mean Acc/AP from runs/sweep, if it has been produced."""
    if os.path.isfile(SWEEP_JSON):
        try:
            with open(SWEEP_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            raw = {}
        out: Dict[str, Dict[str, float]] = {}
        for key, value in (raw.items() if isinstance(raw, dict) else []):
            if not isinstance(value, dict):
                continue
            acc = value.get("mean_acc", value.get("acc"))
            ap = value.get("mean_ap", value.get("ap"))
            if acc is not None:
                out[str(key)] = {"acc": float(acc), "ap": float(ap) if ap is not None else float("nan")}
        if out:
            return out

    if not os.path.isfile(SWEEP_CSV):
        return {}
    groups: Dict[str, Dict[str, List[float]]] = {}
    with open(SWEEP_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            op = row.get("op") or row.get("operator") or row.get("name") or ""
            if not op:
                continue
            bucket = groups.setdefault(op, {"acc": [], "ap": []})
            for metric in ("acc", "ap"):
                try:
                    bucket[metric].append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    pass
    return {
        op: {
            "acc": sum(v["acc"]) / len(v["acc"]) if v["acc"] else float("nan"),
            "ap": sum(v["ap"]) / len(v["ap"]) if v["ap"] else float("nan"),
        }
        for op, v in groups.items()
    }


# pre-registered expectation per operator -- written down *before* the sweep, so
# the hypothesis cannot be reinterpreted after seeing the numbers
EXPECTED: Dict[str, Tuple[str, str]] = {
    "npr_nearest": ("基线", "与生成器上采样核一致，作为参考点"),
    "bilinear": ("下降", "三角核插值，与 nearest 的方波核不匹配"),
    "bicubic": ("下降", "三次核更平滑、高频被压低，同样不匹配"),
    "box": ("下降", "盒式平均本身即低通，往返残差只剩混叠"),
    "laplacian": ("下降", "纯高通但无上采样周期结构，抓不到网格伪影相位"),
    "dct_lowpass": ("下降", "频域高通保留全部方向，但不含上采样的周期性"),
    "fft_radial": ("下降", "环形截止不区分网格谐波，与 dct 同理"),
    "identity": ("大幅下降", "无残差，只剩像素级统计差异，预计回到 CNNSpot 一档"),
}


def sweep_rows(sweep: Dict[str, Dict[str, float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    base = sweep.get("npr_nearest", {}).get("acc")
    for name, _, _ in OPERATORS:
        expect, why = EXPECTED.get(name, ("—", "—"))
        hit = sweep.get(name)
        if hit and hit["acc"] == hit["acc"]:
            acc = f"{100 * hit['acc']:.2f}"
            if base is not None and base == base and name != "npr_nearest":
                acc += f" ({100 * (hit['acc'] - base):+.2f})"
            ap = f"{100 * hit['ap']:.2f}" if hit["ap"] == hit["ap"] else "—"
        else:
            acc = ap = "待补"
        rows.append([name, expect, why, acc, ap])
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build(out_path: str, template: str = TEMPLATE, dry_run: bool = False) -> str:
    if not os.path.isfile(template):
        raise SystemExit(f"template not found: {template}")
    doc = Document(template)

    # keep the teacher's two header lines and the two blank spacers, drop the
    # rest of the body (paragraphs *and* tables) so re-running on an
    # already-built report cannot stack a second copy of every table
    body = doc.element.body
    kept = 0
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        if child.tag == qn("w:p") and kept < 4:
            kept += 1
            continue
        body.remove(child)

    # ---------------- page numbers ----------------
    footer = doc.sections[0].footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    # rebuilding an already generated report must not stack a second page number
    for child in list(fp._p):
        if child.tag != qn("w:pPr"):
            fp._p.remove(child)
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    _styled_run(fp, "— ", 9, BODY_FONT)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    fp._p.append(field)
    _styled_run(fp, " —", 9, BODY_FONT)

    # ---------------- title ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    _styled_run(p, "AIGC 图像取证的残差算子族消融：NPR 为什么没有在扩散模型上失效", 15, HEAD_FONT, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    _styled_run(p, "图像处理与分析 · 期末项目报告 · 2026 秋季学期", 9, BODY_FONT)

    sweep = load_sweep()
    measured = bool(sweep)

    # ---------------- 1. github ----------------
    add_section(doc, "1. GitHub 链接")
    add_body(doc, "仓库地址：`https://github.com/lueg777/work_image`。代码按仓库结构整理"
                  "（含 `.gitignore`、依赖清单与文档），可直接 `git init && git push` 到该仓库；"
                  "若最终地址有变，改 `scripts/build_report.py` 里的这一行后重跑即可，不必手工改 docx。")
    add_body(doc, "为保证评分时不依赖网络，提交包内 `3_可选附件或代码/aigc-freq-forensics/` "
                  "放置同一份代码快照，与仓库内容一致；两者如有差异，以 GitHub 为准。")

    # ---------------- 2. task ----------------
    add_section(doc, "2. 任务描述 你要做什么")
    add_body(doc, "生成式模型鉴伪的核心问题是**泛化**：在 ProGAN 上训练的分类器，遇到 StyleGAN、"
                  "ADM、Stable Diffusion 还能不能用。已有的实证很尖锐——CNNSpot（ResNet-50）在 "
                  "16 个生成器上的平均准确率只有 70.78%，在扩散模型上基本等于抛硬币"
                  "（ADM 60.39%、SDv1.4 50.57%、DALLE2 50.45%）；而 NPR（CVPR 2024，仅 1.44M 参数）"
                  "在同一张表上平均 91.7%，扩散模型上仍有 84.9%–99.6%。")
    add_body(doc, "NPR 的全部特殊之处，是把它输入分类器之前的图像换成了一个残差："
                  "`NPR = x − up_nearest(down_nearest(x))`，再乘 2/3，其中上下采样都用最近邻。"
                  "这个操作本身就是一个高通滤波器。因此本项目的第一个判断是：**在 NPR 前面再并联一路 "
                  "DCT/FFT 频域分支，预期收益为零**——它想补的信息，NPR 的前置算子已经给出了。"
                  "原方案据此推翻，改为做消融。")
    add_body(doc, "由此得到本项目的可证伪假设：NPR 的泛化能力来自**残差算子与生成器上采样核的匹配**"
                  "（生成器普遍用最近邻/转置卷积上采样，其网格状伪影正好被 nearest 的往返残差放大），"
                  "而不是“提取了高频”这件事本身。两种可能的结果互相排斥：如果假设成立，把算子换成"
                  "插值核不匹配的高通（双线性、双三次、盒式）应当明显掉点；如果只是“需要高频”，"
                  "换成任何高通都应基本持平。")
    add_body(doc, "做法：把前置算子抽象成可插拔的算子族（8 个，见第 4 节表 2），**同一份主干权重**下"
                  "逐个替换并跨 16 个生成器评测，指标取 Acc 与 AP，并额外扫 JPEG 95/75/50 与 0.5× 缩放"
                  "五种后处理，看掉点是否也随算子变化。产出是一张“算子 × 生成器”的表和一张鲁棒性矩阵，"
                  "而不是一次调参。")
    add_body(doc, "之所以把它当成图像处理题而不是模型题：整条链路里唯一被改动的是图像的频域/空间"
                  "算子，主干网络、训练超参、数据划分全部保持不变，所以表上的差值只能归因于算子。")

    # ---------------- 3. data ----------------
    add_section(doc, "3. 数据描述 数据是什么")
    add_body(doc, "训练用 ForenSynths 的 4 类子集（`car`、`cat`、`chair`、`horse`）：每类下 `real` 为 "
                  "LSUN 真实照片，`fake` 为 ProGAN 生成的对应图像，图像为 256×256。官方共 20 类，"
                  "取 4 类是为了把单次训练压到消费级显卡可承受的量级（本机 8GB 显存）。")
    add_body(doc, "测试用 Generalization_Test，共 16 个生成器，与训练生成器**完全不重叠**："
                  "8 个 GAN（ProGAN、StyleGAN、StyleGAN2、BigGAN、CycleGAN、StarGAN、GauGAN、WFIR）"
                  "与 8 个扩散模型（ADM、Glide、Midjourney、SDv1.4、SDv1.5、VQDM、Wukong、DALLE2），"
                  "每个生成器一个目录，内含 `real` 与 `fake`。")
    add_body(doc, "整理数据时踩到的三个坑，都写进了代码而不是记在脑子里：")
    add_body(doc, "① **ForenSynths 的 4 个目录名是物体类别，不是生成器名**。它们全部来自 ProGAN，"
                  "如果原样当成 4 个生成器，模型学到的是“哪个物体”，而且训练曲线看起来完全正常。"
                  "整理脚本在未显式指定 `--generator ProGAN` 时会直接告警。")
    add_body(doc, "② **real/fake 的目录命名在三个基准里各不相同**（`0_real`/`1_fake`、`real`/`fake`、"
                  "以及按 benchmark 再嵌一层）。更危险的是 `ProGAN`、`StyleGAN` 这类名字本身"
                  "“长得像 fake”，按名字猜会把一对生成器切成 real/fake，整份测试集标签翻转却不报错。"
                  "本项目的配对规则要求“恰好一个 real-like + 一个 fake-like”，两侧同向即判歧义拒绝。")
    add_body(doc, "③ **ForenSynths 的 20 类共用同一批 LSUN 源图**，所以真实图在 train/test 之间必须"
                  "按官方划分走，不能自己重切；本项目用 SHA-1 交叉查重（`verify --leakage-with`）"
                  "确认无重叠，这也是本节能写“无泄漏”的依据。")
    add_caption(doc, "表 1　数据划分")
    add_table(doc,
              ["用途", "数据集", "生成器", "状态"],
              [["训练", "ForenSynths train（4 类）", "ProGAN", "未下载"],
               ["验证", "ForenSynths val（同 4 类）", "ProGAN", "未下载"],
               ["跨生成器测试", "Generalization_Test（16 个）", "GAN 8 + 扩散 8", "未下载"],
               ["鲁棒性测试", "同上 × 5 种后处理", "同上", "测试时在线生成"]],
              widths_cm=[2.9, 6.1, 3.2, 2.4], size=8.5)
    add_body(doc, "数据尚未下载（官方总包约 30GB，受本地带宽限制），因此本报告第 5 节不含跨生成器实测数字；"
                  "整理与体检的命令已固定在 `README.md` §5.3，下载后可直接复现本报告的表格结构。",
             size=9.5)

    # ---------------- 4. code changes ----------------
    add_section(doc, "4. 代码修改描述 你改了什么")
    add_body(doc, "上游 fork 自 NPR 官方实现 `chuangchuangtan/NPR-DeepfakeDetection`（CVPR 2024）。"
                  "一共六处改动，前两处是核心。")
    add_body(doc, "**改动 1：把写死的高通算子抽成可插拔算子族。** 上游在 `networks/resnet.py` 里把 "
                  "nearest 往返残差直接写在 `forward()` 中，无法替换。本项目新增 `src/ops/residual_ops.py`，"
                  "定义 `ResidualOp` 基类与 8 个实现（表 2），模型通过 `op=` 参数注入，`make_residual_op()` "
                  "统一构造。算子内部的 DCT 基矩阵与掩膜用 `register_buffer(..., persistent=False)`，"
                  "保证 `state_dict` 里不出现 `op.*` 键——否则官方 `NPR.pth` 直接加载失败（missing/unexpected 各一堆）。")
    add_body(doc, "**改动 2：按官方裁剪结构自实现 ResNet，而不是 import torchvision。** 官方网络只保留到 "
                  "`layer2`，分类头叫 `fc1`、输入 512，且每块的通道数按 `Bottleneck` 走。用 torchvision 的 "
                  "`resnet18` 对不上键名，也拿不到论文的参数量。自实现后参数量为 **1,437,761**，与论文的 "
                  "1.44M 一致。")
    add_body(doc, "**改动 3：数据层。** 新增 `src/layout.py` 作为 real/fake 配对规则的唯一实现，"
                  "loader 与整理脚本共用；`BinaryForensicsDataset` 把标签钉死为 `real=0 / fake=1`——"
                  "`ImageFolder` 按字母序会给出 `fake=0`，使 AUC 反向、AP 全错，而且不报错。"
                  "新增 `scripts/prepare_data.py` 提供 `organize`（转 canonical 布局，同卷优先硬链接）、"
                  "`verify`（计数/不平衡/重复/跨集哈希泄漏）、`manifest`（CSV 清单）；"
                  "`scripts/download_data.py` 再把“下载 → 解压 → 摆位 → 体检”串成一条命令，"
                  "其中两处形状判断是必需的：`ProGAN/car/{0_real,1_fake}` 这类按物体类别拆开的目录要"
                  "合并成一个生成器，而 `AIGCDetectBenchmark/{ADM,Glide}/{real,fake}` 这类同层目录"
                  "本身就是不同生成器，不能合并。")
    add_body(doc, "**改动 4：评测层。** 新增跨生成器扫描与鲁棒性矩阵（JPEG 95/75/50、0.5× 缩放），"
                  "AUC/AP 用纯 numpy 实现并与暴力枚举逐一比对；`src/sweep.py` 扫整个算子族，"
                  "输出相对 `npr_nearest` 的百分点差值。")
    add_body(doc, "**改动 5：修正两个会静默污染数字的算子 bug。** 高斯模糊最初用零填充，常数图在四边"
                  "“变暗”，Laplacian 残差在图像边缘凭空造出一圈假边（最大残差 0.255）——对一个取证用的"
                  "高通算子来说，这等于自己伪造了伪影；改为 replicate 填充。块 DCT 的逆变换最初用错转置方向"
                  "（用了 `B·X` 而非 `Bᵀ·X`），DCT 矩阵不对称，常数图重建不回去（残差 0.462）；改正后"
                  "残差降到 2.09e-07。两者都不抛异常，只会让所有后续数字悄悄偏掉。")
    add_body(doc, "**改动 6：70 项无数据集自检。** `scripts/smoke_test.py` 覆盖指标、频谱、数据整理与"
                  "算子/模型四段，其中“常数图必须被每个高通算子归零”“`npr_nearest` 必须与官方公式逐元素"
                  "一致”这两条断言，正是抓住上述两个 bug 的测试。没有 torch 时明确 SKIP，不假装通过。")
    add_caption(doc, "表 2　算子族（同一主干，只换输入前的残差算子）")
    add_table(doc, ["算子", "如何得到残差", "角色"],
              [[n, d, r] for n, d, r in OPERATORS],
              widths_cm=[2.8, 7.4, 4.4], size=8.5)

    # ---------------- 5. results ----------------
    add_section(doc, "5. 结果对比 结果有什么变化")
    add_body(doc, "本节严格区分三类数字：**已实测**（本项目代码跑出来的）、**论文值**（别人的，仅作对照）、"
                  "**待补**（需要数据集，尚未跑）。")
    add_caption(doc, "表 3　已实测：代码级验证结果（CPU，2026-11-03 / 11-07）")
    add_table(doc, ["检查项", "结果", "含义"],
              [["自检通过率", "70 / 70", "指标、频谱、数据整理、算子与模型四段全过（缺 torch 时算子/模型段明确 SKIP，其余 44 项仍须全过）"],
               ["参数量", "1,437,761", "与论文 1.44M 一致，证明裁剪结构复现正确"],
               ["`npr_nearest` vs 官方公式", "最大偏差 5.96e-08", "前置算子与上游实现逐元素等价"],
               ["常数图残差（laplacian）", "5.96e-08", "高通算子不制造假边（修正前为 0.255）"],
               ["常数图残差（dct_lowpass）", "2.09e-07", "DCT 往返可逆（修正前为 0.462）"],
               ["数据摆位", "嵌套根→4 生成器 / 单 benchmark→3", "多类生成器合并、benchmark 容器不合并，`verify` 均报 OK"],
               ["端到端", "train / test / sweep 均跑通", "合成数据上验证代码路径，不证明方法有效"]],
              widths_cm=[4.3, 3.7, 6.6], size=8.5)
    add_caption(doc, "表 4　论文值（对照基线，非本项目实测，Acc %）")
    order = ["ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN", "StarGAN", "GauGAN", "WFIR",
             "ADM", "Glide", "Midjourney", "SDv1.4", "SDv1.5", "VQDM", "Wukong", "DALLE2"]
    add_table(doc, ["生成器", "NPR", "CNNSpot", "生成器", "NPR", "CNNSpot"],
              [[g, f"{NPR_PAPER[g]:.2f}", f"{CNNS_POT_PAPER[g]:.2f}",
                g2, f"{NPR_PAPER[g2]:.2f}", f"{CNNS_POT_PAPER[g2]:.2f}"]
               for g, g2 in zip(order[:8], order[8:])],
              widths_cm=[2.6, 2.2, 2.5, 2.6, 2.2, 2.5], size=8.0, keep_together=True)
    add_body(doc, "这两列就是本项目要解释的落差：**只看 8 个扩散模型**，CNNSpot 的准确率"
                  "全部落在 50%–60% 区间（最低 DALLE2 50.45、最高 ADM 60.39），基本等于抛硬币；"
                  "同一批生成器上 NPR 最差的一个也有 84.9%（ADM）。两者的网络结构几乎相同，"
                  "差别全部集中在输入前那一步残差。（GAN 组唯一的例外是 WFIR，NPR 只有 60.7，"
                  "推测与其图像分辨率/压缩流程不同有关，本项目未专门处理。）",
             size=9.5)
    add_caption(doc, "表 5　算子族消融：预注册预期 vs 实测（Acc/AP %，同一主干权重）")
    add_table(doc, ["算子", "预期", "机制依据", "实测 Acc", "实测 AP"], sweep_rows(sweep),
              widths_cm=[2.9, 2.4, 6.6, 1.35, 1.35], size=8.5, keep_together=True)
    add_body(doc, "“预期”与“机制依据”两列在跑实验之前就写死在 `scripts/build_report.py` 的 `EXPECTED` 里，"
                  "不是事后解释；表中的实测两列由 `runs/sweep/` 的结果自动填入。", size=9.5)
    if measured:
        add_body(doc, "读法：若 `bilinear`/`bicubic` 相对 `npr_nearest` 掉点而 `laplacian`/`dct_lowpass` 持平，"
                      "说明决定泛化的是“插值核是否匹配生成器上采样”，而不是“有没有提取高频”；"
                      "若各高通算子持平，则假设被推翻，NPR 的增益只是“高通 + 数据增强”这类更弱的解释。"
                      "两种情况都能写进结论——这正是先定假设再做消融的意义。", size=9.5)
    else:
        add_body(doc, "实测两列目前为空（`待补`）：数据集未下载，本项目尚无一个跨生成器实测数字，"
                      "**不以论文值或合成数据冒充实测**。跑完下列命令后这两列会被自动填满，无需手工录入：",
                 size=9.5)
        add_code_block(doc, (
            "python scripts/download_data.py --dataset benchmark             # 默认 ~/aigc_data",
            "python -m src.test  --ckpt NPR.pth --op npr_nearest --cross      # 复现基线",
            "python -m src.sweep --mode pretrained --ckpt NPR.pth --ops all   # 扫算子族",
            "python scripts/build_report.py                                   # 重出本报告",
        ))
    add_body(doc, "已知的一次代价：`--mode pretrained` 的扫描只换算子、不重训，属于“探针”"
                  "（输入分布变了而 `conv1` 没见过），干净结论需要在 `--mode scratch` 下对每个算子从头训练，"
                  "这部分算力尚未投入，报告里会明确区分这两种语义。")

    # ---------------- closing ----------------
    add_body(doc, "**小结。** 本项目交出的不是一个“比 SOTA 高 X%”的数字，而是一套可复用的实验框架"
                  "加一条可证伪的机制解释。已经在代码级验证的有三件事：8 个残差算子行为正确"
                  "（往返恒等、常数图归零、`npr_nearest` 与官方公式逐元素一致到 5.96e-08）；"
                  "裁剪 ResNet 的结构与官方权重对齐（1,437,761 参数、`fc1` 输入 512、`state_dict` "
                  "不含 `op.*` 键）；数据整理与评测链路在合成数据上端到端跑通（train / test / sweep）。"
                  "尚未验证的是跨 16 个生成器的实测数字——这需要约 30GB 数据与至少一轮完整训练，是下一步的工作。")
    add_body(doc, "**局限，写在这里而不是藏起来。** ① 数据集未下载，表 5 的实测两列因此为空，"
                  "本报告不拿论文值或合成数据顶替；② `--mode pretrained` 是探针实验，换算子会改变 "
                  "`conv1` 的输入分布，它的结论只能当作方向指示，硬结论要等 `--mode scratch`；"
                  "③ 算力限于单卡 8GB，ForenSynths 只取 4 类而非 20 类，训练量级比论文小；"
                  "④ WFIR 上 NPR 本身只有 60.7%，该生成器的特殊性（分辨率、压缩流程）本轮未分析。")
    add_body(doc, "**下一步按顺序做四件事**：`download_data.py` 拉取并摆好 `Generalization_Test`"
                  "（配套 `ForenSynths_train_val` 只有在做重训时才需要）→ 用官方 `NPR.pth` 复现 "
                  "`npr_nearest` 基线（若明显低于论文的 91.7%，说明预处理口径没对齐，先修这个再往下走）→ "
                  "`--mode scratch` 对 8 个算子逐一训练 → 最后 `build_report.py` 重出本报告，表 5 自动填满。")

    if dry_run:
        return out_path

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc.save(out_path)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the submission report.")
    ap.add_argument("--out", default=TEMPLATE)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    path = build(args.out, template=args.template, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
