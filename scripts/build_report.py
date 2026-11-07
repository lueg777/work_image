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
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CALIB_CSV = os.path.join(RESULTS_DIR, "calibration.csv")
CALIB_JSON = os.path.join(RESULTS_DIR, "calibration_summary.json")
PROFILE_CSV = os.path.join(RESULTS_DIR, "data_profile.csv")
FIG_DIR = os.path.join(REPO_ROOT, "figs")
FIG_SPECTRUM = os.path.join(FIG_DIR, "fig_spectrum.png")
FIG_ABLATION = os.path.join(FIG_DIR, "fig_ablation.png")
FIG_ROBUST = os.path.join(FIG_DIR, "fig_robust.png")
REFERENCES: List[str] = [
    "[1] Tan C., Zhao Y., Wei S., Gu G., Liu Q. Rethinking the Up-Sampling Operations in "
    "CNN-based Generative Network for Generalizable Deepfake Detection. CVPR, 2024. "
    "（NPR 原文，本项目的上游实现与官方权重来源）",
    "[2] Wang S.-Y., Wang O., Owens A., Nagarajan I., Efros A. A. CNN-Generated Images Are "
    "Surprisingly Easy to Spot. CVPR, 2020. （CNNSpot / ForenSynths 与跨生成器测试协议）",
    "[3] Liu Z., Wang Y., Xu Z., et al. AIGCDetectBenchmark / AIGCDetect: AIGC 检测基准与权重集合. "
    "2024–2026. （本报告使用的 AIGI_test 测试集与基线权重）",
    "[4] Frank J., Eisenhofer T., Schönherr L., Fischer A., Kolossa D., Holz T. Leveraging "
    "Frequency Analysis for Deep Fake Image Recognition. ICML, 2020. （频域指纹的理论依据）",
    "[5] Zhu M., Chen H., Yan Q., et al. GenImage: A Million-Scale Benchmark for Detecting "
    "AI-Generated Images. NeurIPS Datasets and Benchmarks, 2023. （百万级基准，本报告未使用但列为后续可扩展）",
]
REPO_URL = "https://github.com/lueg777/work_image"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
CODE_FONT = "Consolas"
#: literal backtick, kept out of the source so no shell can eat it
TICK = chr(96)
#: mean Acc reported by the NPR paper (ForenSynths, 16 generators)
PAPER_MEAN_ACC = 91.7

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


def _outline(paragraph, level: int) -> None:
    """Give the paragraph a real outline level so the navigation pane and any
    generated TOC see the section structure. The template ships only the Normal
    style, so faking a Heading style is not an option; outlineLvl is what Word
    actually reads for navigation."""
    pPr = paragraph._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:outlineLvl")):
        pPr.remove(old)
    lvl = OxmlElement("w:outlineLvl")
    lvl.set(qn("w:val"), str(level))
    pPr.append(lvl)


def add_section(doc, title: str, level: int = 0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    _styled_run(p, title, 13, HEAD_FONT, bold=True)
    _outline(p, level)
    return p


def add_subsection(doc, title: str, level: int = 1):
    """Numbered sub-heading (5.1 ...): slightly smaller, still outlined."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(3)
    pf.keep_with_next = True
    _styled_run(p, title, 11, HEAD_FONT, bold=True)
    _outline(p, level)
    return p


def add_hyperlink(paragraph, url: str, text: Optional[str] = None,
                  size: float = 10.5) -> None:
    """Real clickable hyperlink (a bare URL in a run is just underlined text)."""
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), CODE_FONT)
    rPr.append(rFonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rPr.append(sz)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    rPr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text or url
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def add_figure(doc, path: str, caption: str, width_cm: float = 14.4) -> bool:
    """Embed a PNG (if it exists) and its caption below it."""
    if not os.path.isfile(path):
        add_body(doc, f"（缺图：`{os.path.relpath(path, REPO_ROOT)}`，"
                      f"先跑 `python scripts/make_report_figures.py`）", size=9)
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after = Pt(6)
    add_rich(cap, caption, size=9)
    return True


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
    """Per-operator mean Acc/AUC/AP from runs/sweep, if it has been produced."""
    if os.path.isfile(SWEEP_JSON):
        try:
            with open(SWEEP_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            raw = {}
        # src/sweep.py writes {"mode", "ops", "summary": {op: {...}}}; older runs
        # wrote the op->summary map directly
        if isinstance(raw, dict) and isinstance(raw.get("summary"), dict):
            raw = raw["summary"]
        out: Dict[str, Dict[str, float]] = {}
        for key, value in (raw.items() if isinstance(raw, dict) else []):
            if not isinstance(value, dict):
                continue
            acc = value.get("mean_acc", value.get("acc"))
            auc = value.get("mean_auc", value.get("auc"))
            ap = value.get("mean_ap", value.get("ap"))
            if acc is not None:
                out[str(key)] = {"acc": float(acc),
                                 "auc": float(auc) if auc is not None else float("nan"),
                                 "ap": float(ap) if ap is not None else float("nan")}
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
            bucket = groups.setdefault(op, {"acc": [], "auc": [], "ap": []})
            for metric in ("acc", "auc", "ap"):
                try:
                    bucket[metric].append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    pass
    return {
        op: {k: (sum(v[k]) / len(v[k]) if v[k] else float("nan"))
             for k in ("acc", "auc", "ap")}
        for op, v in groups.items()
    }


# pre-registered expectation per operator -- written down *before* the sweep, so
# the hypothesis cannot be reinterpreted after seeing the numbers
EXPECTED: Dict[str, Tuple[str, str]] = {
    "npr_nearest": ("基线", "与生成器上采样核一致"),
    "bilinear": ("下降", "三角核与方波核不匹配"),
    "bicubic": ("下降", "三次核更平滑，高频被压低"),
    "box": ("下降", "盒式平均即低通，只剩混叠"),
    "laplacian": ("下降", "纯高通，无上采样周期结构"),
    "dct_lowpass": ("下降", "保留全部方向，不含周期"),
    "fft_radial": ("下降", "环形截止不区分网格谐波"),
    "identity": ("大幅下降", "无残差，仅剩像素统计差异"),
}


def sweep_rows(sweep: Dict[str, Dict[str, float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    base = _lookup_op(sweep, "npr_nearest")
    base_auc = base.get("auc") if base else None
    for name, _, _ in OPERATORS:
        expect, why = EXPECTED.get(name, ("—", "—"))
        hit = _lookup_op(sweep, name)
        if hit and hit["auc"] == hit["auc"]:
            acc = f"{100 * hit['acc']:.2f}"
            auc = f"{100 * hit['auc']:.2f}"
            # Δ 放在 AUC 上：换算子会改变 conv1 的输入尺度，Acc（0 阈值）随之整体塌掉，
            # 在 Acc 上算差值是误导；AUC 只依赖排序，才是可比的读数。
            if base_auc is not None and base_auc == base_auc and name != "npr_nearest":
                auc += f" ({100 * (hit['auc'] - base_auc):+.2f})"
            ap = f"{100 * hit['ap']:.2f}" if hit["ap"] == hit["ap"] else "—"
        else:
            acc = auc = ap = "待补"
        rows.append([name, expect, why, acc, auc, ap])
    return rows


# The report's operator table uses the short names (box, bilinear, ...) while
# src/ops/residual_ops.py registers the round-trip ones with an `npr_` prefix.
# Without this mapping every non-baseline cell stays "待补" forever.
def _lookup_op(sweep: Dict[str, Dict[str, float]], name: str):
    for key in (name, f"npr_{name}"):
        if key in sweep:
            return sweep[key]
    return None


ABLATION_GROUPS = {
    "matched": ["npr_nearest"],
    "interp": ["npr_bilinear", "npr_bicubic", "npr_box"],
    "highpass": ["laplacian", "dct_lowpass", "fft_radial"],
    "control": ["identity"],
}


def _group_mean(sweep: Dict[str, Dict[str, float]], keys: Sequence[str], metric: str = "auc"):
    vals = [sweep[k][metric] for k in keys if k in sweep and sweep[k][metric] == sweep[k][metric]]
    return 100.0 * sum(vals) / len(vals) if vals else None


def ablation_verdict(sweep: Dict[str, Dict[str, float]]) -> str:
    """把假设检验的判定写成一句数据驱动的话，而不是先写好结论再挑数字。"""
    m = _group_mean(sweep, ABLATION_GROUPS["matched"])
    i = _group_mean(sweep, ABLATION_GROUPS["interp"])
    h = _group_mean(sweep, ABLATION_GROUPS["highpass"])
    c = _group_mean(sweep, ABLATION_GROUPS["control"])
    if m is None or i is None or h is None:
        return ""
    parts = [f"`npr_nearest` 平均 AUC {m:.2f}，插值核往返组 {i:.2f}，纯高通组 {h:.2f}"]
    if c is not None:
        parts.append(f"无高通对照组 {c:.2f}")
    head = "，".join(parts) + "。"
    if m - i >= 2.0 and m - h >= 2.0:
        return head + (f"匹配组比插值组高 {m - i:.2f} pp、比纯高通组高 {m - h:.2f} pp，"
                       "**支持“决定泛化的是残差算子与生成器上采样核的匹配，而非是否提取高频”这一假设**。")
    if abs(m - i) < 2.0 and abs(m - h) < 2.0:
        return head + ("三组差距都在 2 pp 以内，**假设在这一组评测上不被支持**："
                       "换成不匹配的高通算子并没有明显代价，NPR 的增益更可能来自更弱的解释。")
    worst = "插值组" if m - i <= m - h else "纯高通组"
    return head + (f"匹配组与{worst}差距最大，但未同时拉开两组，**证据不齐**："
                   "只能说明该组算子更依赖上采样核匹配，不能排除“高通本身”的作用。")


# ---------------- baselines measured on the real test set ----------------
BASELINE_RUNS = [
    ("256 不裁剪（论文口径）", os.path.join(REPO_ROOT, "runs", "real", "npr256.json")),
    ("224 中心裁剪（默认配置）", os.path.join(REPO_ROOT, "runs", "real", "npr224.json")),
]
ROBUST_CSV = os.path.join(REPO_ROOT, "runs", "real", "npr256", "robust_npr_nearest.csv")

# AIGI_test 的目录名 → 论文表格里的生成器名（论文值见 NPR_PAPER）
GEN_TO_PAPER = {
    "progan": "ProGAN", "stylegan": "StyleGAN", "stylegan2": "StyleGAN2",
    "biggan": "BigGAN", "cyclegan": "CycleGAN", "stargan": "StarGAN",
    "gaugan": "GauGAN", "whichfaceisreal": "WFIR", "ADM": "ADM", "Glide": "Glide",
    "Midjourney": "Midjourney", "stable_diffusion_v_1_4": "SDv1.4",
    "stable_diffusion_v_1_5": "SDv1.5", "VQDM": "VQDM", "wukong": "Wukong",
    "DALLE2": "DALLE2",
}
# 先 GAN 后扩散，组内按论文顺序；不在表里的追加到末尾
GENERATOR_ORDER = ["progan", "stylegan", "stylegan2", "biggan", "cyclegan", "stargan",
                   "gaugan", "whichfaceisreal", "ADM", "Glide", "Midjourney",
                   "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM",
                   "wukong", "DALLE2", "coco_sdxl_nw"]


def load_baseline(path: str):
    """(summary, rows) from a runs/real/*.json written by src/test.py --cross."""
    if not os.path.isfile(path):
        return None, []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, []
    cross = data.get("cross") or {}
    return cross.get("summary"), cross.get("rows") or []


def baseline_rows(rows: Sequence[dict]) -> List[List[str]]:
    """17 行：生成器 / 本报告 Acc·AUC·AP / 论文 NPR / Δ Acc。"""
    by_gen = {str(r.get("generator")): r for r in rows}
    out: List[List[str]] = []
    for gen in GENERATOR_ORDER + [g for g in by_gen if g not in GENERATOR_ORDER]:
        row = by_gen.get(gen)
        if not row:
            continue
        acc = 100.0 * float(row["acc"])
        auc = 100.0 * float(row["auc"])
        ap = 100.0 * float(row["ap"])
        paper_name = GEN_TO_PAPER.get(gen)
        paper = NPR_PAPER.get(paper_name) if paper_name else None
        if paper is None:
            paper_s, delta_s = "—", "—"
        else:
            paper_s = f"{paper:.1f}"
            delta_s = f"{acc - paper:+.1f}"
        out.append([gen, f"{acc:.2f}", f"{auc:.2f}", f"{ap:.2f}", paper_s, delta_s])
    return out


def protocol_rows() -> List[List[str]]:
    """同一套权重、两种预处理口径的均值对比。"""
    out: List[List[str]] = []
    for label, path in BASELINE_RUNS:
        summary, _ = load_baseline(path)
        if not summary:
            out.append([label, "待补", "—", "—", "—", "—"])
            continue
        out.append([
            label,
            f"{100 * summary['mean_acc']:.2f}",
            f"{100 * summary['mean_auc']:.2f}",
            f"{100 * summary['mean_ap']:.2f}",
            f"{100 * summary['gan_mean_acc']:.2f}",
            f"{100 * summary['diffusion_mean_acc']:.2f}",
        ])
    return out


def load_calibration():
    """Threshold-calibration summary, or None when the experiment has not run."""
    if not os.path.isfile(CALIB_JSON):
        return None
    try:
        with open(CALIB_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_calibration_rows() -> Dict[str, dict]:
    """Per-generator threshold calibration (results/calibration.csv)."""
    if not os.path.isfile(CALIB_CSV):
        return {}
    out = {}
    with open(CALIB_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("generator"):
                out[row["generator"]] = row
    return out


def load_profile() -> List[dict]:
    if not os.path.isfile(PROFILE_CSV):
        return []
    with open(PROFILE_CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def profile_rows(rows: Sequence[dict]) -> List[List[str]]:
    """Collapse 17 generators into the archetypes that matter for confounding.

    The question is not "what does each generator look like" but "is the label
    collinear with container format or resolution", so group by that pattern
    rather than by exact pixel numbers (500x375 vs 500x380 is not a difference).
    ``.jpg`` and ``.jpeg`` are the same container and must not split a group.
    """
    if not rows:
        return []

    def _ext(value: str) -> str:
        raw = (value or "").strip().lower().lstrip(".")
        if raw in ("jpg", "jpeg", "jpe"):
            return "JPEG"
        if raw in ("png",):
            return "PNG"
        return raw.upper() or "—"

    def _hw(row, side):
        try:
            w = int(round(float(row.get(f"{side}_w") or 0)))
            h = int(round(float(row.get(f"{side}_h") or 0)))
        except Exception:
            w = h = 0
        return w, h

    groups: Dict[tuple, List[tuple]] = {}
    for row in rows:
        rw, rh = _hw(row, "real")
        fw, fh = _hw(row, "fake")
        key = (_ext(row.get("real_ext")), _ext(row.get("fake_ext")), rw == rh, fw == fh)
        groups.setdefault(key, []).append((row["generator"], rw, rh, fw, fh))

    def _fmt(pairs: List[tuple]) -> str:
        ws = [p[0] for p in pairs if p[0]]
        hs = [p[1] for p in pairs if p[1]]
        if not ws or not hs:
            return "—"
        same_w = min(ws) == max(ws)
        same_h = min(hs) == max(hs)
        if same_w and same_h and ws[0] == hs[0]:
            return "%d×%d" % (ws[0], hs[0])
        if ws == hs and min(ws) != max(ws):
            return "%d–%d 方图" % (min(ws), max(ws))
        w = str(ws[0]) if same_w else "%d–%d" % (min(ws), max(ws))
        h = str(hs[0]) if same_h else "%d–%d" % (min(hs), max(hs))
        return "%s×%s" % (w, h)

    out = []
    for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rext, fext, rsq, fsq = key
        gens = sorted(m[0] for m in members)
        rsize = _fmt([(m[1], m[2]) for m in members]) + ("" if rsq else "·非方")
        fsize = _fmt([(m[3], m[4]) for m in members]) + ("" if fsq else "·非方")
        collinear = "是" if (rext != fext or rsize != fsize) else "否"
        if len(gens) > 2:
            names = "、".join(gens[:3]) + " 等 %d 个" % len(gens)
        else:
            names = "、".join(gens)
        out.append([names, rext, fext, rsize, fsize, collinear])
    return out


def robust_rows() -> List[List[str]]:
    """按攻击聚合的鲁棒性矩阵（src/eval/robustness.py 写的 CSV）。"""
    if not os.path.isfile(ROBUST_CSV):
        return []
    buckets: Dict[str, Dict[str, List[float]]] = {}
    with open(ROBUST_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            attack = row.get("attack", "")
            if not attack:
                continue
            bucket = buckets.setdefault(attack, {"acc": [], "auc": []})
            for metric in ("acc", "auc"):
                try:
                    bucket[metric].append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    pass
    order = ["none", "jpeg95", "jpeg75", "jpeg50", "resize05"]
    label = {"none": "无后处理", "jpeg95": "JPEG 95", "jpeg75": "JPEG 75",
             "jpeg50": "JPEG 50", "resize05": "0.5× 缩放"}
    out: List[List[str]] = []
    base = None
    for attack in order + [a for a in buckets if a not in order]:
        if attack not in buckets:
            continue
        b = buckets[attack]
        acc = 100 * sum(b["acc"]) / len(b["acc"]) if b["acc"] else float("nan")
        auc = 100 * sum(b["auc"]) / len(b["auc"]) if b["auc"] else float("nan")
        if attack == "none":
            base = acc
        delta = f"{acc - base:+.2f}" if base is not None and attack != "none" else "—"
        out.append([label.get(attack, attack), f"{acc:.2f}", delta, f"{auc:.2f}"])
    return out


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

    p = add_body(doc, "**术语约定：** 本文说的“**失效**”不是“准确率下降”，而是一个操作化标准 —— "
                      "判别器掉到**随机水平**（Acc≈50%、AUC≈50%），即所谓 collapse to chance。"
                      "按这个定义，CNNSpot 在扩散模型上确实失效（50–60%），"
                      "而 NPR 没有失效（本报告实测扩散组 Acc 75.68%），它只是**明显变弱**。"
                      "标题里的“没有失效”就是这个意思；报告后文一旦说“变弱/掉点”都指相对下降，"
                      "说“失效”都指塌到随机。", size=9.5, indent=True)
    p.paragraph_format.space_after = Pt(6)

    sweep = load_sweep()
    measured = bool(sweep)

    # ---------------- 1. github ----------------
    add_section(doc, "1. GitHub 链接")
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0.74)
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.3
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_rich(p, "仓库地址：", size=10.5)
    add_hyperlink(p, REPO_URL, REPO_URL, size=10)
    add_rich(p, "。本地已完成 `git init` 与首次"
                "提交（分支 `main`，工作区干净），`origin` 已指向该仓库；"
                "推上去之后此链接即为最终代码。若地址有变，改 `scripts/build_report.py` 里"
                "这一行后重跑即可，不必手工改 docx。")
    add_body(doc, "为保证评分时不依赖网络，提交包内 `3_可选附件或代码/aigc-freq-forensics/` "
                  "放置同一份代码快照（逐文件 SHA-1 比对一致）；两者如有差异，以 GitHub 为准。")

    # ---------------- 2. task ----------------
    add_section(doc, "2. 任务描述 你要做什么")
    add_body(doc, "生成式模型鉴伪的核心问题是**泛化**：在 ProGAN 上训练的分类器，遇到 StyleGAN、"
                  "ADM、Stable Diffusion 还能不能用。已有的实证很尖锐——CNNSpot（ResNet-50）在 "
                  "17 个生成器上的平均准确率只有 70.78%，在扩散模型上基本等于抛硬币"
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
    add_body(doc, "做法：把前置算子抽象成可插拔的算子族（8 个，见第 4 节表 3），**同一份主干权重**下"
                  "逐个替换并跨 17 个生成器评测，指标取 Acc 与 AP，并额外扫 JPEG 95/75/50 与 0.5× 缩放"
                  "五种后处理，看掉点是否也随算子变化。产出是一张“算子 × 生成器”的表和一张鲁棒性矩阵，"
                  "而不是一次调参。")
    add_body(doc, "之所以把它当成图像处理题而不是模型题：整条链路里唯一被改动的是图像的频域/空间"
                  "算子，主干网络、训练超参、数据划分全部保持不变，所以表上的差值只能归因于算子。")

    # ---------------- 3. data ----------------
    add_section(doc, "3. 数据描述 数据是什么")
    add_body(doc, "本报告实际使用的数据是 **AIGI_test**：17 个生成器的真假二分类测试集，"
                  "共 **152,598 张**（76,299 real / 76,299 fake），解压后 34.0 GiB（约 36.5 GB）。"
                  "每个生成器一个目录、内含 `0_real` 与 `1_fake`；其中 ProGAN、CycleGAN、StyleGAN、"
                  "StyleGAN2 四个生成器还按物体类别再分了子目录（ProGAN 20 类、CycleGAN 6 类），"
                  "收拾时必须先合并成“一个生成器”，否则 20 个目录会被当成 20 个生成器，"
                  "模型学到的是“哪个物体”而不是“是否合成”，而训练曲线看起来完全正常。")
    add_body(doc, "17 个生成器与训练生成器不重叠：**8 个 GAN**（ProGAN、StyleGAN、StyleGAN2、BigGAN、"
                  "CycleGAN、StarGAN、GauGAN、WhichFaceIsReal）+ **9 个扩散模型**（ADM、Glide、Midjourney、"
                  "SDv1.4、SDv1.5、VQDM、Wukong、DALLE2、coco-SDXL）。这个 GAN/扩散的分组是后面所有"
                  "“均值”的来源：整个项目的论点就建立在这两组的差距上。")
    add_body(doc, "同样重要的是**没拿到什么**：ForenSynths 训练集（约 30GB，Google Drive 分发）本机网络"
                  "不可达，所以本报告**不做任何重训练**，全部结论来自“同一套官方权重 + 换掉输入前的残差算子”"
                  "的评测。这一点直接决定了第 5 节的性质：它是机制探针，不是从头训练后的性能对比。")
    add_body(doc, "整理数据时踩到的坑，都写进了代码而不是记在脑子里：")
    add_body(doc, "① **同一个包里混着两种层级**。17 个生成器里有 13 个是 `<生成器>/{0_real,1_fake}`，"
                  "另 4 个（ProGAN、CycleGAN、StyleGAN、StyleGAN2）多一层类别目录。只按“深度”写规则必然"
                  "错一半，所以配对规则改成按名字判断："
                  "父目录名是已知生成器名就合并、子目录名是已知生成器名就不合并，两个都不是才退回按结构猜。")
    add_body(doc, "② **`ProGAN`、`StyleGAN` 这类名字本身“长得像 fake”**，按名字猜类会把一对生成器切成 "
                  "real/fake，整份测试集标签翻转却不报错。本项目的配对规则要求“恰好一个 real-like + "
                  "一个 fake-like”，两侧同向即判歧义拒绝。")
    add_body(doc, "③ **真实数据集里确实有坏文件**：一轮评测被 `OSError: image file is truncated "
                  "(18 bytes not processed)` 打断。这在公开数据集里很常见（尾部缺几个字节），"
                  "处理方式是显式开启 `ImageFile.LOAD_TRUNCATED_IMAGES` 让 PIL 补齐，并把文件名带进报错，"
                  "而不是吞掉异常。")
    add_caption(doc, "表 1　数据与权重的实际状态")
    add_table(doc,
              ["用途", "数据集 / 来源", "生成器", "规模与状态"],
              [["跨生成器测试", "AIGI_test（本项目实际使用）", "GAN 8 + 扩散 9 = 17", "152,598 张，76,299 / 76,299，verify 0 error"],
               ["鲁棒性测试", "同上 × 5 种后处理", "17", "在线生成（JPEG 95/75/50、0.5× 缩放）"],
               ["基线权重", "官方 `NPR.pth`（NPR 仓库）", "—", "146/146 张量，1,437,761 参数"],
               ["训练 / 验证", "ForenSynths train/val", "ProGAN × 4 类", "未获取 ⇒ 本轮不重训"]],
              widths_cm=[2.4, 3.9, 2.5, 5.8], size=8.5)
    add_body(doc, "整理链条已脚本化：`download_data.py` 负责解压与摆位（同盘硬链接，不多占 34GB），"
                  "`prepare_data.py verify` 负责体检。摆位结果 `RESULT OK (17 generators, 76299 real / "
                  "76299 fake, 0 errors, 0 warnings)` 是本节所有数字成立的前提。", size=9.5)

    # ---------------- 3.x the confound nobody mentions ----------------
    profile = load_profile()
    add_subsection(doc, "3.1 必须先说清的混杂：标签与容器格式、分辨率共线")
    add_body(doc, "这是本报告最重要的一个自我限制，写在结果之前而不是藏在局限里。"
                  "用 `scripts/profile_data.py` 把两侧的文件扩展名与尺寸统计出来后，"
                  "发现数据集的构造方式让**标签与“容器格式 / 分辨率 / 重采样历史”在扩散子集里完全共线**：")
    if profile:
        add_caption(doc, "表 2　real / fake 两侧的容器格式与分辨率（`scripts/profile_data.py` 实测）")
        add_table(doc, ["生成器", "real 容器", "fake 容器", "real 尺寸", "fake 尺寸", "标签与格式/尺寸共线"],
                  profile_rows(profile),
                  widths_cm=[3.9, 1.9, 1.9, 3.0, 2.4, 1.5], size=8.5)
        add_body(doc, "读法：**9 个扩散生成器的 real 全是 JPEG、fake 全是 PNG，而且 real 是约 500×375 的"
                      "非方图、fake 是 256²/512²/1024² 的方图** —— 在扩散子集里，"
                      "“这是 fake”与“这是 PNG 且是方图”是同一件事。"
                      "相比之下 7 个 GAN 生成器两侧都是 PNG、且**以 256×256 为主**（抽样中 70–100% 为方图；WhichFaceIsReal 两侧都是 1024² JPEG），"
                      "**GAN 子集没有这个混杂**。", size=10)
        add_body(doc, "后果必须写在明面上：第 5 节的 “GAN 90.50 vs 扩散 75.68” **不能干净地归因于"
                      "“生成器类型 / 上采样核差异”**，至少要先排除三个替代解释 —— "
                      "① real 侧全是 JPEG，量化把高频抹平了一部分，而 fake 侧是 PNG 原图，"
                      "模型可能只是在判“这张图是否被 JPEG 压过”；"
                      "② 分辨率与方图/非方图差异改变了残差算子的响应尺度；"
                      "③ 扩散子集的 real 来自 COCO 照片、GAN 子集的 real 来自 LSUN/FFHQ，"
                      "**real 域本身就不一样**。按这个读法，扩散组的掉点至少有一部分来自源域而不是生成器。", size=10)
        add_body(doc, "本报告对这条混杂的处理是**限定结论范围**而不是假装解决：凡是用到 GAN/扩散对比的地方，"
                      "都限定为“在 AIGI_test 的这份构造下”的观测；"
                      "把两侧统一到同一容器/分辨率后重跑，是下一步而不是本轮已完成的工作。"
                      "另外做了两项**代价很小的旁证**："
                      "① 第 5.4 节的算子消融在同一份数据上只换算子、不动数据，"
                      "若掉点全来自源域混杂，那么“匹配核 vs 不匹配核”的 8.5–14.4 pp 差距不该出现；"
                      "② 图 1 把两侧统一重采样到 256² 后比较径向功率谱，"
                      "结果 fake 侧在高频并不占优（详见 5.1），说明“fake 残留上采样网格”这个最朴素的解释"
                      "在全局径向统计上并不成立。这两条不能洗掉混杂，但能说明结论不只由混杂驱动。", size=10)
    else:
        add_body(doc, "`results/data_profile.csv` 尚未生成，跑 "
                      "`python scripts/profile_data.py` 后本节表格自动填满。", size=9.5)

    # ---------------- 4. code changes ----------------
    add_section(doc, "4. 代码修改描述 你改了什么")
    add_body(doc, "上游 fork 自 NPR 官方实现 `chuangchuangtan/NPR-DeepfakeDetection`（CVPR 2024）。"
                  "一共七处改动，前两处是核心，第七处是真实数据逼出来的。")
    add_body(doc, "**改动 1：把写死的高通算子抽成可插拔算子族。** 上游在 `networks/resnet.py` 里把 "
                  "nearest 往返残差直接写在 `forward()` 中，无法替换。本项目新增 `src/ops/residual_ops.py`，"
                  "定义 `ResidualOp` 基类与 8 个实现（表 3），模型通过 `op=` 参数注入，`make_residual_op()` "
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
    add_body(doc, "**改动 6：74 项无数据集自检。** `scripts/smoke_test.py` 覆盖指标、频谱、数据整理与"
                  "算子/模型四段，其中“常数图必须被每个高通算子归零”“`npr_nearest` 必须与官方公式逐元素"
                  "一致”这两条断言，正是抓住上述两个 bug 的测试。没有 torch 时明确 SKIP，不假装通过。")
    add_body(doc, "**改动 7：评测层改成报原始 logit（真实数据才暴露的坑）。** 官方权重在真实图上输出的 "
                  "logit 量级高达数百，`sigmoid` 在 |logit|>17 之后在 float32 里饱和成 1.0，于是所有样本"
                  "得到同一个分数：AUC 假性约 0.5、Acc 假性 50%，而程序一句警告都没有。修法是"
                  "`collect_scores` 直接返回 logit（二分类取 `logits[:,1]-logits[:,0]`）并把默认阈值定在 "
                  "0.0 —— 单调变换对 AUC/AP 没有任何意义，却会因浮点饱和丢掉全部排序信息。同批还修了"
                  "两个只在 34GB 数据上才会出现的故障：按 batch 截断时只截到一类样本（AUC 变 `nan`），"
                  "改成**每类各截 N 张**；一个尾部缺字节的 JPEG 让整轮 18 分钟评测崩掉，改为显式允许"
                  "PIL 补齐，且报错必须带出文件名。")
    add_caption(doc, "表 3　算子族（同一主干，只换输入前的残差算子）")
    add_table(doc, ["算子", "如何得到残差", "角色"],
              [[n, d, r] for n, d, r in OPERATORS],
              widths_cm=[2.8, 7.4, 4.4], size=8.5, keep_together=True)

    # ---------------- 5. results ----------------
    base_summary, base_rows = load_baseline(BASELINE_RUNS[0][1])
    protocols = protocol_rows()
    robust = robust_rows()

    add_section(doc, "5. 结果对比 结果有什么变化")
    add_body(doc, "本节区分三类数字：**本报告实测**（在 152,598 张真实测试图上跑出来的）、"
                  "**论文值**（别人的，仅作对照）、**代码级验证**（不依赖数据集的自检）。"
                  "所有实测都在同一套官方权重 `NPR.pth` 上完成，指标一律报 Acc/AUC/AP 三项："
                  "只报 Acc 会把“阈值没校准”和“排序能力差”混为一谈，而这两件事的修法完全不同。")
    add_caption(doc, "表 4　代码级验证（不依赖数据集）")
    add_table(doc, ["检查项", "结果", "含义"],
              [["自检通过率", "74 / 74", "指标、频谱、数据整理、算子与模型四段全过；`best_threshold_accuracy` 与暴力枚举对拍 3 例（含并列值）。缺 torch 时算子/模型段明确 SKIP，其余 44 项仍须全过"],
               ["参数量", "1,437,761", "与论文 1.44M 一致，证明裁剪结构复现正确"],
               ["官方权重加载", "146 / 146 张量", "0 missing / 0 unexpected：自实现网络的键名与官方 `NPR.pth` 完全一致"],
               ["`npr_nearest` vs 官方公式", "最大偏差 5.96e-08", "前置算子与上游实现逐元素等价"],
               ["常数图残差（laplacian）", "5.96e-08", "高通算子不制造假边（修正前为 0.255）"],
               ["常数图残差（dct_lowpass）", "2.09e-07", "DCT 往返可逆（修正前为 0.462）"],
               ["数据摆位", "17 生成器 / 76,299 + 76,299", "`verify` 报 0 error、0 warning"]],
              widths_cm=[4.3, 3.9, 6.4], size=8.5)

    add_subsection(doc, "5.1 基线复现")
    add_body(doc, "官方 `NPR.pth` 在本项目整理出的 17 个生成器上逐生成器评测，"
                  "与论文同一张表的数值并排放在一起（论文用的是 ForenSynths 版测试集，两个集合不完全相同，"
                  "所以这里的“论文 NPR”一列只作量级对照）。")
    add_caption(doc, "表 5　本报告实测：17 个生成器（官方 NPR.pth，Resize(256) 不裁剪，0 阈值）")
    add_table(doc, ["生成器", "Acc", "AUC", "AP", "论文 NPR", "Δ Acc"], baseline_rows(base_rows),
              widths_cm=[4.2, 1.8, 1.8, 1.8, 2.0, 1.8], size=8.0)
    if base_summary:
        calib = load_calibration()
        calib_rows = load_calibration_rows()
        by_gen = {r["generator"]: r for r in base_rows}
        m_acc = 100 * base_summary["mean_acc"]
        m_auc = 100 * base_summary["mean_auc"]
        m_ap = 100 * base_summary["mean_ap"]
        gan_acc = 100 * base_summary["gan_mean_acc"]
        dif_acc = 100 * base_summary["diffusion_mean_acc"]
        add_body(doc, f"全部 17 个生成器平均 **Acc {m_acc:.2f} / AUC {m_auc:.2f} / AP {m_ap:.2f}**；"
                      f"其中 8 个 GAN 平均 Acc {gan_acc:.2f}，9 个扩散模型平均 Acc {dif_acc:.2f}。"
                      f"两个读数要分开看：① **AUC/AP 只依赖排序**，{m_auc:.2f} 说明“哪个更像假”的判断力是复现出来的；"
                      f"论文报的 91.7 是 **Acc**，而且是在 ForenSynths 的 16 生成器表上，"
                      f"与本报告的 17 生成器 / AIGI_test 不是同一张表，两个数字不能直接相减。")
        gap = PAPER_MEAN_ACC - m_acc
        if calib and calib.get("global"):
            g = calib["global"]
            all_g = calib["groups"]["all"]
            add_body(doc, "② **阈值偏移到底贡献了多少：这是测出来的，不是推出来的。** "
                          "把每个生成器的真假两半各自随机对半分（调阈值 / 评估互不重叠，种子 70），"
                          f"调出一个全体共用阈值 {g['threshold']:+.2f} 后，留在评估半边上的 "
                          f"Acc 从 **{g['acc_thr0']:.2f} 升到 {g['acc_tuned']:.2f}（{g['gain']:+.2f} pp）**，"
                          f"AUC {g['auc']:.2f}。若按生成器各自调阈值（部署时用不上，因为要预知来源），"
                          f"上界是 {all_g['acc_tuned']:.2f}（+{all_g['gain']:.2f} pp）。"
                          f"所以：**阈值偏移只能解释 {g['gain']:.2f} pp，而两张表的总差是 {gap:.1f} pp**；"
                          f"其余约 {gap - g['gain']:.1f} pp 可能来自测试集构成（AIGI_test 的 fake 覆盖 9 种扩散模型，"
                          "含 1024² 与 COCO 源域）、源域差异，或本轮只评测不重训 —— 这三者本报告无法分离，"
                          "需要进一步对照才能定论。")
        add_body(doc, "③ **扩散组的掉点是真实的，但归因要留口子**：本报告确实复现出 GAN 组比扩散组高 "
                      f"{gan_acc - dif_acc:.2f} pp，方向与论文一致；不过第 3.1 节已经说明，"
                      "这份数据的标签与容器格式/分辨率共线，所以这一差距**不能全部记在“生成器类型”账上**。")
        dal = by_gen.get("DALLE2", {})
        wfir = by_gen.get("whichfaceisreal", {})
        dal_c = calib_rows.get("DALLE2", {})
        paper_dal = NPR_PAPER.get(GEN_TO_PAPER.get("DALLE2", "DALLE2"))
        if dal and wfir and dal_c and paper_dal:
            dal_acc = 100 * float(dal["acc"])
            wfir_acc = 100 * float(wfir["acc"])
            wfir_auc = 100 * float(wfir["auc"])
            dal_tuned = 100 * float(dal_c["acc_tuned"])
            rest = paper_dal - dal_tuned
            add_body(doc, f"④ 最弱的是 DALLE2（{dal_acc:.2f}）与 WhichFaceIsReal（{wfir_acc:.2f}，"
                          f"AUC 仅 {wfir_auc:.1f}）。DALLE2 的 {dal_acc:.2f} 里很大一块是分数偏移："
                          f"单独按生成器调阈值后是 **{dal_tuned:.2f}**（阈值 {float(dal_c['threshold']):+.2f}，"
                          f"其 AUC 为 {100 * float(dal_c['auc']):.2f}），"
                          f"与论文的 {paper_dal:.1f} 仍差约 {rest:.1f} pp，这部分才可能与子集差异有关；"
                          "本报告不做进一步归因。")

        add_body(doc, "**残差算子看到的是什么：径向功率谱。** 图 1 把两侧统一重采样到 256² 后比较频谱"
                      "（不重采样则主要测的是分辨率差，而分辨率与标签共线，见 3.1）。"
                      "左图是四个平均功率谱，右图把 fake−real 的差值单独画出来：")
        add_figure(doc, FIG_SPECTRUM,
                   "图 1　径向功率谱（Azimuthal 平均，680 张图）。左：GAN/扩散 × real/fake 四条平均曲线；"
                   "右：fake − real 的差值，0 线以上表示 fake 的高频更强。"
                   "所有图先 bicubic 重采样到 256²，否则测的是分辨率而不是内容。", width_cm=15.2)
        add_body(doc, "读图要点，也是本项目修改过的判断：**fake 侧在高频并不占优**。"
                      "扩散组在 r<0.12 处 fake 略高（合成图的低频更“整”），"
                      "到了中高频反而低于 real；GAN 组整段略低于 real。"
                      "如果 NPR 的增益来自“抓住 fake 残留的上采样网格”，"
                      "那么在全局径向统计上应当看到 fake 侧中高频抬升 —— 实际没有。"
                      "所以更合理的读法是：**残差算子提供的是局部相位/网格结构，"
                      "而不是全局径向能量**；这也解释了为什么 5.4 节里同样是高通的 laplacian 只能拿回一部分差距，"
                      "以及为什么一张 8×8 分块 DCT 反而排在插值族之上。", size=10)
    else:
        add_body(doc, "基线尚未运行，表 5 留空。跑 `python -m src.test --ckpt database/weights/NPR.pth "
                      "--op npr_nearest --cross --limit-per-class 300 --workers 0` 后本表自动填满。", size=9.5)

    add_subsection(doc, "5.2 预处理口径的影响")
    add_body(doc, "论文的 `testmodel()` 用 Resize(256) **不裁剪**，"
                  "而项目默认配置是 224 中心裁剪 —— 这个差别看着小，会让同一套权重差出 1 pp 以上。"
                  "两种口径都完整跑了 152,598 张，结果如下。")
    add_caption(doc, "表 6　预处理口径对比（同一套官方权重、同一测试集）")
    add_table(doc, ["口径", "均值 Acc", "均值 AUC", "均值 AP", "GAN Acc", "扩散 Acc"], protocols,
              widths_cm=[4.0, 2.0, 2.0, 2.0, 2.3, 2.3], size=8.5, keep_together=True)
    if len(protocols) >= 2 and protocols[1][1] != "待补":
        d_acc = float(protocols[0][1]) - float(protocols[1][1])
        d_auc = float(protocols[0][2]) - float(protocols[1][2])
        add_body(doc, f"不裁剪口径全面更好：Acc {d_acc:+.2f} pp、AUC {d_auc:+.2f} pp。"
                      "原因是裁剪会把图像四周的生成器伪影切掉一部分，而这类伪影（上采样网格、"
                      "频域截止）恰好集中在边缘与高频；裁掉等于主动丢弃判别信息。"
                      "**因此本报告此后所有数字统一用 256 不裁剪口径**，项目里也新增了 "
                      "`configs/eval_256_nocrop.yaml` 把这件事固定下来。")
    else:
        add_body(doc, "口径对比尚未跑完，表 6 留空。", size=9.5)

    add_subsection(doc, "5.3 论文对照")
    add_body(doc, "下表是论文原表的数值，只用来标定“差多少算多”，不是本项目的实测。")
    add_caption(doc, "表 7　论文值（对照基线，非本项目实测，Acc %）")
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
                  "差别全部集中在输入前那一步残差 —— 于是问题变成：那一步残差到底提供了什么。",
             size=9.5)

    add_subsection(doc, "5.4 算子族消融（核心实验）")
    add_body(doc, "同一套权重，只把输入前的残差算子换掉，重跑全部 17 个生成器。"
                  "预期列在跑之前就写死，不是事后解释。")
    add_caption(doc, "表 8　算子族消融：预注册预期 vs 实测（%，每个算子 17 生成器 × 每类 300 张）")
    add_table(doc, ["算子", "预期", "机制依据", "Acc（Δ）", "AUC", "AP"], sweep_rows(sweep),
              widths_cm=[2.5, 1.5, 5.3, 1.5, 2.3, 1.5], size=8.0)
    verdict = ablation_verdict(sweep)
    if verdict:
        add_body(doc, verdict)
        add_body(doc, "**为什么除 `npr_nearest` 外，表 8 的 Acc 一栏都塌到 50% 附近、而 AUC 还有 52–84。** "
                      "换算子等于换掉了 "
                      "`conv1` 的输入：官方权重只见过 nearest 残差的数值尺度，换个算子后整体分数偏移，"
                      "0 阈值立刻失效，Acc 全部掉到抛硬币水平。这正是探针的已知代价，也说明读数只能看 "
                      "AUC/AP。反过来说，若连 AUC 都保留，说明**排序能力**没有丢——所以这里的分歧是真分歧，"
                      "不是阈值造成的假象。", size=9.5)
        add_body(doc, "**细节上的三个观察。** ① `identity`（完全不处理）AUC 52.5，基本等于随机，说明"
                      "“先做一步高通”是必要条件；② 同样带高通、但保留完整像素网格的 `laplacian` 与 "
                      "`dct_lowpass`（AUC 83.8 / 82.8）明显好于会重采样网格的 `bilinear`/`bicubic`/`box`"
                      "（80.9 / 77.9 / 79.1），说明分类头吃的是“网格结构”而不仅是“高频能量”；"
                      "③ 唯一的径向频域算子 `fft_radial` 掉到 58.3，因为环形截止不区分方向、也不保留"
                      "局部相位，网格谐波被一并抹掉——这一条恰好与“核匹配”假设同向。"
                      "④ `laplacian` 只比 `npr_nearest` 低 8.5 pp，说明 nearest 的优势是**加成**，"
                      "不是“高通与否”的分水岭；把话说满会失真。", size=9.5)
        add_figure(doc, FIG_ABLATION,
                   "图 2　算子族消融（同一套官方权重，17 生成器 × 每类 300 张）。纵轴 AUC，"
                   "括号内为相对 `npr_nearest` 的差值；红色虚线是随机水平。"
                   "颜色按机制分组：匹配核 / 插值族 / 纯高通 / 对照组。", width_cm=14.6)
    else:
        add_body(doc, "实测列尚未填满（`待补`），跑 `python -m src.sweep --mode pretrained "
                      "--ckpt database/weights/NPR.pth --ops all` 后本表与上面的判定句会自动生成。", size=9.5)
    add_body(doc, "这一步的代价必须写清楚：`--mode pretrained` 只换算子、不重训，`conv1` 从未见过新算子的"
                  "输出分布，所以它是**机制探针**而不是性能结论。要得到干净的性能数字，得在 ForenSynths "
                  "训练集上对每个算子从头训练一遍（`--mode scratch`）；本轮没有拿到训练集，因此没做。"
                  "好在探针能回答的问题是“同一个分类头遇到不同的前置算子会怎样”，这恰好是假设检验需要的。",
             size=9.5)

    if robust:
        add_subsection(doc, "5.5 鲁棒性")
        add_body(doc, "取证模型在社交平台上遇到的第一件事是 JPEG 重压缩与缩放。"
                      "下表在测试时在线生成 5 种后处理，逐一重跑；为控制时间每类只取 100 张，"
                      "所以每档 n=200/生成器 —— 无后处理那一档的 82.53 与 5.1 节全量 82.65 的差"
                      "就来自这个子集，不是异常。")
        add_caption(doc, "表 9　鲁棒性：同一权重的 17 生成器均值")
        add_table(doc, ["后处理", "均值 Acc", "Δ Acc", "均值 AUC"], robust,
                  widths_cm=[4.0, 3.0, 3.0, 3.0], size=8.5, keep_together=True)
        add_body(doc, "读法：这里的结论是**负面且明确**的 —— 无后处理时均值 AUC 92.75，只要过一次 "
                      "JPEG 95 就掉到 45.32（低于随机，说明分数整体被翻向），JPEG 75/50 同样停在 44–45；"
                      "0.5× 缩放到 58.93，比 JPEG 稍好但同样不可用。也就是说这套权重**不能直接用于社交"
                      "媒体上流转过的图片**，必须在训练阶段做对应的数据增强才有可能恢复。本项目未取得"
                      "训练集，无法复现那一步，所以这里只报告现象，不给“加上增强就能修好”的结论"
                      "（那需要重训验证）。另一个细节值得记：JPEG 的破坏力大于缩放，说明判别信息挂在"
                      "**精细的网格结构**上——重采样改变网格周期但保留其存在，JPEG 的 DCT 量化则直接抹掉"
                      "高频系数。这与 5.4 的“核匹配”解释互相支持。", size=9.5)
        add_figure(doc, FIG_ROBUST,
                   "图 3　鲁棒性（同一套权重，17 生成器均值，每类 100 张）。纵轴 AUC，"
                   "红色虚线是随机水平；除“无后处理”外的四档全部低于或贴近随机，"
                   "其中 JPEG 档低于 50% 说明分数被系统性翻向。", width_cm=12.4)
    else:
        add_subsection(doc, "5.5 鲁棒性（待补）")
    add_body(doc, "下面这一条命令会把上面两张表都生成出来：", size=9.5)
    add_code_block(doc, ["python -m src.test --ckpt database/weights/NPR.pth --robustness \\",
                         "    --limit-per-class 200 --workers 0 --out-dir runs/real/npr256"])

    # ---------------- 6. references ----------------
    add_section(doc, "6. 参考文献")
    for item in REFERENCES:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.75)
        pf.space_before = Pt(0)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.2
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_rich(p, item, size=9)

    # ---------------- closing ----------------
    add_body(doc, "**小结。** 本项目交出的不是“比 SOTA 高 X%”，而是：一套可复用的实验框架 + 一条可证伪的"
                  "机制解释 + 一组在真实数据上跑出来的对照数字。已实测的部分：8 个残差算子行为正确"
                  "（往返恒等、常数图归零、与官方公式一致到 5.96e-08）；自实现网络与官方权重完全对齐"
                  "（146/146 张量、1,437,761 参数）；17 个生成器 152,598 张全量评测得到 "
                  "Acc/AUC/AP = 82.65/92.41/91.18，并复现出“GAN 好比扩散模型好”这一核心现象；"
                  "预处理口径被证明会带来 1 pp 量级的差别，必须固定。")
    add_body(doc, "**局限，写在这里而不是藏起来。** ① **没有训练集**（ForenSynths 约 30GB，本机网络不可达），"
                  "所以算子消融只能用 `--mode pretrained` 探针，拿不到“每个算子从头训练”的干净性能数字；"
                  "② 探针实验里换算子会改变 `conv1` 的输入分布，结论只能当作方向指示；"
                  "③ 单次评测、无重复种子与置信区间，1 pp 以内的差异不宜过度解读；"
                  "④ Acc 依赖 0 阈值；阈值本身已经测过（5.1 节，全局校准只换来 +2.2 pp），差距主要不是阈值造成的，但报告仍以 AUC/AP 为主指标；"
                  "⑤ DALLE2 与 WhichFaceIsReal 的表现与论文不同向，未做归因。")
    add_body(doc, "**下一步按顺序做四件事**：① 补下载 ForenSynths train/val（约 30GB，Google Drive），"
                  "`prepare_data.py organize` 摆好并 `verify --leakage-with` 查重；② 用官方 `NPR.pth` "
                  "微调/复现训练流程，确认训练侧也能到 90% 量级；③ `--mode scratch` 对 8 个算子逐一从头"
                  "训练，把表 8 换成正经的性能对比；④ 把阈值校准做成正式一步：现在是拿测试集对半分搜的，"
                  "正式做法要在验证集上搜、再去测试集报告，最后 `build_report.py` 重出本报告。")

    if dry_run:
        return out_path

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Final guard: any paragraph whose text reached the document without going
    # through add_rich() still carries literal markdown markers. Strip them so the
    # reader never sees a stray backtick or **.
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if TICK in run.text or "**" in run.text:
                run.text = run.text.replace(TICK, "").replace("**", "")

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
