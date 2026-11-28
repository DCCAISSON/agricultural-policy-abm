"""Export《中国工程科学》manuscript Markdown + 300-dpi PNG figures to styled Word (.docx)."""

from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

WT_ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = WT_ROOT / "docs/Mesa农业经营接续V2重构"
MD_PATH = DOC_DIR / "农业经营接续下种粮补贴传导与动态调控研究.md"
DOCX_PATH = DOC_DIR / "农业经营接续下种粮补贴传导与动态调控研究.docx"


def _set_run_font(run, *, name_cn="SimSun", name_en="Times New Roman", size_pt=10.5, bold=False, italic=False, color=None):
    run.font.name = name_en
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), name_cn)
    rFonts.set(qn("w:ascii"), name_en)
    rFonts.set(qn("w:hAnsi"), name_en)


def _add_formatted_runs(p, text: str, *, base_size=10.5, base_bold=False, color=None):
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2])
            _set_run_font(r, size_pt=base_size, bold=True, color=color)
        elif tok.startswith("*") and tok.endswith("*"):
            r = p.add_run(tok[1:-1])
            _set_run_font(r, size_pt=base_size, bold=base_bold, italic=True, color=color)
        else:
            r = p.add_run(tok)
            _set_run_font(r, size_pt=base_size, bold=base_bold, color=color)


def _set_cell_border(cell, *, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is not None:
        tcPr.remove(tcBorders)
    borders_xml = [f"<w:tcBorders {nsdecls('w')}>"]
    if top:
        borders_xml.append(f'<w:top w:val="single" w:sz="{top}" w:space="0" w:color="000000"/>')
    else:
        borders_xml.append('<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>')
    borders_xml.append('<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>')
    if bottom:
        borders_xml.append(f'<w:bottom w:val="single" w:sz="{bottom}" w:space="0" w:color="000000"/>')
    else:
        borders_xml.append('<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>')
    borders_xml.append('<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>')
    borders_xml.append("</w:tcBorders>")
    tcPr.append(parse_xml("".join(borders_xml)))


def _add_three_line_table(doc: Document, lines: list[str]) -> None:
    rows_raw = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in lines
    ]
    # Remove markdown separator row (---)
    rows = [r for i, r in enumerate(rows_raw) if not (i == 1 and all(set(c) <= set("-: ") for c in r))]
    if not rows:
        return
    n_rows, n_cols = len(rows), len(rows[0])
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    col_widths = None
    if n_cols >= 6:
        first_cm = 3.4 if n_cols >= 8 else 3.8
        other_cm = (16.0 - first_cm) / (n_cols - 1)
        col_widths = [Cm(first_cm)] + [Cm(other_cm)] * (n_cols - 1)
    cell_font_sz = 8.0 if n_cols >= 8 else 8.5
    # Clear default grid borders at table level
    tblPr = tbl._tbl.tblPr
    existing_b = tblPr.find(qn("w:tblBorders"))
    if existing_b is not None:
        tblPr.remove(existing_b)
    tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/>'
        '<w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
    ))

    for r_i, row_vals in enumerate(rows):
        row = tbl.rows[r_i]
        for c_i in range(n_cols):
            val = row_vals[c_i] if c_i < len(row_vals) else ""
            cell = row.cells[c_i]
            if col_widths is not None:
                cell.width = col_widths[c_i]
            top_sz = 12 if r_i == 0 else None          # 1.5 pt top rule
            bot_sz = 6 if r_i == 0 else (12 if r_i == n_rows - 1 else None)  # 0.75 pt header, 1.5 pt bottom
            _set_cell_border(cell, top=top_sz, bottom=bot_sz)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            if r_i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_i == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            _add_formatted_runs(p, val.replace("`", ""), base_size=cell_font_sz, base_bold=(r_i == 0))


def convert_markdown_to_docx(md_path: Path = MD_PATH, docx_path: Path = DOCX_PATH) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.4)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.strip() == "---":
            i += 1
            continue

        # Markdown Table
        if ln.lstrip().startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            _add_three_line_table(doc, tbl_lines)
            continue

        # Markdown Image ![alt](rel_path)
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", ln.strip())
        if m_img:
            alt_txt, rel_p = m_img.group(1), m_img.group(2)
            img_file = (md_path.parent / rel_p).resolve()
            if img_file.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(6)
                p_img.paragraph_format.space_after = Pt(2)
                p_img.add_run().add_picture(str(img_file), width=Cm(15.2))
                if alt_txt:
                    p_cap = doc.add_paragraph()
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.paragraph_format.space_after = Pt(4)
                    _add_formatted_runs(p_cap, alt_txt, base_size=9.0, base_bold=True, color=RGBColor(0x14, 0x23, 0x3C))
            i += 1
            continue

        # Headings
        if ln.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(10)
            _add_formatted_runs(p, ln[2:].strip(), base_size=16.5, base_bold=True, color=RGBColor(0x14, 0x23, 0x3C))
            i += 1
            continue
        if ln.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            _add_formatted_runs(p, ln[3:].strip(), base_size=13.0, base_bold=True, color=RGBColor(0x1F, 0x4E, 0x8C))
            i += 1
            continue
        if ln.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            _add_formatted_runs(p, ln[4:].strip(), base_size=11.5, base_bold=True, color=RGBColor(0x1E, 0x29, 0x3B))
            i += 1
            continue
        if ln.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            _add_formatted_runs(p, ln[5:].strip(), base_size=10.5, base_bold=True, color=RGBColor(0x1E, 0x29, 0x3B))
            i += 1
            continue

        # Table/Figure footnotes starting with *注： or 注：
        clean_ln = ln.replace("`", "")
        if clean_ln.startswith("*注：") or clean_ln.startswith("注："):
            txt_fn = clean_ln.strip("*")
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.2
            _add_formatted_runs(p, txt_fn, base_size=8.5, color=RGBColor(0x47, 0x55, 0x69))
            i += 1
            continue

        # Bibliography entries [1] ... with hanging indent (supports optional second bilingual English line)
        if re.match(r"^\[\d+\]\s+", clean_ln):
            has_en_subline = (
                i + 1 < len(lines)
                and bool(re.match(r"^\s{2,}[A-Za-z]", lines[i + 1].replace("`", "")))
            )
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            p.paragraph_format.line_spacing = 1.22
            p.paragraph_format.space_after = Pt(1 if has_en_subline else 3)
            _add_formatted_runs(p, clean_ln, base_size=9.0)
            i += 1
            if has_en_subline:
                en_ln = lines[i].replace("`", "").strip()
                p_en = doc.add_paragraph()
                p_en.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p_en.paragraph_format.left_indent = Cm(0.75)
                p_en.paragraph_format.first_line_indent = Cm(0.0)
                p_en.paragraph_format.line_spacing = 1.18
                p_en.paragraph_format.space_after = Pt(3.5)
                _add_formatted_runs(p_en, en_ln, base_size=8.5, color=RGBColor(0x33, 0x41, 0x55))
                i += 1
            continue

        # Body paragraph (merge consecutive plain lines separated by single newline unless bullet/numbered)
        p = doc.add_paragraph()
        is_caption_line = clean_ln.startswith("**表") or clean_ln.startswith("**图")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if is_caption_line else WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(0 if (is_caption_line or clean_ln.startswith("**摘") or clean_ln.startswith("**关")) else 21)
        p.paragraph_format.line_spacing = 1.45
        p.paragraph_format.space_after = Pt(4)
        _add_formatted_runs(p, clean_ln, base_size=10.5)
        i += 1

    doc.save(str(docx_path))
    print(f"Exported CAE Word deliverable -> {docx_path} ({docx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    convert_markdown_to_docx()
