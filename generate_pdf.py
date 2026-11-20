"""
Generate IEEE-style two-column conference paper PDF.
Matches the formatting of churn_conference_paper.pdf:
- US Letter (612x792 pt)
- Two-column layout
- Times Roman fonts
- Title: 19pt centered
- Author: 9pt centered, italic department
- Abstract: 9.3pt bold
- Section headers: 10pt Roman numerals (I. II. III.)
- Body: 10pt Times Roman
- Page numbers centered
"""
import re
import textwrap
from pathlib import Path
from fpdf import FPDF

SOURCE = Path(r"C:\Users\SAHMO\.gemini\antigravity\brain\684100ab-8c47-43ac-9e64-59cb0ddbb94b\Research_Paper.md")
OUTPUT = Path(r"C:\Users\SAHMO\Downloads\projecto\results\Research_Paper_Mohit_Kumar_2026PAD2529.pdf")

ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

# Page geometry (US Letter)
PAGE_W = 215.9  # mm
PAGE_H = 279.4  # mm
MARGIN_TOP = 19
MARGIN_BOTTOM = 25
MARGIN_LEFT = 17
MARGIN_RIGHT = 17
COL_GAP = 6
USABLE_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
COL_W = (USABLE_W - COL_GAP) / 2


def clean(text):
    """Strip markdown formatting, convert special chars to latin-1 safe."""
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$\\\\[a-zA-Z]+\{([^}]*)\}\$', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = text.replace('\\geq', '>=')
    text = text.replace('\\leq', '<=')
    text = text.replace('\\times', 'x')
    text = text.replace('\\cdot', '.')
    text = text.replace('\\text{', '').replace('}', '')
    text = text.replace('\\hat{y}', 'y_hat')
    text = text.replace('\\arg\\max', 'argmax')
    text = text.replace('\\sigma', 'sigma')
    text = text.replace('\\beta', 'beta')
    text = text.replace('\\delta', 'delta')
    text = text.replace('\\epsilon', 'epsilon')
    text = text.replace('\\sum', 'sum')
    text = text.replace('\\frac', '')
    text = text.replace('\\mathbb{1}', '1')
    text = text.replace('\\mathcal{L}', 'L')
    text = text.replace('\\top', 'T')
    text = text.replace('\\mid', '|')
    text = text.replace('\\quad', '  ')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('**', '')
    text = text.replace('*', '')
    # Unicode to latin-1
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2265', '>=').replace('\u2264', '<=')
    text = text.replace('\u00d7', 'x')
    text = text.replace('\u2192', '->')
    return text.encode('latin-1', 'replace').decode('latin-1')


class IEEEPaper(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'Letter')
        self.set_margins(MARGIN_LEFT, MARGIN_TOP, MARGIN_RIGHT)
        self.set_auto_page_break(False)
        self.col = 0  # 0 = left, 1 = right
        self.section_num = 0

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', '', 8)
        self.set_text_color(0, 0, 0)
        self.cell(0, 5, str(self.page_no()), align='C')

    def col_x(self):
        if self.col == 0:
            return MARGIN_LEFT
        return MARGIN_LEFT + COL_W + COL_GAP

    def check_page_break(self, h):
        """Check if we need to switch column or page."""
        if self.get_y() + h > PAGE_H - MARGIN_BOTTOM:
            if self.col == 0:
                self.col = 1
                self.set_y(MARGIN_TOP)
                self.set_x(self.col_x())
                return True
            else:
                self.add_page()
                self.col = 0
                self.set_y(MARGIN_TOP)
                self.set_x(self.col_x())
                return True
        return False

    def col_cell(self, h, text, font_family='Times', font_style='', font_size=10, align='L'):
        """Write a single-line cell in the current column."""
        self.check_page_break(h)
        self.set_font(font_family, font_style, font_size)
        self.set_x(self.col_x())
        self.cell(COL_W, h, text, align=align)
        self.ln(h)

    def col_multi_cell(self, h, text, font_family='Times', font_style='', font_size=10, align='J'):
        """Write a multi-line cell in the current column."""
        self.set_font(font_family, font_style, font_size)
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                lines.append('')
                continue
            # Approximate character wrapping for column width
            chars_per_line = int(COL_W / (font_size * 0.22))
            wrapped = textwrap.wrap(paragraph, width=chars_per_line)
            lines.extend(wrapped)

        for line in lines:
            self.check_page_break(h)
            self.set_x(self.col_x())
            self.set_font(font_family, font_style, font_size)
            if line == '':
                self.ln(h * 0.5)
            else:
                self.cell(COL_W, h, line, align='L')
                self.ln(h)

    def write_title(self, title):
        """Centered title spanning full width, 19pt Times."""
        self.set_font('Times', '', 19)
        self.set_text_color(0, 0, 0)
        self.set_xy(MARGIN_LEFT, MARGIN_TOP)
        self.multi_cell(USABLE_W, 9, title, align='C')
        self.ln(3)

    def write_authors(self, name, dept, loc, email):
        """Author block centered, 9pt."""
        self.set_font('Times', '', 9)
        self.cell(USABLE_W, 4, name, align='C')
        self.ln(4)
        self.set_font('Times', 'I', 9)
        self.cell(USABLE_W, 4, dept, align='C')
        self.ln(4)
        self.cell(USABLE_W, 4, loc, align='C')
        self.ln(4)
        self.set_font('Times', '', 9)
        self.cell(USABLE_W, 4, email, align='C')
        self.ln(6)

    def write_abstract(self, text):
        """Bold abstract block, 9.3pt, full width then switch to two-col."""
        self.set_font('Times', 'B', 9)
        self.set_x(MARGIN_LEFT)
        label = 'Abstract - '
        body = clean(text)
        full = label + body
        chars_per_line = int(USABLE_W / (9 * 0.22))
        wrapped = textwrap.wrap(full, width=chars_per_line)
        for line in wrapped:
            self.set_x(MARGIN_LEFT)
            self.cell(USABLE_W, 4, line)
            self.ln(4)
        self.ln(2)

    def write_index_terms(self, terms):
        """Index Terms line, 9pt bold."""
        self.set_font('Times', 'B', 9)
        self.set_x(MARGIN_LEFT)
        full = 'Index Terms - ' + clean(terms)
        chars_per_line = int(USABLE_W / (9 * 0.22))
        wrapped = textwrap.wrap(full, width=chars_per_line)
        for line in wrapped:
            self.set_x(MARGIN_LEFT)
            self.cell(USABLE_W, 4, line)
            self.ln(4)
        self.ln(4)

    def write_section_header(self, title):
        """Roman numeral section header, 10pt centered in column."""
        self.section_num += 1
        self.check_page_break(10)
        self.ln(2)
        roman = ROMAN[self.section_num] if self.section_num < len(ROMAN) else str(self.section_num)
        header_text = f'{roman}. {clean(title).upper()}'
        self.set_x(self.col_x())
        self.set_font('Times', '', 10)
        self.cell(COL_W, 5, header_text, align='C')
        self.ln(6)

    def write_subsection_header(self, title):
        """Italic subsection header."""
        self.check_page_break(8)
        self.ln(1)
        self.set_x(self.col_x())
        self.set_font('Times', 'I', 10)
        self.cell(COL_W, 4, clean(title))
        self.ln(5)

    def write_body(self, text):
        """Body text, 10pt Times, justified in column."""
        self.col_multi_cell(4, clean(text), 'Times', '', 10)
        self.ln(1)

    def write_table(self, headers, rows, caption=''):
        """Simple table in column."""
        n = len(headers)
        cell_w = COL_W / n
        est_h = (len(rows) + 2) * 4 + 8
        self.check_page_break(est_h)

        if caption:
            self.set_x(self.col_x())
            self.set_font('Times', 'B', 8)
            self.cell(COL_W, 4, clean(caption))
            self.ln(4)

        # Header
        self.set_font('Times', 'B', 7)
        self.set_x(self.col_x())
        for h in headers:
            self.cell(cell_w, 4, clean(h).strip(), border='TB', align='C')
        self.ln(4)

        # Rows
        self.set_font('Times', '', 7)
        for row in rows:
            self.set_x(self.col_x())
            for j, cell in enumerate(row):
                ct = clean(cell).strip()
                a = 'L' if j == 0 else 'C'
                self.cell(cell_w, 3.5, ct, align=a)
            self.ln(3.5)

        # Bottom line
        self.set_x(self.col_x())
        self.cell(COL_W, 0, '', border='T')
        self.ln(3)

    def write_reference(self, text):
        """Reference entry, 8pt."""
        self.check_page_break(8)
        self.set_x(self.col_x())
        self.set_font('Times', '', 8)
        chars_per_line = int(COL_W / (8 * 0.22))
        wrapped = textwrap.wrap(clean(text), width=chars_per_line)
        for line in wrapped:
            self.set_x(self.col_x())
            self.cell(COL_W, 3.5, line)
            self.ln(3.5)
        self.ln(1)


def parse_table(lines, idx):
    headers = [c.strip() for c in lines[idx].strip('|').split('|')]
    rows = []
    i = idx + 2
    while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
        row = [c.strip() for c in lines[i].strip('|').split('|')]
        rows.append(row)
        i += 1
    return headers, rows, i


def build():
    md = SOURCE.read_text(encoding='utf-8')
    lines = md.split('\n')

    pdf = IEEEPaper()
    pdf.alias_nb_pages()
    pdf.add_page()

    # === TITLE ===
    title = lines[0].lstrip('# ').strip()
    pdf.write_title(clean(title))

    # === AUTHOR ===
    pdf.write_authors(
        'Mohit Kumar',
        'Department of Computer Science & Engineering',
        'Netaji Subhas University of Technology (NSUT), New Delhi, India',
        'mohit.kumar.pg26@nsut.ac.in'
    )

    # === Find and write ABSTRACT ===
    abstract_start = None
    abstract_end = None
    keywords_line = None
    for idx, line in enumerate(lines):
        if line.strip() == '## Abstract':
            abstract_start = idx + 2  # skip blank line after header
        if abstract_start and line.strip().startswith('**Keywords'):
            keywords_line = idx
            abstract_end = idx
            break
        if abstract_start and line.strip() == '---' and idx > abstract_start + 2:
            abstract_end = idx
            break

    if abstract_start and abstract_end:
        abstract_text = ' '.join(l.strip() for l in lines[abstract_start:abstract_end] if l.strip())
        pdf.write_abstract(abstract_text)

    if keywords_line:
        kw = lines[keywords_line].replace('**Keywords:**', '').replace('**', '').strip()
        pdf.write_index_terms(kw)

    # === TWO COLUMN MODE ===
    pdf.col = 0
    start_y = pdf.get_y()

    # Parse remaining content
    i = abstract_end + 1 if abstract_end else 17
    # Skip to first ## section after abstract
    while i < len(lines) and not (lines[i].strip().startswith('## ') and 'Abstract' not in lines[i]):
        i += 1

    table_counter = 0
    in_references = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == '---':
            i += 1
            continue

        # Appendix headers
        if stripped.startswith('## Appendix'):
            pdf.write_section_header(stripped[3:])
            i += 1
            continue

        # Main section (## )
        if stripped.startswith('## ') and not stripped.startswith('### '):
            section_name = stripped[3:]
            if 'Reference' in section_name:
                in_references = True
            pdf.write_section_header(section_name)
            i += 1
            continue

        # Subsection (### )
        if stripped.startswith('### '):
            pdf.write_subsection_header(stripped[4:])
            i += 1
            continue

        # Table
        if '|' in stripped and stripped.startswith('|') and i + 1 < len(lines) and '---' in lines[i + 1]:
            # Look for caption above
            caption = ''
            if i > 0 and ('Table' in lines[i-1] or 'TABLE' in lines[i-1]):
                caption = lines[i-1].strip()
            headers, rows, end_idx = parse_table(lines, i)
            pdf.write_table(headers, rows, caption)
            i = end_idx
            continue

        # Block quote / note
        if stripped.startswith('> '):
            quote_text = stripped[2:]
            # Strip alert markers
            quote_text = re.sub(r'\[!(NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]', '', quote_text).strip()
            if quote_text:
                pdf.col_multi_cell(3.5, quote_text, 'Times', 'I', 9)
            i += 1
            continue

        # Numbered list
        if re.match(r'^\d+\.', stripped):
            text = re.sub(r'^\d+\.\s*', '', stripped)
            pdf.col_multi_cell(4, '  ' + clean(text), 'Times', '', 9)
            i += 1
            continue

        # Bullet list
        if stripped.startswith('- '):
            text = stripped[2:]
            pdf.col_multi_cell(4, '  - ' + clean(text), 'Times', '', 9)
            i += 1
            continue

        # Reference entries [N]
        if in_references and re.match(r'^\[\d+\]', stripped):
            pdf.write_reference(stripped)
            i += 1
            continue

        # Regular paragraph
        para = []
        while i < len(lines):
            cl = lines[i].strip()
            if not cl or cl.startswith('#') or (cl.startswith('|') and i+1 < len(lines) and '---' in lines[i+1]) \
                    or cl == '---' or cl.startswith('> ') or re.match(r'^\d+\.', cl) or cl.startswith('- ') \
                    or (in_references and re.match(r'^\[\d+\]', cl)):
                break
            para.append(cl)
            i += 1
        if para:
            text = ' '.join(para)
            # Check if it's a table caption line
            if text.strip().startswith('Table') or text.strip().startswith('TABLE'):
                # Will be picked up as caption before table
                pass
            else:
                pdf.write_body(text)

    pdf.output(str(OUTPUT))
    print(f'IEEE-format PDF saved to: {OUTPUT}')
    print(f'Pages: {pdf.page_no()}')

if __name__ == '__main__':
    build()
