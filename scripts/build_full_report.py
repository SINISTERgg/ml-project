# -*- coding: utf-8 -*-
"""
PlaceIQ Academic Minor Project Report Master Compiler.
Generates:
  1. reports/project_report.md
  2. reports/project_report.html
"""

import os
import re
import html
import base64
from report_chapters_1_3 import get_chapters_1_to_3_md
from report_chapters_4_6 import get_chapters_4_to_6_md
from report_chapters_7_10 import get_chapters_7_to_10_md
from generate_reports import get_front_matter_md

WORKSPACE = r"c:\Users\M S\Desktop\ml project"
REPORTS_DIR = os.path.join(WORKSPACE, "reports")
PLOTS_DIR = os.path.join(WORKSPACE, "models", "plots")
MD_PATH = os.path.join(REPORTS_DIR, "project_report.md")
HTML_PATH = os.path.join(REPORTS_DIR, "project_report.html")

def get_image_base64_map():
    img_map = {}
    # Scan models/plots
    for root, dirs, files in os.walk(PLOTS_DIR):
        for f in files:
            if f.endswith(".png"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, REPORTS_DIR).replace("\\", "/")
                with open(full_path, "rb") as img_file:
                    b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                    data_uri = f"data:image/png;base64,{b64_data}"
                    img_map[rel_path] = data_uri
                    img_map[f] = data_uri
                    img_map[f"../models/plots/{f}"] = data_uri
                    if "eda" in full_path:
                        img_map[f"../models/plots/eda/{f}"] = data_uri
    return img_map

def build_markdown():
    print("Assembling Markdown components...")
    front = get_front_matter_md()
    c1_3 = get_chapters_1_to_3_md()
    c4_6 = get_chapters_4_to_6_md()
    c7_10 = get_chapters_7_to_10_md()

    full_md = front + "\n" + c1_3 + "\n" + c4_6 + "\n" + c7_10
    
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(full_md)
    
    size_kb = len(full_md.encode("utf-8")) / 1024
    lines = len(full_md.splitlines())
    words = len(full_md.split())
    print(f"Markdown built successfully: {MD_PATH}")
    print(f"Stats: {lines} lines, {words} words, {size_kb:.1f} KB")
    return full_md

def convert_md_to_html(md_content, img_map):
    print("Converting Markdown to publication-grade HTML...")
    
    # CSS Stylesheet designed specifically for Academic Minor Project Reports
    css = """
    :root {
      --primary-color: #0f172a;
      --secondary-color: #1e3a8a;
      --accent-color: #2563eb;
      --text-color: #1f2937;
      --bg-color: #f8fafc;
      --card-bg: #ffffff;
      --border-color: #e2e8f0;
      --table-header-bg: #f1f5f9;
      --table-stripe: #f8fafc;
      --font-serif: "Cambria", "Times New Roman", Times, "Liberation Serif", serif;
      --font-sans: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: "Consolas", "Courier New", Courier, monospace;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: var(--font-serif);
      font-size: 11.5pt;
      line-height: 1.65;
      color: var(--text-color);
      background-color: var(--bg-color);
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    .report-container {
      max-width: 960px;
      margin: 30px auto;
      background: var(--card-bg);
      padding: 60px 75px;
      box-shadow: 0 4px 25px rgba(0, 0, 0, 0.08);
      border-radius: 4px;
    }

    /* Academic Cover Page */
    .cover-page {
      text-align: center;
      padding: 40px 20px 20px 20px;
      border: 3px double #1e293b;
      margin-bottom: 50px;
      background-color: #ffffff;
    }

    .cover-inst-heading {
      font-family: var(--font-sans);
      font-size: 14pt;
      font-weight: 700;
      letter-spacing: 1.5px;
      color: #334155;
      text-transform: uppercase;
      margin-bottom: 25px;
    }

    .cover-title {
      font-family: var(--font-sans);
      font-size: 20pt;
      font-weight: 800;
      color: var(--secondary-color);
      line-height: 1.3;
      margin: 25px 0 15px 0;
      text-transform: uppercase;
      border-top: 2px solid #cbd5e1;
      border-bottom: 2px solid #cbd5e1;
      padding: 20px 10px;
    }

    .cover-subtext {
      font-size: 11pt;
      font-style: italic;
      color: #475569;
      margin: 20px 0;
    }

    .cover-degree {
      font-family: var(--font-sans);
      font-size: 12pt;
      font-weight: 700;
      color: #0f172a;
      text-transform: uppercase;
      margin-bottom: 35px;
    }

    .cover-meta-grid {
      display: flex;
      justify-content: space-between;
      text-align: left;
      margin: 40px 20px 30px 20px;
      font-family: var(--font-sans);
      font-size: 10.5pt;
      line-height: 1.6;
    }

    .cover-meta-block {
      flex: 1;
    }

    .cover-meta-block strong {
      color: #0f172a;
    }

    .cover-dept-footer {
      font-family: var(--font-sans);
      font-size: 11pt;
      font-weight: 600;
      color: #1e293b;
      margin-top: 35px;
      border-top: 1px solid #e2e8f0;
      padding-top: 20px;
    }

    /* Headings */
    h1 {
      font-family: var(--font-sans);
      font-size: 18pt;
      font-weight: 700;
      color: var(--secondary-color);
      border-bottom: 2px solid #cbd5e1;
      padding-bottom: 8px;
      margin-top: 45px;
      margin-bottom: 18px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    h2 {
      font-family: var(--font-sans);
      font-size: 14pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 35px;
      margin-bottom: 14px;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 5px;
    }

    h3 {
      font-family: var(--font-sans);
      font-size: 12pt;
      font-weight: 600;
      color: #334155;
      margin-top: 25px;
      margin-bottom: 10px;
    }

    h4 {
      font-family: var(--font-sans);
      font-size: 11pt;
      font-weight: 600;
      color: #475569;
      margin-top: 18px;
      margin-bottom: 8px;
    }

    p {
      text-align: justify;
      margin-top: 0;
      margin-bottom: 14px;
    }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 22px 0;
      font-family: var(--font-sans);
      font-size: 9.5pt;
      border-top: 2.5px solid #1e293b;
      border-bottom: 2.5px solid #1e293b;
      page-break-inside: avoid;
    }

    th {
      background-color: var(--table-header-bg);
      color: #0f172a;
      font-weight: 600;
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1.5px solid #334155;
    }

    td {
      padding: 8px 12px;
      border-bottom: 1px solid #e2e8f0;
      vertical-align: top;
    }

    tr:nth-child(even) td {
      background-color: var(--table-stripe);
    }

    /* Figures and Captions */
    .report-figure {
      text-align: center;
      margin: 30px auto;
      page-break-inside: avoid;
    }

    .report-figure img {
      max-width: 88%;
      height: auto;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
      padding: 4px;
      background: #ffffff;
    }

    .report-figure figcaption {
      font-family: var(--font-sans);
      font-size: 9.5pt;
      color: #334155;
      margin-top: 10px;
      font-weight: 600;
    }

    /* Code & Formulas */
    pre {
      background-color: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      padding: 14px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 9pt;
      line-height: 1.45;
      page-break-inside: avoid;
      margin: 18px 0;
    }

    code {
      font-family: var(--font-mono);
      font-size: 9.5pt;
      background-color: #f1f5f9;
      padding: 2px 5px;
      border-radius: 3px;
      border: 1px solid #e2e8f0;
      color: #0f172a;
    }

    pre code {
      background-color: transparent;
      padding: 0;
      border: none;
    }

    .math-block {
      background-color: #f8fafc;
      border-left: 3px solid var(--accent-color);
      padding: 12px 18px;
      margin: 16px 0;
      font-family: var(--font-serif);
      font-size: 11pt;
      overflow-x: auto;
    }

    /* Lists */
    ul, ol {
      margin-top: 0;
      margin-bottom: 14px;
      padding-left: 28px;
    }

    li {
      margin-bottom: 6px;
      text-align: justify;
    }

    /* Page Breaks */
    .page-break {
      page-break-before: always;
      break-before: page;
      margin-top: 40px;
      border-top: 1px dashed #cbd5e1;
      padding-top: 20px;
    }

    /* Citations */
    .citation {
      font-weight: 600;
      color: var(--secondary-color);
    }

    .reference-item {
      padding-left: 28px;
      text-indent: -28px;
      margin-bottom: 12px;
      font-size: 10.5pt;
      text-align: justify;
    }

    /* Print Styles */
    @media print {
      body {
        background: #ffffff;
        font-size: 10.5pt;
      }
      .report-container {
        max-width: 100%;
        margin: 0;
        padding: 0;
        box-shadow: none;
      }
      .page-break {
        border-top: none;
        padding-top: 0;
        page-break-before: always;
        break-before: page;
      }
      table, figure, pre, .no-break {
        page-break-inside: avoid;
      }
      h1, h2 {
        page-break-after: avoid;
      }
      @page {
        size: A4;
        margin: 20mm 18mm 22mm 18mm;
      }
    }
    """

    # Parse markdown into HTML elements
    lines = md_content.splitlines()
    html_lines = []
    
    in_code_block = False
    code_lang = ""
    code_content = []
    
    in_table = False
    table_rows = []
    
    in_math_block = False
    math_content = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return ""
        
        out = ["<table>"]
        # Filter separator row
        clean_rows = []
        for r in table_rows:
            if re.match(r"^\s*\|?\s*:?-+:?\s*\|", r):
                continue
            clean_rows.append(r)
        
        if clean_rows:
            # Header
            header_cells = [c.strip() for c in clean_rows[0].strip().strip("|").split("|")]
            out.append("  <thead>\n    <tr>")
            for c in header_cells:
                out.append(f"      <th>{format_inline(c)}</th>")
            out.append("    </tr>\n  </thead>")
            
            # Body
            if len(clean_rows) > 1:
                out.append("  <tbody>")
                for r in clean_rows[1:]:
                    cells = [c.strip() for c in r.strip().strip("|").split("|")]
                    out.append("    <tr>")
                    for c in cells:
                        out.append(f"      <td>{format_inline(c)}</td>")
                    out.append("    </tr>")
                out.append("  </tbody>")
        out.append("</table>")
        in_table = False
        table_rows = []
        return "\n".join(out)

    def format_inline(text):
        # Format inline bold, italic, code, links
        t = text
        # Images: ![alt](src)
        def replace_img(match):
            alt, src = match.group(1), match.group(2)
            base_src = src.replace("\\", "/")
            actual_src = img_map.get(base_src, img_map.get(os.path.basename(base_src), src))
            return f'<figure class="report-figure"><img src="{actual_src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'
        
        t = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, t)
        
        # Replace image src in raw html figure tags if any
        def replace_html_img(match):
            src = match.group(1)
            base_src = src.replace("\\", "/")
            actual_src = img_map.get(base_src, img_map.get(os.path.basename(base_src), src))
            return f'src="{actual_src}"'
        
        t = re.sub(r'src="([^"]+)"', replace_html_img, t)
        
        # Bold
        t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
        # Italic
        t = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', t)
        # Inline code
        t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
        # Math inline
        t = re.sub(r'\$([^$]+)\$', r'<span class="math-inline"><em>\1</em></span>', t)
        return t

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        
        # Check code block
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                html_lines.append(f'<pre><code class="language-{code_lang}">{html.escape(chr(10).join(code_content))}</code></pre>')
                code_content = []
                code_lang = ""
            else:
                if in_table:
                    html_lines.append(flush_table())
                in_code_block = True
                code_lang = line.replace("```", "").strip()
            i += 1
            continue
        
        if in_code_block:
            code_content.append(line)
            i += 1
            continue
        
        # Check math block $$...$$
        if line.strip().startswith("$$"):
            if in_math_block:
                in_math_block = False
                html_lines.append(f'<div class="math-block">$$ {html.escape(chr(10).join(math_content))} $$</div>')
                math_content = []
            else:
                if in_table:
                    html_lines.append(flush_table())
                # Check if single line $$ ... $$
                stripped = line.strip()
                if stripped.endswith("$$") and len(stripped) > 4:
                    eq = stripped[2:-2].strip()
                    html_lines.append(f'<div class="math-block">$$ {html.escape(eq)} $$</div>')
                else:
                    in_math_block = True
            i += 1
            continue
        
        if in_math_block:
            math_content.append(line)
            i += 1
            continue
        
        # Check table line
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_rows.append(line)
            i += 1
            continue
        else:
            if in_table:
                html_lines.append(flush_table())
        
        # Empty line
        if not line.strip():
            i += 1
            continue
        
        # Page break div
        if "<div class=\"page-break\"></div>" in line:
            html_lines.append('<div class="page-break"></div>')
            i += 1
            continue
        
        # Horizontal rule
        if line.strip() == "---":
            html_lines.append('<hr style="border: 0; border-top: 1px solid #cbd5e1; margin: 30px 0;">')
            i += 1
            continue
            
        # Headings
        if line.startswith("# "):
            title_text = line[2:].strip().replace("**", "")
            html_lines.append(f'<h1>{format_inline(title_text)}</h1>')
            i += 1
            continue
        elif line.startswith("## "):
            h2_text = line[3:].strip().replace("**", "")
            html_lines.append(f'<h2>{format_inline(h2_text)}</h2>')
            i += 1
            continue
        elif line.startswith("### "):
            h3_text = line[4:].strip().replace("**", "")
            html_lines.append(f'<h3>{format_inline(h3_text)}</h3>')
            i += 1
            continue
        elif line.startswith("#### "):
            h4_text = line[5:].strip().replace("**", "")
            html_lines.append(f'<h4>{format_inline(h4_text)}</h4>')
            i += 1
            continue
        
        # Check raw figure / img block
        if "<figure" in line or "<div" in line or "</div" in line or "<figcaption" in line or "</figure" in line:
            html_lines.append(format_inline(line))
            i += 1
            continue
            
        # Reference items
        ref_match = re.match(r'^\s*(\[\d+\])\s*(.*)', line)
        if ref_match:
            ref_num = ref_match.group(1)
            ref_body = ref_match.group(2)
            html_lines.append(f'<div class="reference-item"><strong>{ref_num}</strong> {format_inline(ref_body)}</div>')
            i += 1
            continue

        # Lists (unordered)
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            item = line.strip()[2:]
            html_lines.append(f'<ul><li>{format_inline(item)}</li></ul>')
            i += 1
            continue

        # Lists (ordered)
        ol_match = re.match(r'^\s*(\d+)\.\s*(.*)', line)
        if ol_match:
            item = ol_match.group(2)
            html_lines.append(f'<ol start="{ol_match.group(1)}"><li>{format_inline(item)}</li></ol>')
            i += 1
            continue

        # Standard paragraph
        html_lines.append(f'<p>{format_inline(line)}</p>')
        i += 1

    if in_table:
        html_lines.append(flush_table())

    body_html = "\n".join(html_lines)
    
    # Consolidate adjacent ul and ol tags
    body_html = re.sub(r'</ul>\s*<ul>', '', body_html)
    body_html = re.sub(r'</ol>\s*<ol[^>]*>', '', body_html)

    # Wrap in complete HTML5 structure
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PlaceIQ: Academic Minor Project Report</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="report-container">
{body_html}
  </div>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    size_html_kb = len(full_html.encode("utf-8")) / 1024
    print(f"HTML report built successfully: {HTML_PATH}")
    print(f"Stats: {size_html_kb:.1f} KB")

def main():
    md = build_markdown()
    img_map = get_image_base64_map()
    print(f"Mapped {len(img_map)} image references into base64 data URIs.")
    convert_md_to_html(md, img_map)
    print("\n[SUCCESS] Both academic reports generated flawlessly!")

if __name__ == "__main__":
    main()
