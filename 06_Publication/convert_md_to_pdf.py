import os
import re
import subprocess
import sys
import markdown

def convert_md_to_pdf(md_path, pdf_path=None):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return False
    
    if pdf_path is None:
        pdf_path = os.path.splitext(md_path)[0] + ".pdf"
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Protect math blocks and inline math before Markdown parsing
    math_store = {}
    math_counter = 0

    # 1. Protect block math $$ ... $$
    def save_block_math(match):
        nonlocal math_counter
        placeholder = f"<!--MATH_BLOCK_{math_counter}-->"
        math_store[placeholder] = f'<div class="math-block">\n$${match.group(1)}$$\n</div>'
        math_counter += 1
        return placeholder

    md_text = re.sub(r'\$\$(.*?)\$\$', save_block_math, md_text, flags=re.DOTALL)

    # 2. Protect inline math $ ... $
    def save_inline_math(match):
        nonlocal math_counter
        placeholder = f"<!--MATH_INLINE_{math_counter}-->"
        math_store[placeholder] = f"${match.group(1)}$"
        math_counter += 1
        return placeholder

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', save_inline_math, md_text)

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'toc', 'attr_list', 'def_list']
    )

    # Restore math placeholders
    for placeholder, math_code in math_store.items():
        html_content = html_content.replace(placeholder, math_code)
        html_content = html_content.replace(f"<p>{placeholder}</p>", math_code)

    base_dir = os.path.dirname(os.path.abspath(md_path))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<base href="file:///{base_dir.replace('\\', '/')}/">
<title>Document</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 800px;
    margin: 0 auto;
    padding: 0;
  }}
  h1 {{
    font-size: 18pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 12px;
    line-height: 1.3;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
  }}
  h2 {{
    font-size: 13pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 22px;
    margin-bottom: 10px;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 11.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
  }}
  hr {{
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 18px 0;
  }}
  strong {{
    color: #0f172a;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  code {{
    font-family: "Cascadia Code", Consolas, Monaco, "Courier New", monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    color: #0f172a;
  }}
  pre code {{
    display: block;
    padding: 10px;
    overflow-x: auto;
    background-color: #0f172a;
    color: #f8fafc;
    border-radius: 6px;
    line-height: 1.4;
  }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    page-break-inside: avoid;
  }}
  em {{
    color: #475569;
  }}
  .math-block {{
    margin: 14px 0;
    text-align: center;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  blockquote {{
    margin: 14px 0;
    padding: 8px 14px;
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    color: #1e3a8a;
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    temp_html_path = os.path.splitext(md_path)[0] + "_temp.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Generated HTML: {temp_html_path}")

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        temp_html_path
    ]

    print(f"Running command...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)

    if os.path.exists(pdf_path):
        print(f"SUCCESS: Generated PDF at {pdf_path}")
        return True
    else:
        print(f"FAILED to generate PDF. Error: {res.stderr}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_md = sys.argv[1]
    else:
        target_md = r"d:\Mohamed\Desktop\Concisness Framework\Consciousness and Prime Base Intelligence\Final_Output\04_Visualizations\Actualizer_Engine_Paper.md"
    
    target_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    convert_md_to_pdf(target_md, target_pdf)
