"""
Convert API_DOCS.md -> API_DOCS.pdf using Python-Markdown + Chrome headless.

Usage:  python scripts/md_to_pdf.py API_DOCS.md API_DOCS.pdf
"""
import os
import shutil
import subprocess
import sys
import tempfile

import markdown


CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


CSS = """
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1f2328;
    max-width: 100%;
  }
  h1 { font-size: 22pt; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; margin-top: 0; }
  h2 { font-size: 16pt; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 22pt; }
  h3 { font-size: 13pt; margin-top: 16pt; }
  h4 { font-size: 11.5pt; margin-top: 12pt; }
  p, li { font-size: 11pt; }
  code {
    background: #f4f4f5;
    padding: 1.5px 5px;
    border-radius: 4px;
    font-family: "JetBrains Mono", "Fira Code", Consolas, "Courier New", monospace;
    font-size: 9.8pt;
  }
  pre {
    background: #f6f8fa;
    padding: 10px 12px;
    border-radius: 6px;
    overflow-x: auto;
    border: 1px solid #d0d7de;
    font-size: 9pt;
    line-height: 1.4;
    page-break-inside: avoid;
  }
  pre code { background: transparent; padding: 0; font-size: 9pt; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9.8pt;
  }
  th, td {
    border: 1px solid #d0d7de;
    padding: 6px 10px;
    text-align: left;
    vertical-align: top;
  }
  th { background: #f6f8fa; font-weight: 600; }
  blockquote {
    border-left: 4px solid #d0d7de;
    padding: 0 12px;
    color: #57606a;
    margin: 10px 0;
  }
  hr { border: none; border-top: 1px solid #d0d7de; margin: 20px 0; }
  a { color: #0969da; text-decoration: none; }
</style>
"""


def find_browser():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    raise RuntimeError("No Chrome/Edge found in standard install locations")


def md_to_pdf(md_path: str, pdf_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )

    title = os.path.splitext(os.path.basename(md_path))[0]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>{CSS}</head>
<body>{html_body}</body></html>"""

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(html)
        html_path = tf.name

    browser = find_browser()
    out_abs = os.path.abspath(pdf_path)
    html_url = "file:///" + html_path.replace("\\", "/")

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_abs}",
        html_url,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(html_path)
    print(f"[ok] wrote {out_abs}  ({os.path.getsize(out_abs)/1024:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python md_to_pdf.py <input.md> <output.pdf>", file=sys.stderr)
        sys.exit(2)
    md_to_pdf(sys.argv[1], sys.argv[2])
