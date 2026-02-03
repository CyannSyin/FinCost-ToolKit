from pathlib import Path

import markdown


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def _build_html(title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light dark;
    }}
    body {{
      max-width: 900px;
      margin: 32px auto;
      padding: 0 20px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
    }}
    h1, h2, h3 {{
      line-height: 1.25;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: left;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    pre {{
      padding: 12px;
      overflow-x: auto;
      background: rgba(0, 0, 0, 0.04);
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""


def render_markdown_to_html(markdown_path: str, output_path: str | None = None) -> str:
    input_path = Path(markdown_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")

    if output_path:
        html_path = Path(output_path).expanduser().resolve()
    else:
        html_path = input_path.with_suffix(input_path.suffix + ".html")

    markdown_text = _read_text(input_path)
    body_html = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "toc"],
        output_format="html5",
    )
    html = _build_html(input_path.name, body_html)
    _write_text(html_path, html)
    return str(html_path)
