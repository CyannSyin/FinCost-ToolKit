from __future__ import annotations

import os
from pathlib import Path

import markdown


_ASSET_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".pdf")


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
    .chart-block {{
      margin: 12px 0 24px;
    }}
    .chart-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-start;
      margin: 12px 0 24px;
    }}
    .chart-item {{
      flex: 1 1 calc(50% - 8px);
      max-width: calc(50% - 8px);
      overflow: hidden;
    }}
    .chart-grid img,
    .chart-block img {{
      width: 100%;
      height: auto;
      max-height: none;
      max-width: none;
    }}
    .chart-grid object,
    .chart-grid iframe {{
      width: 100%;
      height: 430px;
    }}
    .chart-block object,
    .chart-block iframe {{
      width: 100%;
      height: 520px;
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""


def _urlize(path: str) -> str:
    return path.replace(os.sep, "/").replace(" ", "%20")


def _resolve_asset(markdown_path: Path, html_dir: Path, asset_name: str) -> tuple[str, str]:
    base = Path(asset_name)
    md_dir = markdown_path.parent
    asset_path = (md_dir / base).resolve()
    if base.suffix and asset_path.exists():
        rel = os.path.relpath(asset_path, html_dir)
        return _urlize(rel), asset_path.suffix.lower()

    if not base.suffix:
        for ext in _ASSET_EXTS:
            candidate = (md_dir / f"{asset_name}{ext}").resolve()
            if candidate.exists():
                rel = os.path.relpath(candidate, html_dir)
                return _urlize(rel), candidate.suffix.lower()

    matches: list[Path] = []
    for ext in _ASSET_EXTS:
        matches.extend(sorted(md_dir.rglob(f"*{asset_name}*{ext}")))
    if matches:
        chosen = matches[0].resolve()
        rel = os.path.relpath(chosen, html_dir)
        return _urlize(rel), chosen.suffix.lower()

    rel = os.path.relpath(asset_path, html_dir)
    return _urlize(rel), asset_path.suffix.lower()


def _chart_html(markdown_path: Path, html_dir: Path, asset_name: str, alt: str) -> str:
    rel, suffix = _resolve_asset(markdown_path, html_dir, asset_name)
    if suffix == ".pdf":
        return (
            "<div class=\"chart-block\">"
            f"<object data=\"{rel}\" type=\"application/pdf\">"
            f"<iframe src=\"{rel}\"></iframe>"
            "</object>"
            f"<div><a href=\"{rel}\">{alt}</a></div>"
            "</div>"
        )
    return "<div class=\"chart-block\">" f"<img src=\"{rel}\" alt=\"{alt}\" />" "</div>"


def _chart_item_html(markdown_path: Path, html_dir: Path, asset_name: str, alt: str) -> str:
    rel, suffix = _resolve_asset(markdown_path, html_dir, asset_name)
    if suffix == ".pdf":
        pdf_src = f"{rel}#toolbar=0&navpanes=0&scrollbar=0"
        return (
            "<div class=\"chart-item\">"
            f"<object data=\"{pdf_src}\" type=\"application/pdf\">"
            f"<iframe src=\"{pdf_src}\"></iframe>"
            "</object>"
            "</div>"
        )
    return f"<div class=\"chart-item\"><img src=\"{rel}\" alt=\"{alt}\" /></div>"


def _chart_grid_html(markdown_path: Path, html_dir: Path, assets: list[tuple[str, str]]) -> str:
    items = []
    for asset_name, alt in assets:
        items.append(_chart_item_html(markdown_path, html_dir, asset_name, alt))
    return "<div class=\"chart-grid\">" + "".join(items) + "</div>"


def _insert_after_heading(markdown_text: str, heading: str, block_html: str) -> str:
    lines = markdown_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == heading:
            insert_at = i + 2 if i + 1 < len(lines) else i + 1
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, block_html)
            lines.insert(insert_at + 2, "")
            break
    return "\n".join(lines) + "\n"


def render_diagnosis_to_html(markdown_path: str, output_path: str | None = None) -> str:
    input_path = Path(markdown_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")

    if output_path:
        html_path = Path(output_path).expanduser().resolve()
    else:
        html_path = input_path.with_suffix(input_path.suffix + ".html")

    markdown_text = _read_text(input_path)
    markdown_text = _insert_after_heading(
        markdown_text,
        "## Trading Period:",
        _chart_html(
            input_path,
            html_path.parent,
            "line_chart",
            "daily_line_chart",
        ),
    )
    markdown_text = _insert_after_heading(
        markdown_text,
        "## Total Cost:",
        _chart_grid_html(
            input_path,
            html_path.parent,
            [
                ("pie_chart_no_monthly", "daily_pie_chart_no_monthly"),
                ("pie_chart", "daily_pie_chart"),
            ],
        ),
    )

    body_html = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "toc"],
        output_format="html5",
    )
    html = _build_html(input_path.name, body_html)
    _write_text(html_path, html)
    print(f"Saved diagnosis HTML to: {html_path}")
    return str(html_path)
