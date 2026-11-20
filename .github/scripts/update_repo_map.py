import ast
import json
import re
from pathlib import Path

ROOT = Path(".")
TARGET = ROOT / ".github" / "copilot-instructions.md"
IGNORE_DIRS = {".git", ".github", "__pycache__", ".venv", "venv", "node_modules"}

def scan_py(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    items = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(f"`def {node.name}()`")
        elif isinstance(node, ast.ClassDef):
            items.append(f"`class {node.name}`")
    return items

def scan_ipynb(path: Path) -> list[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    titles = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        first_line = "".join(cell.get("source", [])[:1]).strip()
        if first_line.startswith("# @title"):
            titles.append(f"Cell {i}: `{first_line.replace('# @title', '').strip()}`")
    return titles

lines = []
for p in sorted(ROOT.rglob("*")):
    if any(part in IGNORE_DIRS for part in p.parts) or not p.is_file():
        continue
    if p.suffix == ".py":
        symbols = scan_py(p)
        summary = f" — {', '.join(symbols[:8])}" if symbols else ""
        lines.append(f"- `{p.as_posix()}`{summary}")
    elif p.suffix == ".ipynb":
        cells = scan_ipynb(p)
        lines.append(f"- `{p.as_posix()}`")
        for c in cells:
            lines.append(f"  - {c}")

block = "<!-- AUTO-MAP:START -->\n" + "\n".join(lines) + "\n<!-- AUTO-MAP:END -->"
content = TARGET.read_text(encoding="utf-8")
updated = re.sub(
    r"<!-- AUTO-MAP:START -->.*?<!-- AUTO-MAP:END -->",
    block,
    content,
    flags=re.DOTALL,
)

if updated != content:
    TARGET.write_text(updated, encoding="utf-8")
