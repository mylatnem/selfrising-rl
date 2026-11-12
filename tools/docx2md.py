#!/usr/bin/env python
"""把 docs/学习手册.docx 转成排版良好的 Markdown。

用法：
    python tools/docx2md.py

转换规则：
    标题 1/2/3      -> # / ## / ###
    题注（表 N …）   -> 表格上方的加粗标题行
    List Bullet     -> - 列表
    "1　xxx" 编号段  -> 1. 有序列表
    Consolas 段落    -> 连续的多段合并成一个 ``` 代码块，并猜语言
    表格            -> GitHub 风格管道表
    目录（TOC）      -> 丢弃，按实际标题重新生成（纯文本，不生成可能失效的锚点）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

CODE_FONT = "consolas"


def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def style_name(p: Paragraph) -> str:
    try:
        return p.style.name or ""
    except Exception:  # noqa: BLE001
        return ""


def heading_level(name: str) -> int:
    n = name.lower()
    if "heading 1" in n or name == "标题 1":
        return 1
    if "heading 2" in n or name == "标题 2":
        return 2
    if "heading 3" in n or name == "标题 3":
        return 3
    return 0


def is_toc(name: str, text: str) -> bool:
    if name.lower().startswith("toc"):
        return True
    return bool(re.search(r"\.{3,}\s*\d+\s*$", text))


def is_code_para(p: Paragraph) -> bool:
    if not p.text.strip():
        return False
    fonted = None
    for r in p.runs:
        fn = (r.font.name or "").lower()
        if fn:
            fonted = fn
            break
    if fonted != CODE_FONT:
        return False
    # 中文成句的段落即使字体是 Consolas，也是正文（编辑时继承了字体）
    t = p.text.strip()
    if re.match(r"^\s*(\[|#|//|;|\$|\{|\}|--|<!--)", t):
        return True  # 以代码符号开头的，肯定还是代码
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    ratio = cjk / max(1, len(t))
    if t.endswith("。") or (ratio > 0.35 and re.search(r"[。，、；：]", t)):
        return False
    return True


def max_font_size(p: Paragraph) -> float:
    sizes = [r.font.size.pt for r in p.runs if r.font.size]
    return max(sizes) if sizes else 0.0


def guess_lang(lines: list[str]) -> str:
    joined = "\n".join(lines)
    if re.search(r"^\s*(#include|void loop|static const|float target)", joined, re.M):
        return "cpp"
    if re.search(r"^\s*(\[env:|platform =|board =|framework =|lib_deps|; )", joined, re.M):
        return "ini"
    if re.search(r"^\s*<[a-z]", joined, re.M):
        return "xml"
    if re.search(r"^\s*(import |from \w+ import|def |while |for .* in |print\(|np\.)", joined, re.M):
        return "python"
    if re.search(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=", joined, re.M):
        return "python"
    if re.search(r"^\s*(sudo |pip |python |pio |tensorboard|git |cd |source )", joined, re.M):
        return "bash"
    return "text"


def cell_text(cell) -> str:
    return " ".join(cell.text.split()).replace("|", "\\|")


def md_table(tbl: Table) -> list[str]:
    rows = [[cell_text(c) for c in row.cells] for row in tbl.rows]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join([" --- "] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def convert(src: Path, dst: Path) -> None:
    doc = Document(str(src))

    cover: list[str] = []
    blocks: list[tuple[str, object]] = []

    for item in iter_block_items(doc):
        if isinstance(item, Table):
            blocks.append(("table", item))
            continue

        p: Paragraph = item
        text = p.text.strip()
        name = style_name(p)

        if not text:
            continue
        if text == "目录":
            continue
        if is_toc(name, text):
            continue

        lvl = heading_level(name)
        if lvl:
            blocks.append((f"h{lvl}", text))
            continue
        if name.startswith("题注") or name.startswith("Caption"):
            blocks.append(("caption", text))
            continue
        if is_code_para(p):
            blocks.append(("code", p.text.rstrip()))
            continue
        if "List Bullet" in name:
            blocks.append(("bullet", text))
            continue
        m = re.match(r"^(\d+)[\u3000](.*)$", text)
        if m:
            blocks.append(("num", (m.group(1), m.group(2).strip())))
            continue
        if not blocks:
            # 第一个标题之前的内容当作封面
            size = max_font_size(p)
            if size >= 18:
                cover.append(("title", text))
            elif size >= 14:
                cover.append(("sub", text))
            else:
                cover.append(("meta", text))
            continue
        blocks.append(("para", text))
    # 合并连续代码段
    merged: list[tuple[str, object]] = []
    for kind, payload in blocks:
        if kind == "code" and merged and merged[-1][0] == "code":
            merged[-1] = ("code", str(merged[-1][1]) + "\n" + str(payload))
        else:
            merged.append((kind, payload))

    lines: list[str] = []

    def blank() -> None:
        if lines and lines[-1] != "":
            lines.append("")

    title = next((t for k, t in cover if k == "title"), None) or src.stem
    lines.append(f"# {title}")
    lines.append("")
    has_sub = any(k == "sub" for k, _ in cover)
    has_meta = any(k == "meta" for k, _ in cover)
    for k, t in cover:
        if k == "sub":
            lines.append(f"**{t}**")
            lines.append("")
    if has_sub and has_meta:
        pass
    for k, t in cover:
        if k == "meta":
            lines.append(f"> {t}")
    if has_meta:
        lines.append("")

    # 封面区域里可能还有一张信息表，先输出来
    first_h1 = next((i for i, (k, _) in enumerate(merged) if k == "h1"), len(merged))
    for kind, payload in merged[:first_h1]:
        if kind == "table":
            lines.extend(md_table(payload))  # type: ignore[arg-type]
            lines.append("")
        elif kind == "para":
            lines.append(str(payload))
            lines.append("")
    merged = merged[first_h1:]

    lines.append("> 仓库：<https://github.com/mylatnem/selfrising-rl> · 本文是 `docs/学习手册.docx` 的 Markdown 版")
    lines.append("")

    headings = [(int(k[1]), str(t)) for k, t in merged if k in ("h1", "h2", "h3")]
    lines.append("## 目录")
    lines.append("")
    for lvl, text in headings:
        indent = "    " * (lvl - 1)
        lines.append(f"{indent}- {text}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for kind, payload in merged:
        if kind == "h1":
            blank()
            lines.append(f"## {payload}")
            lines.append("")
        elif kind == "h2":
            blank()
            lines.append(f"### {payload}")
            lines.append("")
        elif kind == "h3":
            blank()
            lines.append(f"#### {payload}")
            lines.append("")
        elif kind == "caption":
            blank()
            lines.append(f"**{payload}**")
            lines.append("")
        elif kind == "code":
            blank()
            body = str(payload).rstrip()
            lines.append(f"```{guess_lang(body.splitlines())}")
            lines.append(body)
            lines.append("```")
            lines.append("")
        elif kind == "table":
            lines.extend(md_table(payload))  # type: ignore[arg-type]
            lines.append("")
        elif kind == "bullet":
            lines.append(f"- {payload}")
        elif kind == "num":
            n, t = payload  # type: ignore[misc]
            lines.append(f"{n}. {t}")
        else:
            blank()
            lines.append(str(payload))
            lines.append("")

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.rstrip() + "\n\n---\n\n*本文档由 `tools/docx2md.py` 从 `docs/学习手册.docx` 自动生成。改内容请改 docx，再重新生成；两边的章节号与表格编号一致。*\n"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"已生成 {dst}  ({len(text.splitlines())} 行, {len(text)} 字符)")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="docx -> markdown")
    ap.add_argument("-i", "--input", default=str(root / "docs" / "学习手册.docx"))
    ap.add_argument("-o", "--output", default=str(root / "docs" / "学习手册.md"))
    args = ap.parse_args()
    convert(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

