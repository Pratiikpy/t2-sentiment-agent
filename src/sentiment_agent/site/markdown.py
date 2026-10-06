"""A small Markdown renderer for the whitepaper page.

The published page must need nothing from outside the folder (DESIGN.md §14.7), so the whitepaper
is rendered at export time into the site's own stylesheet rather than fetched or rendered in the
browser. The subset is what ``WHITEPAPER.md`` uses and nothing more: ATX headings, paragraphs,
bullet and numbered lists, pipe tables, fenced code, block quotes, horizontal rules, and inline
code, emphasis and links. Everything is HTML-escaped first; a Markdown file cannot inject markup.
"""

from __future__ import annotations

import re
from html import escape
from typing import Final

__all__ = ["render_markdown"]

_INLINE_CODE: Final = re.compile(r"`([^`]+)`")
_STRONG: Final = re.compile(r"\*\*(.+?)\*\*")
_EM: Final = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_LINK: Final = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+|[A-Za-z0-9_./#-]+)\)")
_HEADING: Final = re.compile(r"^(#{1,4})\s+(.*?)\s*#*$")
_BULLET: Final = re.compile(r"^[-*]\s+(.*)$")
_NUMBERED: Final = re.compile(r"^\d+\.\s+(.*)$")
_TABLE_RULE: Final = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_ANCHOR_STRIP: Final = re.compile(r"[^a-z0-9]+")


def _inline(text: str) -> str:
    """Inline markup on an already-escaped line. Code spans are rendered first and protected, so
    emphasis markers inside them stay literal."""
    spans: list[str] = []

    def keep(match: re.Match[str]) -> str:
        spans.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    out = _INLINE_CODE.sub(keep, text)
    out = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    out = _STRONG.sub(r"<strong>\1</strong>", out)
    out = _EM.sub(r"<em>\1</em>", out)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], out)


def _anchor(title: str) -> str:
    plain = re.sub(r"<[^>]+>", "", title).lower()
    return _ANCHOR_STRIP.sub("-", plain).strip("-") or "section"


def _cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def render_markdown(text: str) -> str:
    """The Markdown subset above, as HTML. Unknown constructs render as paragraphs, never as
    markup: the input is escaped before any rule runs."""
    lines = escape(text, quote=False).replace("&quot;", '"').splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            code: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(code) + "</code></pre>")
            continue
        heading = _HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = _inline(heading.group(2))
            out.append(f'<h{level} id="{_anchor(title)}">{title}</h{level}>')
            i += 1
            continue
        if stripped in {"---", "***", "___"}:
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < n and _TABLE_RULE.match(lines[i + 1].strip()):
            head = _cells(stripped)
            i += 2
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            ths = "".join(f"<th>{_inline(c)}</th>" for c in head)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>")
            continue
        if stripped.startswith("&gt;"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith("&gt;"):
                quote.append(lines[i].strip()[4:].strip())
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(quote))}</p></blockquote>")
            continue
        bullet = _BULLET.match(stripped)
        numbered = _NUMBERED.match(stripped)
        if bullet or numbered:
            tag = "ul" if bullet else "ol"
            pattern = _BULLET if bullet else _NUMBERED
            items: list[str] = []
            while i < n:
                current = lines[i].strip()
                match = pattern.match(current)
                if match:
                    items.append(match.group(1))
                    i += 1
                elif current and lines[i].startswith("  ") and items:
                    items[-1] += " " + current
                    i += 1
                else:
                    break
            lis = "".join(f"<li>{_inline(item)}</li>" for item in items)
            out.append(f"<{tag}>{lis}</{tag}>")
            continue
        para: list[str] = [stripped]
        i += 1
        while i < n:
            current = lines[i].strip()
            if (
                not current
                or current.startswith(("```", "|", "#", "&gt;"))
                or _BULLET.match(current)
                or _NUMBERED.match(current)
                or current in {"---", "***", "___"}
            ):
                break
            para.append(current)
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)
