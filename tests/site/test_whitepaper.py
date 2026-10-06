"""The whitepaper page: rendered from the one Markdown file, into the site's stylesheet, with
nothing fetched from outside the folder."""

from __future__ import annotations

from pathlib import Path

from sentiment_agent.site.markdown import render_markdown
from sentiment_agent.site.render import render_whitepaper

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "src" / "sentiment_agent" / "site" / "templates" / "WHITEPAPER.md"


def test_the_shipped_copy_is_the_repository_whitepaper() -> None:
    assert TEMPLATE.read_bytes() == (ROOT / "WHITEPAPER.md").read_bytes()


def test_markdown_subset_renders() -> None:
    html = render_markdown(
        "# Title\n\nA **bold** `code` [link](index.html).\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n- one\n- two\n\n> quoted\n\n```\nx = 1\n```\n"
    )
    assert '<h1 id="title">Title</h1>' in html
    assert "<strong>bold</strong> <code>code</code>" in html
    assert '<a href="index.html">link</a>' in html
    assert "<th>a</th>" in html
    assert "<td>2</td>" in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<blockquote><p>quoted</p></blockquote>" in html
    assert "<pre><code>x = 1</code></pre>" in html


def test_markup_in_the_source_is_escaped_not_rendered() -> None:
    html = render_markdown("<script>alert(1)</script> and `<b>`")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<code>&lt;b&gt;</code>" in html


def test_the_page_carries_the_whitepaper_and_no_external_request() -> None:
    html = render_whitepaper(TEMPLATE.read_text(encoding="utf-8"))
    assert "t2-sentiment-agent: whitepaper" in html
    assert 'id="1-abstract"' in html
    assert "default-src" in html
    assert "https://fonts." not in html
    assert "<script src=" not in html
