"""The static page: complete, self-contained, well formed, every card linked, text escaped.

Visual QA (light, dark, 400 px) is done by screenshot with Playwright when the page changes; a test
cannot start a browser (``tests/conftest.py`` refuses every child process but Python). What a test
can prove is here: no external request is possible, the structure is valid, and every section and
card is present.
"""

import importlib.util
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

import pytest

from sentiment_agent.site.export import ExportRefused
from sentiment_agent.site.render import (
    CSP,
    SECTIONS,
    RenderError,
    load_export,
    render_card,
    render_site,
)
from sentiment_agent.types import ScreenedItem, TextItem
from site_world import ROOT, Site

VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
URL_ATTRIBUTES = frozenset({"href", "src", "action", "formaction", "poster", "data", "srcset"})


class Page(HTMLParser):
    """A strict walk of one page: tag balance, ids, links and every URL-bearing attribute."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: list[str] = []
        self.links: list[str] = []
        self.urls: list[str] = []
        self.tags: list[str] = []
        self.meta: list[dict[str, str]] = []
        self.doctype = False

    def handle_decl(self, decl: str) -> None:
        self.doctype = decl.lower() == "doctype html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        self.tags.append(tag)
        if tag == "meta":
            self.meta.append(values)
        if "id" in values:
            self.ids.append(values["id"])
        for name, value in values.items():
            if name in URL_ATTRIBUTES:
                self.urls.append(value)
            if name.startswith("on"):
                self.errors.append(f"inline event handler {name} on <{tag}>")
        if tag == "a" and "href" in values:
            self.links.append(values["href"])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        self.tags.append(tag)
        for name, value in values.items():
            if name in URL_ATTRIBUTES:
                self.urls.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1] if self.stack else 'nothing'}>")
            return
        self.stack.pop()


def parse(path: Path) -> Page:
    page = Page()
    page.feed(path.read_text("utf-8"))
    page.close()
    return page


def pages(site: Site) -> list[Path]:
    return [site.public / "index.html", *sorted((site.public / "cards").glob("*.html"))]


# ------------------------------------------------------------------------------------------------


def test_the_index_and_one_page_per_card_are_rendered(site: Site) -> None:
    ex = load_export(site.public)
    expected = {site.public / "index.html"} | {
        site.public / "cards" / f"{c.card_id}.html" for c in ex.cards
    }
    assert set(site.pages) == expected
    assert all(p.is_file() for p in site.pages)


def test_every_page_is_well_formed(site: Site) -> None:
    for path in pages(site):
        page = parse(path)
        assert page.doctype, path.name
        assert page.errors == [], path.name
        assert page.stack == [], path.name
        for tag in ("html", "head", "body"):
            assert page.tags.count(tag) == 1, (path.name, tag)
        head = path.read_text("utf-8").split("</head>", 1)[0]
        assert head.count("<title>") == 1, f"{path.name}: one document title"
        assert len(page.ids) == len(set(page.ids)), f"{path.name}: ids are unique"
        names = {m.get("name") or m.get("http-equiv") or m.get("charset") for m in page.meta}
        assert {"viewport", "Content-Security-Policy", "utf-8", "color-scheme"} <= names


def test_no_page_can_make_an_external_request(site: Site) -> None:
    for path in pages(site):
        page = parse(path)
        text = path.read_text("utf-8")
        for url in page.urls:
            assert not re.match(r"(?i)^\s*(?:[a-z][a-z0-9+.-]*:|//)", url), (path.name, url)
        assert "<link" not in text, "no external stylesheet"
        assert not re.search(r"<script[^>]+src=", text), "no external script"
        assert "@import" not in text
        for found in re.findall(r"url\(([^)]*)\)", text):
            assert found.strip("'\" ").startswith("data:"), found
        csp = next(m for m in page.meta if m.get("http-equiv") == "Content-Security-Policy")
        assert csp["content"] == CSP
        assert "default-src 'none'" in CSP
        assert "connect-src" not in CSP, "default-src 'none' already forbids every fetch"


def test_every_card_is_linked_from_the_index_and_every_link_resolves(site: Site) -> None:
    index = parse(site.public / "index.html")
    ex = load_export(site.public)
    for card in ex.cards:
        assert f"cards/{card.card_id}.html" in index.links
    for path in pages(site):
        page = parse(path)
        for link in page.links:
            target, _, anchor = link.partition("#")
            if not target:
                assert anchor in page.ids, (path.name, link)
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.is_file(), (path.name, link)
            assert resolved.is_relative_to(site.public.resolve())


def test_every_section_is_on_the_index_and_in_the_nav(site: Site) -> None:
    index = parse(site.public / "index.html")
    for anchor, _ in SECTIONS:
        assert anchor in index.ids
        assert f"#{anchor}" in index.links
    text = (site.public / "index.html").read_text("utf-8")
    assert "SIMULATED record, not the scored paper log" in text
    assert "REPLAY of recorded Bitget data" in text
    assert "G6_turnover" in text
    assert "Coin-flip null (6 seeds" in text


def test_both_themes_and_narrow_screens_are_styled(site: Site) -> None:
    text = (site.public / "index.html").read_text("utf-8")
    assert "@media (prefers-color-scheme: dark)" in text
    assert ':root[data-theme="dark"]' in text
    assert ':root:not([data-theme="light"])' in text
    assert "@media (max-width: 720px)" in text
    assert 'id="theme-toggle"' in text
    assert "overflow-x: auto" in text, "wide tables and charts scroll inside their frame"


def test_untrusted_text_is_escaped(site: Site) -> None:
    ex = load_export(site.public)
    card = next(c for c in ex.cards if c.decision is not None and c.shown_text)
    decision = card.decision
    assert decision is not None
    hostile = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    base = card.shown_text[0]
    item = ScreenedItem(
        item=TextItem.model_validate(
            {**base.item.model_dump(), "text": hostile, "url": "javascript:alert(3)"}
        ),
        detections=(),
        withheld=False,
        prompt_text=hostile,
    )
    evil = card.model_copy(
        update={
            "shown_text": (item,),
            "decision": decision.model_copy(update={"summary": hostile}),
        }
    )
    html = render_card(evil, {})
    assert "<script>alert" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'href="javascript' not in html
    page = Page()
    page.feed(html)
    assert page.errors == []
    assert not any(u.startswith("javascript") for u in page.urls)


def test_the_withheld_injection_is_shown_as_withheld(site: Site) -> None:
    ex = load_export(site.public)
    card = next(c for c in ex.cards if any(t.withheld for t in c.shown_text))
    text = (site.public / "cards" / f"{card.card_id}.html").read_text("utf-8")
    assert "The model was shown only" in text
    assert "What quarantine withheld" in text
    withheld = next(t for t in card.shown_text if t.withheld)
    assert withheld.prompt_text != withheld.item.text


def test_an_incomplete_export_is_not_rendered(site: Site, tmp_path: Path) -> None:
    public = tmp_path / "public"
    shutil.copytree(site.public, public)
    (public / "summary.json").unlink()
    with pytest.raises(RenderError, match=r"summary\.json"):
        render_site(public)


def test_a_card_id_that_is_not_a_file_name_is_refused(site: Site, tmp_path: Path) -> None:
    public = tmp_path / "public"
    shutil.copytree(site.public, public)
    index = json.loads((public / "cards" / "index.json").read_text("utf-8"))
    index[0]["card_id"] = "../../escape"
    (public / "cards" / "index.json").write_text(json.dumps(index), "utf-8")
    with pytest.raises(RenderError, match="safe file name"):
        render_site(public)


def test_the_rendered_pages_are_scanned_before_they_are_published(
    site: Site, tmp_path: Path
) -> None:
    public = tmp_path / "public"
    shutil.copytree(site.public, public)
    before = (public / "index.html").read_bytes()
    summary = json.loads((public / "summary.json").read_text("utf-8"))
    summary["policy_version"] = "loaded from C:\\Users\\someone\\policy.json"
    (public / "summary.json").write_text(json.dumps(summary), "utf-8")
    with pytest.raises(ExportRefused):
        render_site(public)
    assert (public / "index.html").read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if ".staging-" in p.name]


def test_stale_card_pages_are_removed(site: Site, tmp_path: Path) -> None:
    public = tmp_path / "public"
    shutil.copytree(site.public, public)
    stale = public / "cards" / "dec-gone.html"
    stale.write_text("<!DOCTYPE html>", "utf-8")
    written = render_site(public)
    assert not stale.exists()
    assert (public / "index.html") in written


# ------------------------------------------------------------------------------------------------
# The video script
# ------------------------------------------------------------------------------------------------


def video_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "record_video", ROOT / "scripts" / "record_video.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_video_script_fits_three_minutes_and_visits_real_sections(site: Site) -> None:
    module = video_script()
    planned = module.planned_seconds()
    assert isinstance(planned, float)
    assert planned <= module.MAX_SECONDS - 5
    assert module.MAX_SECONDS == 180.0
    source = (ROOT / "scripts" / "record_video.py").read_text("utf-8")
    anchors = set(re.findall(r'_scroll_to\("([a-z]+)"\)', source))
    ex = load_export(site.public)
    card = next(c for c in ex.cards if c.orders and c.decision is not None)
    ids = set(parse(site.public / "index.html").ids)
    ids |= set(parse(site.public / "cards" / f"{card.card_id}.html").ids)
    assert anchors
    assert anchors <= ids, anchors - ids


class _Locator:
    def __init__(self, page: "_Page", selector: str) -> None:
        self.page = page
        self.selector = selector

    @property
    def first(self) -> "_Locator":
        return self

    def count(self) -> int:
        return self.page.present.get(self.selector, 0)

    def click(self) -> None:
        self.page.actions.append(f"click {self.selector}")


class _Page:
    """The few Playwright page calls the card steps make, recorded."""

    def __init__(self, present: dict[str, int]) -> None:
        self.present = present
        self.actions: list[str] = []

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    def wait_for_load_state(self, state: str) -> None:
        self.actions.append(f"wait {state}")

    def go_back(self) -> None:
        self.actions.append("back")


FILLED = "ol.timeline li:has(.badge.good) a"


@pytest.mark.parametrize(
    ("present", "clicked", "went_back"),
    [
        ({FILLED: 1, "ol.timeline a": 2}, FILLED, True),
        ({"ol.timeline a": 1}, "ol.timeline a", True),  # no fill yet: the first card
        ({}, None, False),  # no card at all: stay on the record, never navigate away
    ],
)
def test_the_video_card_step_falls_back_and_never_leaves_the_record(
    present: dict[str, int], clicked: str | None, went_back: bool
) -> None:
    module = video_script()
    module._OPENED.clear()
    page = _Page(present)
    module._open_card(FILLED)(page)
    clicks = [a for a in page.actions if a.startswith("click")]
    assert clicks == ([] if clicked is None else [f"click {clicked}"])
    module._back(page)
    assert ("back" in page.actions) is went_back
