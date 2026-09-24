"""The demo walkthrough of the published page: a Playwright-scripted video of 3 minutes or less.

    python scripts/record_video.py --public public/ --out video/ [--mp4]

Every frame comes from the real exported record: the script opens ``public/index.html`` (after
``t2sa export``), walks the page section by section the way a judge would read it, opens the
first decision card whose orders filled and walks it top to bottom, returns, shows the
venue-integrity replay and the verification steps, switches to the dark theme, and stops. Nothing
is staged: the page is static, so what the video shows is exactly what the files say.

**When no order has filled yet** (a dry run, or a paper record before its first fill) the card step
falls back to the first decision card on the timeline, and when the timeline has no card at all it
stays on the record: the card-only steps then find no anchor and hold on the page, and the "back"
step does nothing, so the walkthrough still records end to end instead of navigating away.

The walkthrough is a fixed script with a fixed time budget per step (:data:`STEPS`), checked
before anything is recorded to fit inside :data:`MAX_SECONDS`, and the recording is re-measured
afterwards. The browser runs headless with no network access at all (every request that is not
``file:`` is aborted), so the recording also proves the page needs nothing from outside.

Output: ``<out>/walkthrough.webm`` (Playwright's own recording), and ``<out>/walkthrough.mp4``
when ``--mp4`` is given and ``ffmpeg`` is on PATH (H.264, the format X accepts).

Requires the ``video`` extra (``pip install -e ".[video]"``) and ``python -m playwright install
chromium``.
"""

import argparse
import importlib
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


def _sync_playwright() -> Any:
    """Playwright's sync entry point, imported only when a recording is made (the ``video`` extra is
    optional, and nothing else in the project needs it)."""
    try:
        module = importlib.import_module("playwright.sync_api")
    except ImportError:
        raise SystemExit(
            'Playwright is not installed: pip install -e ".[video]" && '
            "python -m playwright install chromium"
        ) from None
    return module.sync_playwright


MAX_SECONDS: Final = 180.0
"""The handbook's demo video budget used for this entry: three minutes."""

WIDTH: Final = 1280
HEIGHT: Final = 720


@dataclass(frozen=True, slots=True)
class Step:
    what: str
    seconds: float
    """How long the viewer is given on this step, after the action."""
    action: Callable[[Any], None]


def _scroll_to(anchor: str) -> Callable[[Any], None]:
    def act(page: Any) -> None:
        if page.locator(f"#{anchor}").count() == 0:
            return
        page.evaluate(
            "id => document.getElementById(id)"
            ".scrollIntoView({behavior: 'smooth', block: 'start'})",
            anchor,
        )

    return act


def _glide(pixels: int) -> Callable[[Any], None]:
    def act(page: Any) -> None:
        page.evaluate("dy => window.scrollBy({top: dy, behavior: 'smooth'})", pixels)

    return act


_OPENED: Final[list[bool]] = []
"""Whether the card step navigated to a card (one entry per recording), read by :func:`_back`."""


def _open_card(selector: str) -> Callable[[Any], None]:
    def act(page: Any) -> None:
        link = page.locator(selector).first
        if link.count() == 0:  # no card with a fill yet: the first decision card instead
            link = page.locator("ol.timeline a").first
        if link.count() == 0:  # no card at all: stay on the record
            print("  (no decision card on the timeline; the walkthrough stays on the record)")
            _OPENED.append(False)
            return
        link.click()
        page.wait_for_load_state("load")
        _OPENED.append(True)

    return act


def _open_first(selector: str) -> Callable[[Any], None]:
    def act(page: Any) -> None:
        summary = page.locator(f"{selector} > summary").first
        if summary.count():
            summary.click()

    return act


def _then(*actions: Callable[[Any], None]) -> Callable[[Any], None]:
    def act(page: Any) -> None:
        for action in actions:
            action(page)

    return act


def _back(page: Any) -> None:
    if _OPENED and _OPENED[-1]:
        page.go_back()
        page.wait_for_load_state("load")


def _dark(page: Any) -> None:
    button = page.locator("#theme-toggle")
    if button.is_visible():
        button.click()


def _top(page: Any) -> None:
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")


STEPS: Final[tuple[Step, ...]] = (
    Step("the record opens: mode, headline numbers, the no-edge envelope", 9.0, _top),
    Step("event -> decision -> execution, every cycle", 8.0, _scroll_to("timeline")),
    Step("the timeline, continued", 7.0, _glide(520)),
    Step(
        "a decision card that sent orders",
        6.0,
        _open_card("ol.timeline li:has(.badge.good) a"),
    ),
    Step("what woke the agent and what it saw", 7.0, _scroll_to("coverage")),
    Step("the text the model was shown, quarantine marked", 7.0, _scroll_to("text")),
    Step("Qwen's decision: thesis, invalidation, crowd vs us", 9.0, _scroll_to("decision")),
    Step("the kernel's ruling, guard by guard", 9.0, _scroll_to("kernel")),
    Step(
        "orders: dry-run payload, clientOid, venue orderId, fills",
        9.0,
        _then(_scroll_to("orders"), _open_first("#orders details")),
    ),
    Step("proof: ledger rows and blob hashes", 6.0, _scroll_to("proof")),
    Step("back to the record", 3.0, _back),
    Step("equity against every arm", 9.0, _scroll_to("equity")),
    Step("the guard funnel", 9.0, _scroll_to("kernel")),
    Step("governed against ungoverned", 7.0, _scroll_to("twin")),
    Step("the live-marked mirror", 7.0, _scroll_to("mirror")),
    Step("the red team", 6.0, _scroll_to("redteam")),
    Step("the Bitget toolkit, used and not used", 9.0, _scroll_to("toolkit")),
    Step("environment proof and pre-registration", 9.0, _scroll_to("proof")),
    Step("replay: a recorded venue-integrity refusal", 10.0, _scroll_to("replay")),
    Step("how to verify all of it", 7.0, _scroll_to("verify")),
    Step("the same record, dark theme", 6.0, _dark),
    Step("end", 3.0, _top),
)


def planned_seconds() -> float:
    return sum(step.seconds for step in STEPS)


def _allow_only_files(route: Any) -> None:
    if route.request.url.startswith("file:"):
        route.continue_()
    else:
        route.abort()


def record(public: Path, out: Path, *, mp4: bool) -> Path:
    sync_playwright = _sync_playwright()
    index = (public / "index.html").resolve()
    if not index.is_file():
        raise SystemExit(f"{index.name} not found: run the export and the render first")
    budget = planned_seconds()
    if budget > MAX_SECONDS - 5:
        raise SystemExit(f"the script plans {budget:.0f} s, over the {MAX_SECONDS:.0f} s budget")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    blocked: list[str] = []
    _OPENED.clear()
    started = time.monotonic()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            color_scheme="light",
            record_video_dir=str(raw),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        context.route("**/*", _allow_only_files)
        page = context.new_page()
        page.on("requestfailed", lambda request: blocked.append(request.url))
        page.goto(index.as_uri())
        page.wait_for_load_state("load")
        for number, step in enumerate(STEPS, start=1):
            print(f"[{number:02d}/{len(STEPS)}] {step.what}", flush=True)
            step.action(page)
            page.wait_for_timeout(int(step.seconds * 1000))
        video = page.video
        context.close()
        browser.close()
        if video is None:
            raise SystemExit("Playwright recorded no video")
        source = Path(video.path())
    elapsed = time.monotonic() - started
    webm = out / "walkthrough.webm"
    shutil.move(str(source), webm)
    shutil.rmtree(raw, ignore_errors=True)
    outside = [u for u in blocked if not u.startswith("file:")]
    print(f"recorded {elapsed:.0f} s ({budget:.0f} s planned) -> {webm}")
    if outside:
        print(f"WARNING: the page tried {len(outside)} request(s) outside the folder: {outside}")
    if elapsed > MAX_SECONDS:
        raise SystemExit(f"the recording ran {elapsed:.0f} s, over {MAX_SECONDS:.0f} s")
    if not mp4:
        return webm
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is not on PATH; kept the .webm only")
        return webm
    target = out / "walkthrough.mp4"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(webm),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True)  # noqa: S603 - fixed argv, the resolved ffmpeg binary
    print(f"converted -> {target}")
    return target


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--public", type=Path, default=Path("public"))
    parser.add_argument("--out", type=Path, default=Path("video"))
    parser.add_argument("--mp4", action="store_true", help="also write an H.264 .mp4 (ffmpeg)")
    parser.add_argument(
        "--plan", action="store_true", help="print the script and its timing, record nothing"
    )
    args = parser.parse_args(argv)
    if args.plan:
        for step in STEPS:
            print(f"{step.seconds:5.1f} s  {step.what}")
        print(f"{planned_seconds():5.1f} s  total (budget {MAX_SECONDS:.0f} s)")
        return 0
    record(args.public, args.out, mp4=args.mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
