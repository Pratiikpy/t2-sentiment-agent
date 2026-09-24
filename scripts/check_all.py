"""The gate every change passes: ruff, ruff format, mypy --strict, pytest.

    python scripts/check_all.py            # offline: no network, no credentials, no Qwen
    python scripts/check_all.py --live-public   # also runs keyless public Bitget market tests

Exits non-zero on the first failing step and names it.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    pytest_args = ["-q"]
    if "--live-public" in argv:
        pytest_args.append("--run-live-public")
    steps: list[tuple[str, list[str]]] = [
        ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
        ("ruff format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("mypy --strict", [sys.executable, "-m", "mypy"]),
        ("pytest", [sys.executable, "-m", "pytest", *pytest_args]),
    ]
    for name, command in steps:
        print(f"== {name}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603 - fixed argv
        if result.returncode != 0:
            print(f"FAILED: {name}", flush=True)
            return result.returncode
    print("all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
