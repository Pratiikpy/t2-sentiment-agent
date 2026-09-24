"""Fixtures for the site tests: one simulated day, exported and rendered once per session.

The day is played by the real pipeline (``tests/site_world.py``). Tests that need to break
something (a planted secret, a tampered blob, an edited ledger line) work on their own copy or on a
minimal ledger of their own (``site_world.copy_world``, ``site_world.minimal_ledger``), never on
the shared export.
"""

import pytest

from site_world import Site, build_site


@pytest.fixture(scope="session")
def site(tmp_path_factory: pytest.TempPathFactory) -> Site:
    return build_site(tmp_path_factory.mktemp("site"))
