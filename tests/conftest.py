import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def _nb_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("nbhome")
    os.environ["NEUROBLOCKS_HOME"] = str(home)
    os.environ.setdefault("NEUROBLOCKS_EVENTS", "silent")
    os.environ["NEUROBLOCKS_NO_RUN_DIR"] = "1"
    yield home
