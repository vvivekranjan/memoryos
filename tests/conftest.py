import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

# Ensure repository root is on sys.path during pytest collection
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.dont_write_bytecode = True

TEST_TMP = ROOT / ".tmp-check" / "tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)

os.environ["TMP"] = str(TEST_TMP)
os.environ["TEMP"] = str(TEST_TMP)
os.environ["TMPDIR"] = str(TEST_TMP)
tempfile.tempdir = str(TEST_TMP)


@pytest.fixture
def workspace_tmp_path() -> Path:
    path = TEST_TMP / str(uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path
