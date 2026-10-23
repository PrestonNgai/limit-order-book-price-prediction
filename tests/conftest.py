import json
from pathlib import Path

import pytest


@pytest.fixture
def messages():
    return json.loads((Path(__file__).parent / "fixtures/coinbase_messages.json").read_text())
