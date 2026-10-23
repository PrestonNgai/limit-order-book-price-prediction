from pathlib import Path

import pytest

from market_microstructure.utils.config import CollectorConfig, load_collector_config


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_queue", 0),
        ("max_queue", True),
        ("idle_timeout", float("nan")),
        ("report_seconds", 0),
        ("max_reconnects", -1),
        ("compress", "false"),
        ("max_file_bytes", 1.5),
        ("symbol", "../BTC-USD"),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError):
        CollectorConfig(**{key: value})


def test_paths_are_relative_to_config_or_explicit_cli(tmp_path):
    path = tmp_path / "collector.yaml"
    path.write_text("symbol: BTC-USD\noutput_dir: raw\n")
    assert load_collector_config(path).output_dir == tmp_path / "raw"
    assert load_collector_config(path, output_dir="raw").output_dir == Path("raw").resolve()


def test_unknown_keys_are_errors(tmp_path):
    path = tmp_path / "collector.yaml"
    path.write_text("max_quee: 2\n")
    with pytest.raises(ValueError, match="unknown"):
        load_collector_config(path)
