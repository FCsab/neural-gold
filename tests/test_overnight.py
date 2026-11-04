"""The detached runner's spec stream must never idle on finished work."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "overnight.py"
    spec = importlib.util.spec_from_file_location("overnight_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spec_stream_advances_past_the_supplied_seeds():
    module = _module()
    stream = module.spec_stream(
        k_up=(1.0,), k_down=(1.0,), holding=(1,), window=(1,), seeds=(7, 8)
    )

    seeds = [next(stream)["seed"] for _ in range(5)]

    # Two supplied seeds, then fresh ones: no pass can ever be all-duplicates
    # of keys already recorded on disk.
    assert seeds == [7, 8, 9, 10, 11]


def test_spec_stream_repeats_the_grid_for_each_seed():
    module = _module()
    stream = module.spec_stream(
        k_up=(2.0, 3.0), k_down=(1.0,), holding=(1,), window=(1,), seeds=(7,)
    )

    first = [next(stream) for _ in range(2)]
    second = next(stream)

    assert [spec["k_up"] for spec in first] == [2.0, 3.0]
    assert [spec["pass_index"] for spec in first] == [1, 2]
    assert first[0]["pass_total"] == 2
    assert second["k_up"] == 2.0
    assert second["seed"] == 8
    assert second["pass_index"] == 1
