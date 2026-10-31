from __future__ import annotations

import pytest

from gold_neural.search import (
    Candidate,
    SearchConfig,
    generate_candidates,
    refine_candidates,
    score,
)


def test_untradeable_targets_are_pruned():
    config = SearchConfig(fee_bps=10, slippage_bps=2, min_target_ratio=2.5)
    candidates = generate_candidates(atr_bps=8.0, config=config)

    assert candidates
    cost = config.round_trip_cost_bps
    for candidate in candidates:
        assert candidate.target_bps >= config.min_target_ratio * cost
        assert candidate.k_down <= candidate.k_up


def test_costs_that_cannot_be_beaten_produce_nothing():
    # 100 bps fee + 10 bps slippage each side = 220 bps round trip
    config = SearchConfig(fee_bps=100, slippage_bps=10)
    assert generate_candidates(atr_bps=8.0, config=config) == []


def test_holding_must_plausibly_reach_the_target():
    config = SearchConfig(fee_bps=0, slippage_bps=0, min_target_ratio=1.0)
    candidates = generate_candidates(atr_bps=8.0, config=config)

    # a 12-bar hold cannot cover 16 ATR
    assert all(not (c.k_up == 16.0 and c.holding < 96) for c in candidates)
    for candidate in candidates:
        assert candidate.k_up <= 1.5 * (candidate.holding ** 0.5)


def test_candidates_are_ordered_by_headroom_then_holding():
    config = SearchConfig()
    candidates = generate_candidates(atr_bps=8.0, config=config)
    ordering = [(c.ratio_to_cost, c.holding, c.k_down, c.window) for c in candidates]
    assert ordering == sorted(ordering)


def test_candidate_cap_is_respected():
    config = SearchConfig(max_candidates=5)
    assert len(generate_candidates(atr_bps=8.0, config=config)) == 5


def test_refine_uses_top_candidates_with_new_windows():
    config = SearchConfig(refine_top=1, refine_windows=(32, 128))
    best = [
        {
            "candidate": {
                "k_up": 8.0,
                "k_down": 3.0,
                "holding": 96,
                "window": 64,
                "target_bps": 64.8,
                "ratio_to_cost": 2.7,
            }
        }
    ]
    refined = refine_candidates(best, config)

    assert [c.window for c in refined] == [32, 128]
    assert all(c.k_up == 8.0 and c.holding == 96 for c in refined)
    assert all(c.phase == 2 for c in refined)


def test_refine_can_be_disabled():
    config = SearchConfig(refine_top=0)
    assert refine_candidates([{"candidate": {"k_up": 8, "k_down": 3, "holding": 96, "window": 64}}], config) == []


def test_score_needs_enough_trades():
    assert score({"validation": {"metrics": {"trades": 3, "total_return": 5.0}}}, 20) == float("-inf")
    assert score({"validation": {"metrics": {}}}, 20) == float("-inf")


def test_score_penalises_drawdown():
    shallow = score(
        {"validation": {"metrics": {"trades": 100, "total_return": 0.10, "max_drawdown": 0.0}}}, 20
    )
    deep = score(
        {"validation": {"metrics": {"trades": 100, "total_return": 0.10, "max_drawdown": 0.4}}}, 20
    )
    assert shallow > deep


def test_search_config_validation():
    with pytest.raises(ValueError):
        SearchConfig(total_minutes=0)
    with pytest.raises(ValueError):
        SearchConfig(per_candidate_minutes=0)
    with pytest.raises(ValueError):
        SearchConfig(max_candidates=0)


def test_candidate_keys_are_distinct():
    a = Candidate(k_up=8, k_down=3, holding=96, window=64)
    b = Candidate(k_up=8, k_down=3, holding=96, window=128)
    assert a.key() != b.key()
    assert "k_up=8" in a.label()
