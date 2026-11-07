from __future__ import annotations

import pytest

from crypto_neural.registry import Registry, new_id


def _make(registry: Registry, name: str = "m1") -> str:
    return registry.create_model(
        name=name,
        interval="5m",
        architecture="transformer_v1",
        window=64,
        label_config={"k_up": 2.0, "k_down": 1.5},
        feature_config={"feature_lag": 1},
    )


def test_create_and_list(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    model_id = _make(registry)

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["id"] == model_id
    assert models[0]["status"] == "training"
    assert models[0]["label_config"] == {"k_up": 2.0, "k_down": 1.5}
    assert models[0]["window"] == 64


def test_update_roundtrip(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    model_id = _make(registry)

    registry.update_model(
        model_id,
        status="candidate",
        checkpoint_path="/tmp/ckpt.pt",
        checkpoint_version=3,
        metrics={"balanced_accuracy": 0.41},
    )

    model = registry.get_model(model_id)
    assert model["status"] == "candidate"
    assert model["checkpoint_version"] == 3
    assert model["metrics"]["balanced_accuracy"] == 0.41


def test_update_rejects_unknown_field(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    model_id = _make(registry)
    with pytest.raises(ValueError, match="cannot update"):
        registry.update_model(model_id, nonsense=1)


def test_missing_model_returns_none(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    assert registry.get_model("model_nope") is None


def test_runs_are_recorded_and_listed(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    model_id = _make(registry)

    registry.record_run(
        model_id=model_id,
        started_at="2026-10-31T10:00:00+00:00",
        status="ok",
        config={"window": 64},
        final_metrics={"balanced_accuracy": 0.4},
        per_class_metrics={"1": {"f1": 0.3}},
        confusion_matrix=[[5, 2, 1], [3, 6, 2], [1, 2, 7]],
        epochs_done=12,
        stop_reason="completed",
        duration_s=42.5,
        device="cuda",
    )

    runs = registry.list_runs(model_id)
    assert len(runs) == 1
    assert runs[0]["confusion_matrix"] == [[5, 2, 1], [3, 6, 2], [1, 2, 7]]
    assert runs[0]["epochs_done"] == 12
    assert runs[0]["device"] == "cuda"
    assert registry.list_runs("model_other") == []


def test_ids_are_prefixed_and_unique():
    first, second = new_id("model"), new_id("model")
    assert first.startswith("model_")
    assert first != second


def test_a_candidate_that_loses_to_the_majority_guess_says_so():
    from crypto_neural.registry import compare_to_baselines

    # The shape of the real BTCUSDC candidates: balanced accuracy clears the
    # bar, plain accuracy loses to always guessing the commonest outcome.
    c = compare_to_baselines({
        "accuracy": 0.4168, "baseline_accuracy": 0.5514, "balanced_accuracy": 0.4616,
        "baseline_balanced_accuracy": 1 / 3, "beats_baseline": True,
    })
    assert c["beats_balanced"] is True and c["beats_majority"] is False
    assert "less often right" in c["verdict"]


def test_comparison_verdicts_cover_every_case():
    from crypto_neural.registry import compare_to_baselines

    base = {"baseline_accuracy": 0.5, "baseline_balanced_accuracy": 1 / 3,
            "balanced_accuracy": 0.4}
    both = compare_to_baselines({**base, "accuracy": 0.6, "beats_baseline": True})
    neither = compare_to_baselines({**base, "accuracy": 0.4, "beats_baseline": False})
    majority_only = compare_to_baselines({**base, "accuracy": 0.6, "beats_baseline": False})
    assert both["verdict"] == "beats both kinds of guessing"
    assert neither["verdict"] == "no better than guessing"
    assert majority_only["beats_majority"] and not majority_only["beats_balanced"]
    assert compare_to_baselines(None)["verdict"] == "not tested"
    assert compare_to_baselines({})["beats_majority"] is None
