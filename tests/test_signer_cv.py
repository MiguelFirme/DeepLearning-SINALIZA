"""Testes do resumo consolidado da validação por articulador."""

from scripts.run_signer_cv import aggregate_results


def test_aggregate_results_calculates_mean_and_std():
    results = {
        "Articulador1": {"top1": 0.1, "loss": 2.0},
        "Articulador2": {"top1": 0.2, "loss": 1.0},
        "Articulador3": {"top1": 0.3, "loss": 3.0},
    }
    aggregate = aggregate_results(results)
    assert abs(aggregate["top1"]["mean"] - 0.2) < 1e-12
    assert aggregate["loss"]["mean"] == 2.0
    assert aggregate["loss"]["min"] == 1.0
    assert aggregate["loss"]["max"] == 3.0
