from analysis.reporting.factor_report import generate_markdown_report


def _sample_backtest_data():
    return {
        "rows": 1234,
        "summary": [
            {
                "score_column": "composite_balanced_score",
                "horizon": 20,
                "rank_ic_mean": 0.01,
                "rank_ic_ir": 0.1,
                "top_quantile_return": 0.02,
                "bottom_quantile_return": 0.01,
                "top_bottom_spread": 0.01,
                "ic_observations": 10,
            },
            {
                "score_column": "composite_cycle_oriented_score",
                "horizon": 20,
                "rank_ic_mean": 0.08,
                "rank_ic_ir": 0.6,
                "top_quantile_return": 0.05,
                "bottom_quantile_return": -0.01,
                "top_bottom_spread": 0.06,
                "ic_observations": 10,
            },
            {
                "score_column": "value_score",
                "horizon": 20,
                "rank_ic_mean": -0.05,
                "rank_ic_ir": -0.3,
                "top_quantile_return": -0.01,
                "bottom_quantile_return": 0.04,
                "top_bottom_spread": -0.05,
                "ic_observations": 10,
            },
            {
                "score_column": "cycle_score",
                "horizon": 5,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.2,
                "top_quantile_return": 0.02,
                "bottom_quantile_return": 0.0,
                "top_bottom_spread": 0.02,
                "ic_observations": 12,
            },
        ],
    }


def test_generate_markdown_report_highlights_best_and_inverse_rows():
    report = generate_markdown_report(_sample_backtest_data(), title="CSI1000 Backtest")

    assert "# CSI1000 Backtest" in report
    assert "Rows: 1234" in report
    assert "Best 20d spread: `composite_cycle_oriented_score`" in report
    assert "Weakest spread: `value_score`" in report
    assert "Potential Inverse Signals" in report
    assert "| 1 | `composite_cycle_oriented_score` | 20 |" in report


def test_generate_markdown_report_can_filter_composite_scores():
    report = generate_markdown_report(_sample_backtest_data(), score_group="composite", horizons=(20,))

    assert "`composite_cycle_oriented_score`" in report
    assert "`composite_balanced_score`" in report
    assert "`value_score`" not in report
    assert "## 5d Ranking" not in report


def test_generate_markdown_report_can_filter_atomic_scores():
    report = generate_markdown_report(_sample_backtest_data(), score_group="atomic")

    assert "`value_score`" in report
    assert "`cycle_score`" in report
    assert "`composite_cycle_oriented_score`" not in report
