from open_llm_router.utils.sequence_utils import dedupe_preserving_order


def test_dedupe_preserving_order_retains_first_occurrence() -> None:
    values = ["alpha", "beta", "alpha", "gamma", "beta", "delta"]
    assert dedupe_preserving_order(values) == ["alpha", "beta", "gamma", "delta"]


def test_dedupe_preserving_order_accepts_iterables() -> None:
    values = (item for item in ["x", "x", "y", "x", "z"])
    assert dedupe_preserving_order(values) == ["x", "y", "z"]
