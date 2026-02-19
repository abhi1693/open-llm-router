from open_llm_router.model_utils import default_model_id, split_model_ref


def test_default_model_id_extracts_suffix_when_provider_is_present() -> None:
    assert default_model_id("openai/gpt-5.2") == "gpt-5.2"


def test_default_model_id_returns_input_for_invalid_refs() -> None:
    assert default_model_id("openai/") == "openai/"
    assert default_model_id("/gpt-5.2") == "/gpt-5.2"


def test_split_model_ref_parses_and_normalizes_valid_refs() -> None:
    assert split_model_ref(" openai / gpt-5.2 ") == ("openai", "gpt-5.2")


def test_split_model_ref_returns_normalized_raw_for_invalid_refs() -> None:
    assert split_model_ref("openai/") == (None, "openai/")
    assert split_model_ref(" /gpt-5.2 ") == (None, "/gpt-5.2")
