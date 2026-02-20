import pytest
from pydantic import ValidationError

from open_llm_router.config import BackendAccount
from open_llm_router.profile.profile_config import ProfileAccountConfig


def test_backend_account_defaults_provider_to_openai() -> None:
    account = BackendAccount(name="acct-a", base_url="http://localhost:8080")
    assert account.provider == "openai"


def test_profile_account_requires_provider() -> None:
    with pytest.raises(ValidationError):
        ProfileAccountConfig.model_validate({"name": "acct-a"})
