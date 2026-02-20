from __future__ import annotations

from pathlib import Path


class CatalogDataPaths:
    @staticmethod
    def data_dir() -> Path:
        return Path(__file__).resolve().parent / "data"

    @classmethod
    def providers_yaml(cls) -> Path:
        return cls.data_dir() / "providers.yaml"

    @classmethod
    def models_yaml(cls) -> Path:
        return cls.data_dir() / "models.yaml"

    @classmethod
    def profiles_yaml(cls) -> Path:
        return cls.data_dir() / "profiles.yaml"
