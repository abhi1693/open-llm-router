from __future__ import annotations

from open_llm_router.catalog_sync import sync_catalog_models_pricing


def test_sync_catalog_models_pricing_updates_using_provider_aliases_and_suffix_matching():
    catalog_document = {
        "version": 1,
        "models": [
            {
                "id": "gpt-5.2",
                "provider": "openai-codex",
                "aliases": [],
                "costs": {"input_per_1k": 0.0, "output_per_1k": 0.0},
            },
            {
                "id": "gemini-2.5-flash",
                "provider": "gemini",
                "aliases": [],
                "costs": {"input_per_1k": 1.0, "output_per_1k": 1.0},
            },
            {
                "id": "qwen2.5-14b-instruct",
                "provider": "openai",
                "aliases": [],
                "costs": {"input_per_1k": 9.0, "output_per_1k": 9.0},
            },
        ],
    }
    openrouter_models = [
        {
            "id": "openai/gpt-5.2",
            "pricing": {"prompt": "0.0000019", "completion": "0.0000067"},
            "created": 1770000001,
        },
        {
            "id": "google/gemini-2.5-flash",
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
            "created": 1770000002,
        },
        {
            "id": "qwen/qwen2.5-14b-instruct",
            "pricing": {"prompt": "0.00000008", "completion": "0.0000002"},
            "created": 1770000003,
        },
    ]

    stats = sync_catalog_models_pricing(
        catalog_document=catalog_document,
        openrouter_models=openrouter_models,
    )

    assert stats.total_local_models == 3
    assert stats.updated == 3
    assert stats.unchanged == 0
    assert stats.missing_remote == 0
    assert stats.missing_pricing == 0

    models = catalog_document["models"]
    assert models[0]["costs"]["input_per_1k"] == 0.0019
    assert models[0]["costs"]["output_per_1k"] == 0.0067
    assert models[1]["costs"]["input_per_1k"] == 0.0002
    assert models[1]["costs"]["output_per_1k"] == 0.0008
    assert models[2]["costs"]["input_per_1k"] == 0.00008
    assert models[2]["costs"]["output_per_1k"] == 0.0002
    assert models[0]["created"] == 1770000001
    assert models[1]["created"] == 1770000002
    assert models[2]["created"] == 1770000003


def test_sync_catalog_models_pricing_tracks_missing_remote_and_missing_pricing():
    catalog_document = {
        "version": 1,
        "models": [
            {
                "id": "model-a",
                "provider": "openai",
                "aliases": [],
                "costs": {"input_per_1k": 1.0, "output_per_1k": 1.0},
            },
            {
                "id": "model-b",
                "provider": "openai",
                "aliases": [],
                "costs": {"input_per_1k": 2.0, "output_per_1k": 2.0},
            },
        ],
    }
    openrouter_models = [
        {"id": "openai/model-a", "pricing": {"prompt": "0.000001"}},
    ]

    stats = sync_catalog_models_pricing(
        catalog_document=catalog_document,
        openrouter_models=openrouter_models,
    )

    assert stats.total_local_models == 2
    assert stats.updated == 0
    assert stats.unchanged == 0
    assert stats.missing_remote == 1
    assert stats.missing_pricing == 1


def test_sync_catalog_models_pricing_updates_created_when_pricing_missing():
    catalog_document = {
        "version": 1,
        "models": [
            {
                "id": "model-a",
                "provider": "openai",
                "aliases": [],
                "costs": {"input_per_1k": 1.0, "output_per_1k": 1.0},
            },
        ],
    }
    openrouter_models = [
        {"id": "openai/model-a", "pricing": {"prompt": "0.000001"}, "created": 1770000004},
    ]

    stats = sync_catalog_models_pricing(
        catalog_document=catalog_document,
        openrouter_models=openrouter_models,
    )

    assert stats.updated == 1
    assert stats.missing_pricing == 1
    assert catalog_document["models"][0]["created"] == 1770000004
