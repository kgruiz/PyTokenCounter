import json
from pathlib import Path
import pytest
import tiktoken
import PyTokenCounter as tc

TEST_DIR = Path(__file__).resolve().parent.parent / "Tests"
ANSWERS_DIR = TEST_DIR / "Answers"


def test_get_model_mappings():
    expected = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "Codex models": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-003": "p50k_base",
        "GPT-3 models like davinci": "r50k_base",
    }
    assert tc.GetModelMappings() == expected


def test_get_valid_models():
    expected = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "Codex models",
        "text-davinci-002",
        "text-davinci-003",
        "GPT-3 models like davinci",
    ]
    assert set(tc.GetValidModels()) == set(expected)


def test_get_valid_encodings():
    expected = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base"]
    assert set(tc.GetValidEncodings()) == set(expected)


def test_get_model_for_encoding():
    mapping = {
        "o200k_base": ["gpt-4o", "gpt-4o-mini"],
        "cl100k_base": [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ],
        "p50k_base": ["Codex models", "text-davinci-002", "text-davinci-003"],
        "r50k_base": "GPT-3 models like davinci",
    }
    for name, expected in mapping.items():
        encoding = tiktoken.get_encoding(encoding_name=name)
        result = tc.GetModelForEncoding(encoding=encoding)
        if isinstance(expected, list):
            assert sorted(result) == sorted(expected)
        else:
            assert result == expected


def test_get_model_for_encoding_name():
    mapping = {
        "o200k_base": ["gpt-4o", "gpt-4o-mini"],
        "cl100k_base": [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ],
        "p50k_base": ["Codex models", "text-davinci-002", "text-davinci-003"],
        "r50k_base": "GPT-3 models like davinci",
    }
    for name, expected in mapping.items():
        result = tc.GetModelForEncodingName(encodingName=name)
        if isinstance(expected, list):
            assert sorted(result) == sorted(expected)
        else:
            assert result == expected


def test_get_encoding_for_model():
    mapping = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-003": "p50k_base",
    }
    for model, encoding_name in mapping.items():
        result = tc.GetEncodingForModel(modelName=model)
        assert result.name == encoding_name


def test_get_encoding_name_for_model():
    mapping = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "Codex models": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-003": "p50k_base",
        "GPT-3 models like davinci": "r50k_base",
    }
    for model, encoding_name in mapping.items():
        result = tc.GetEncodingNameForModel(modelName=model)
        assert result == encoding_name


def test_get_encoding():
    encoding = tc.GetEncoding(model="gpt-3.5-turbo")
    assert encoding.name == "cl100k_base"

    encoding = tc.GetEncoding(encodingName="p50k_base")
    assert encoding.name == "p50k_base"

    encoding = tc.GetEncoding(model="gpt-4-turbo", encodingName="cl100k_base")
    assert encoding.name == "cl100k_base"

    with pytest.raises(ValueError):
        tc.GetEncoding(model="gpt-3.5-turbo", encodingName="p50k_base")


def test_get_encoding_error():
    with pytest.raises(ValueError):
        tc.GetEncoding()

