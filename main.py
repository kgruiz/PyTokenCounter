from pathlib import Path

import tiktoken
from tqdm import tqdm

MODEL_MAPPINGS = {
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


VALID_MODELS = [
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

VALID_ENCODINGS = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base"]

VALID_MODLELS_STR = "\n".join(VALID_MODELS)
VALID_ENCODINGS_STR = "\n".join(VALID_ENCODINGS)


def GetTokenStr(
    string: str, model: str | None = None, encoding: str | None = None
) -> int:

    if model is not None:

        if model not in VALID_MODELS:

            raise ValueError(
                f"Invalid model: {model}\n\nValid models:\n{VALID_MODLELS_STR}"
            )

        encodingName = tiktoken.encoding_name_for_model(model_name=model)

    if encoding is not None:

        if encoding not in VALID_ENCODINGS:

            raise ValueError(
                f"Invalid encoding: {encoding}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
            )

        if model is not None and encodingName != encoding:

            if model not in VALID_MODELS:

                raise ValueError(
                    f"Invalid model: {model}\n\nValid models:\n{VALID_MODLELS_STR}"
                )

            else:

                raise ValueError(
                    f'Model {model} does not have encoding {encoding}\n\nValid encoding for model {model}: "{MODEL_MAPPINGS[model]}"'
                )

        else:

            encodingName = encoding

    if model is None and encoding is None:

        raise ValueError(
            "Either model or encoding must be provided. Valid models:\n{VALID_MODLELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
        )
