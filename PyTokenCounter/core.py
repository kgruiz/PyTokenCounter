# PyTokenCounter/core.py

"""
PyTokenCounter Core Module
===========================

Provides functions to tokenize and count tokens in strings, files, and directories
using specified models or encodings. Includes utilities for managing model-encoding
mappings and validating inputs.

Key Functions
-------------
- ``GetModelMappings`` : Retrieve model to encoding mappings.
- ``GetValidModels`` : List all valid model names.
- ``GetValidEncodings`` : List all valid encoding names.
- ``GetModelForEncodingName`` : Get the model associated with a specific encoding.
- ``GetEncodingNameForModel`` : Get the encoding associated with a specific model.
- ``GetEncoding`` : Obtain the ``tiktoken.Encoding`` based on a model or encoding name.
- ``MapTokens`` : Maps tokens to their corresponding decoded strings based on a specified encoding.
- ``TokenizeStr`` : Tokenize a single string into token IDs.
- ``GetNumTokenStr`` : Count the number of tokens in a string.
- ``TokenizeFile`` : Tokenize the contents of a file into token IDs.
- ``GetNumTokenFile`` : Count the number of tokens in a file.
- ``TokenizeFiles`` : Tokenize multiple files or a directory into token IDs.
- ``GetNumTokenFiles`` : Count the number of tokens across multiple files or in a directory.
- ``TokenizeDir`` : Tokenize all files within a directory.
- ``GetNumTokenDir`` : Count the number of tokens within a directory.
"""

from collections import OrderedDict

import tiktoken

from .progress import _InitializeTask, _tasks, _UpdateTask

MODEL_MAPPINGS = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    "gpt2": "gpt2",
    "gpt-2": "gpt2",
}


VALID_MODELS = [
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5",
    "gpt-35-turbo",
    "davinci-002",
    "babbage-002",
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "davinci",
    "curie",
    "babbage",
    "ada",
    "code-davinci-002",
    "code-davinci-001",
    "code-cushman-002",
    "code-cushman-001",
    "davinci-codex",
    "cushman-codex",
    "text-davinci-edit-001",
    "code-davinci-edit-001",
    "text-similarity-davinci-001",
    "text-similarity-curie-001",
    "text-similarity-babbage-001",
    "text-similarity-ada-001",
    "text-search-davinci-doc-001",
    "text-search-curie-doc-001",
    "text-search-babbage-doc-001",
    "text-search-ada-doc-001",
    "code-search-babbage-code-001",
    "code-search-ada-code-001",
    "gpt2",
    "gpt-2",
]

VALID_ENCODINGS = [
    "o200k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "cl100k_base",
    "p50k_base",
    "p50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "p50k_base",
    "p50k_base",
    "p50k_base",
    "p50k_base",
    "p50k_base",
    "p50k_base",
    "p50k_edit",
    "p50k_edit",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "r50k_base",
    "gpt2",
    "gpt2",
]

VALID_MODELS_STR = "\n".join(VALID_MODELS)
VALID_ENCODINGS_STR = "\n".join(VALID_ENCODINGS)

BINARY_EXTENSIONS = {
    # Image formats
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".pbm",
    ".webp",
    ".avif",
    ".tiff",
    ".tif",
    ".ico",
    ".svgz",  # Compressed SVG
    # Video formats
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".3g2",
    # Audio formats
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".m4a",
    ".wma",
    ".aiff",
    ".ape",
    ".opus",
    # Compressed archives
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".lz",
    ".zst",  # Zstandard compression
    ".cab",
    ".deb",
    ".rpm",
    ".pkg",
    # Disk images
    ".iso",
    ".dmg",
    ".img",
    ".vhd",
    ".vmdk",
    # Executables and libraries
    ".exe",
    ".msi",
    ".bat",  # Batch files may be readable but executed directly
    ".dll",
    ".so",
    ".bin",
    ".o",  # Compiled object files
    ".a",  # Static libraries
    ".dylib",  # macOS dynamic library
    # Fonts
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
    # Documents
    ".pdf",
    ".ps",  # PostScript
    ".eps",  # Encapsulated PostScript
    # Design and graphics
    ".psd",
    ".ai",
    ".indd",
    ".sketch",
    # 3D and CAD files
    ".blend",
    ".stl",
    ".step",
    ".iges",
    ".fbx",
    ".glb",
    ".gltf",
    ".3ds",
    ".obj",
    ".cad",
    # Virtual machines and firmware
    ".qcow2",
    ".vdi",
    ".vhdx",
    ".rom",
    ".bin",  # Generic binary firmware
    ".img",
    # Miscellaneous binary formats
    ".dat",
    ".pak",  # Game resource package files
    ".sav",  # Save game files
    ".nes",  # ROM file for NES emulator
    ".gba",  # Game Boy Advance ROM
    ".nds",  # Nintendo DS ROM
    ".iso",  # CD/DVD disk image
    ".jar",  # Java Archive (binary format)
    ".class",  # Compiled Java class file
    ".wasm",  # WebAssembly binary format
}


def GetModelMappings() -> OrderedDict:
    """
    Get the mappings between models and their encodings.

    Returns
    -------
    OrderedDict
        A OrderedDictionary where keys are model names and values are their corresponding encodings.

    Examples
    --------
            return modelMatches[0]

        else:

            return sorted(modelMatches)

    """

    return OrderedDict(MODEL_MAPPINGS)


def GetModelForEncodingName(encodingName: str) -> list[str] | str:
    """
    Get the model name for a given encoding.

    Parameters
    ----------
    encodingName : str
        The name of the encoding.

    Returns
    -------
    str
        The model name corresponding to the given encoding.

    Raises
    ------
    ValueError
        If the encoding name is not valid.

    Examples
    --------
    >>> from PyTokenCounter import GetModelForEncodingName
    >>> model = GetModelForEncodingName('cl100k_base')
    >>> print(model)
    ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'text-embedding-3-large', 'text-embedding-3-small', 'text-embedding-ada-002']
    """

    if encodingName not in VALID_ENCODINGS:

        raise ValueError(
            f"Invalid encoding name: {encodingName}\n\nValid encoding names:\n{VALID_ENCODINGS_STR}"
        )

    else:

        modelMatches = []

        for model, encoding in MODEL_MAPPINGS.items():

            if encoding == encodingName:

                modelMatches.append(model)

        if len(modelMatches) == 1:

            return modelMatches[0]

        else:

            return sorted(modelMatches)


def GetModelForEncoding(encodingName: str) -> list[str] | str:
    """
    Alias of GetModelForEncodingName for backward compatibility.
    """
    return GetModelForEncodingName(encodingName)


def GetEncodingForModel(modelName: str, quiet: bool = False) -> tiktoken.Encoding:
    """
    Get the encoding for a given model name.

    Parameters
    ----------
    modelName : str
        The name of the model.
    quiet : bool, optional
        If True, suppress progress updates (default is False).

    Returns
    -------
    str
        The encoding corresponding to the given model.

    Raises
    ------
    ValueError
        If the model name is not valid.

    Examples
    --------
    >>> from PyTokenCounter import GetEncodingNameForModel
    >>> encoding = GetEncodingNameForModel('gpt-3.5-turbo')
    >>> print(encoding)
    'cl100k_base'
    """

    if modelName not in VALID_MODELS:

        raise ValueError(
            f"Invalid model: {modelName}\n\nValid models:\n{VALID_MODELS_STR}"
        )

    else:

        encodingName = MODEL_MAPPINGS[modelName]

        return tiktoken.get_encoding(encoding_name=encodingName)


def GetEncodingNameForModel(modelName: str, quiet: bool = False) -> str:
    """
    Get the encoding for a given model name.

    Parameters
    ----------
    modelName : str
        The name of the model.
    quiet : bool, optional
        If True, suppress progress updates (default is False).

    Returns
    -------
    str
        The encoding corresponding to the given model.

    Raises
    ------
    ValueError
        If the model name is not valid.

    Examples
    --------
    >>> from PyTokenCounter import GetEncodingNameForModel
    >>> encoding = GetEncodingNameForModel('gpt-3.5-turbo')
    >>> print(encoding)
    'cl100k_base'
    """

    if modelName not in VALID_MODELS:

        raise ValueError(
            f"Invalid model: {modelName}\n\nValid models:\n{VALID_MODELS_STR}"
        )

    else:

        return MODEL_MAPPINGS[modelName]


def GetValidModels() -> list[str]:
    """
    List all valid model names.
    """
    return list(VALID_MODELS)


def GetValidEncodings() -> list[str]:
    """
    List all unique valid encoding names.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for enc in VALID_ENCODINGS:
        if enc not in seen:
            seen.add(enc)
            unique.append(enc)
    return unique


def GetEncoding(
    model: str | None = None,
    encodingName: str | None = None,
) -> tiktoken.Encoding:
    """
    Get the tiktoken Encoding based on the specified model or encoding name.

    Parameters
    ----------
    model : str or None, optional
        The name of the model to retrieve the encoding for. If provided,
        the encoding associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.

    Returns
    -------
    tiktoken.Encoding
        The encoding corresponding to the specified model or encoding name.

    Raises
    ------
    TypeError
        If the type of "model" or "encodingName" is not a string.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there
        is a mismatch between the model and encoding name.

    Examples
    --------
    >>> from PyTokenCounter import GetEncoding
    >>> encoding = GetEncoding(model='gpt-3.5-turbo')
    >>> print(encoding)
    <Encoding cl100k_base>
    >>> encoding = GetEncoding(encodingName='p50k_base')
    >>> print(encoding)
    <Encoding p50k_base>
    """

    if model is not None and not isinstance(model, str):

        raise TypeError(
            f'Unexpected type for parameter "model". Expected type: str. Given type: {type(model)}'
        )

    if encodingName is not None and not isinstance(encodingName, str):

        raise TypeError(
            f'Unexpected type for parameter "encodingName". Expected type: str. Given type: {type(encodingName)}'
        )

    _encodingName = None

    if model is not None:

        if model not in VALID_MODELS:

            raise ValueError(
                f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
            )

        else:

            _encodingName = tiktoken.encoding_name_for_model(model_name=model)

    if encodingName is not None:

        if encodingName not in VALID_ENCODINGS:

            raise ValueError(
                f"Invalid encoding name: {encodingName}\n\nValid encoding names:\n{VALID_ENCODINGS_STR}"
            )

        if model is not None and _encodingName != encodingName:

            if model not in VALID_MODELS:

                raise ValueError(
                    f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
                )

            else:

                raise ValueError(
                    f'Model {model} does not have encoding name {encodingName}\n\nValid encoding names for model {model}: "{MODEL_MAPPINGS[model]}"'
                )

        else:

            _encodingName = encodingName

    if _encodingName is None:

        raise ValueError(
            "Either model or encoding must be provided. Valid models:\n"
            f"{VALID_MODELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
        )

    return tiktoken.get_encoding(encoding_name=_encodingName)


def MapTokens(
    tokens: list[int] | OrderedDict[str, list[int] | OrderedDict],
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> OrderedDict[str, int] | OrderedDict[str, OrderedDict[str, int] | OrderedDict]:
    """
    Maps tokens to their corresponding decoded strings based on a specified encoding.

    Parameters
    ----------
    tokens : list[int] or OrderedDict[str, list[int] or OrderedDict]
        The tokens to be mapped. This can either be:
        - A list of integer tokens to decode.
        - An `OrderedDict` with string keys and values that are either:
          - A list of integer tokens.
          - Another nested `OrderedDict` with the same structure.

    model : str or None, optional, default="gpt-4o"
        The model name to use for determining the encoding. If provided, the model
        must be valid and compatible with the specified encoding or encoding name

    encodingName : str or None, optional
        The name of the encoding to use. Must be compatible with the provided model








        if both are specified.









    encoding : tiktoken.Encoding or None, optional
        The encoding object to use. Must match the specified model and/or encoding name








        if they are provided.









    Returns
    -------
    OrderedDict[str, int] or OrderedDict[str, OrderedDict[str, int] or OrderedDict]
        A mapping of decoded strings to their corresponding integer tokens.
        If `tokens` is a nested structure, the result will maintain the same nested
        structure with decoded mappings.

    Raises
    ------
    TypeError
        - If `model` is not a string.
        - If `encodingName` is not a string.
        - If `encoding` is not a `tiktoken.Encoding` instance.
        - If `tokens` contains invalid types (e.g., non-integer tokens in a list or non-string keys in a dictionary).

    ValueError
        - If an invalid model or encoding name is provided.
        - If the encoding does not match the model or encoding name.

    KeyError
        - If a token is not in the given encoding's vocabulary.

    RuntimeError
        - If an unexpected error occurs while validating the encoding.

    Notes
    -----
    - Either `model`, `encodingName`, or `encoding` must be provided.
    - The function validates compatibility between the provided model, encoding name, and encoding object.
    - Nested dictionaries are processed recursively to preserve structure.
    """

    if model is not None and not isinstance(model, str):

        raise TypeError(
            f'Unexpected type for parameter "model". Expected type: str. Given type: {type(model)}'
        )

    if encodingName is not None and not isinstance(encodingName, str):

        raise TypeError(
            f'Unexpected type for parameter "encodingName". Expected type: str. Given type: {type(encodingName)}'
        )

    if encoding is not None and not isinstance(encoding, tiktoken.Encoding):

        raise TypeError(
            f'Unexpected type for parameter "encoding". Expected type: tiktoken.Encoding. Given type: {type(encoding)}'
        )

    _encodingName = None

    if model is not None:

        if model not in VALID_MODELS:

            raise ValueError(
                f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
            )

        else:

            _encodingName = tiktoken.encoding_name_for_model(model_name=model)

    if encodingName is not None:

        if encodingName not in VALID_ENCODINGS:

            raise ValueError(
                f"Invalid encoding name: {encodingName}\n\nValid encoding names:\n{VALID_ENCODINGS_STR}"
            )

        if model is not None and _encodingName != encodingName:

            if model not in VALID_MODELS:

                raise ValueError(
                    f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
                )

            else:

                raise ValueError(
                    f'Model {model} does not have encoding name {encodingName}\n\nValid encoding names for model {model}: "{MODEL_MAPPINGS[model]}"'
                )

        else:

            _encodingName = encodingName

    _encoding = None

    if _encodingName is not None:

        _encoding = tiktoken.get_encoding(encoding_name=_encodingName)

    if encoding is not None:

        if _encodingName is not None and _encoding != encoding:

            if encodingName is not None and model is not None:

                raise ValueError(
                    f"Model {model} does not have encoding {encoding}.\n\nValid encoding name for model {model}: \n{_encodingName}\n"
                )

            elif encodingName is not None:

                raise ValueError(
                    f'Encoding name {encodingName} does not match provided encoding "{encoding}"'
                )

            elif model is not None:

                raise ValueError(
                    f'Model {model} does not have provided encoding "{encoding}".\n\nValid encoding name for model {model}: \n{_encodingName}\n'
                )

            else:

                raise RuntimeError(
                    f'Unexpected error. Given model "{model}" and encoding name "{encodingName}" resulted in encoding "{_encoding}".\nFor unknown reasons, this encoding doesn\'t match given encoding "{encoding}".\nPlease report this error.'
                )

        else:

            _encoding = encoding

        if _encodingName is None and _encoding is None:

            raise ValueError(
                "Either model, encoding name, or encoding must be provided. Valid models:\n"
                f"{VALID_MODELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
            )

    if isinstance(tokens, list):

        mappedTokens = OrderedDict()

        nonInts = [token for token in tokens if not isinstance(token, int)]

        if len(nonInts) > 0:

            raise TypeError(
                f"Tokens must be integers. Found non-integer tokens: {nonInts}"
            )

        for token in tokens:

            decoded = _encoding.decode([token])

            mappedTokens[decoded] = token

        return mappedTokens

    elif isinstance(tokens, dict):

        mappedTokens = OrderedDict()

        nonStrNames = [entry for entry in tokens.keys() if not isinstance(entry, str)]

        if len(nonStrNames) > 0:

            raise TypeError(
                f"Directory and file names must be strings. Found non-string names: {nonStrNames}"
            )

        for entryName, content in tokens.items():

            mappedTokens[entryName] = MapTokens(content, model, encodingName, encoding)

        return mappedTokens

    else:

        raise TypeError(
            f'Unexpected type for parameter "tokens". Expected type: list of int or OrderedDict. Given type: {type(tokens)}'
        )


def TokenizeStr(
    string: str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
    mapTokens: bool = True,
) -> list[int]:
    """
    Tokenize a string into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    string : str
        The string to tokenize.
    model : str or None, optional, default="gpt-4o"
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    quiet : bool, optional
        If True, suppress progress updates (default is False).

    Returns
    -------
    list of int
        A list of token IDs representing the tokenized string.

    Raises
    ------
    TypeError
        If the types of "string", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    RuntimeError
        If an unexpected error occurs during encoding.

    Examples
    --------
    >>> from PyTokenCounter import TokenizeStr
    >>> tokens = TokenizeStr(string="Hail to the Victors!", model="gpt-4o")
    >>> print(tokens)
    [39, 663, 316, 290, ..., 914, 0]
    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding("cl100k_base")
    >>> tokens = TokenizeStr(string="2024 National Champions", encoding=encoding)
    >>> print(tokens)
    [1323, 19, 6743, 40544]
    """

    if not isinstance(string, str):

        raise TypeError(
            f'Unexpected type for parameter "string". Expected type: str. Given type: {type(string)}'
        )

    if model is not None and not isinstance(model, str):

        raise TypeError(
            f'Unexpected type for parameter "model". Expected type: str. Given type: {type(model)}'
        )

    if encodingName is not None and not isinstance(encodingName, str):

        raise TypeError(
            f'Unexpected type for parameter "encodingName". Expected type: str. Given type: {type(encodingName)}'
        )

    if encoding is not None and not isinstance(encoding, tiktoken.Encoding):

        raise TypeError(
            f'Unexpected type for parameter "encoding". Expected type: tiktoken.Encoding. Given type: {type(encoding)}'
        )

    _encodingName = None

    if model is not None:

        if model not in VALID_MODELS:

            raise ValueError(
                f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
            )

        else:

            _encodingName = tiktoken.encoding_name_for_model(model_name=model)

    if encodingName is not None:

        if encodingName not in VALID_ENCODINGS:

            raise ValueError(
                f"Invalid encoding name: {encodingName}\n\nValid encoding names:\n{VALID_ENCODINGS_STR}"
            )

        if model is not None and _encodingName != encodingName:

            if model not in VALID_MODELS:

                raise ValueError(
                    f"Invalid model: {model}\n\nValid models:\n{VALID_MODELS_STR}"
                )

            else:

                raise ValueError(
                    f'Model {model} does not have encoding name {encodingName}\n\nValid encoding names for model {model}: "{MODEL_MAPPINGS[model]}"'
                )

        else:

            _encodingName = encodingName

    _encoding = None

    if _encodingName is not None:

        _encoding = tiktoken.get_encoding(encoding_name=_encodingName)

    if encoding is not None:

        if _encodingName is not None and _encoding != encoding:

            if encodingName is not None and model is not None:

                raise ValueError(
                    f"Model {model} does not have encoding {encoding}.\n\nValid encoding name for model {model}: \n{_encodingName}\n"
                )

            elif encodingName is not None:

                raise ValueError(
                    f'Encoding name {encodingName} does not match provided encoding "{encoding}"'
                )

            elif model is not None:

                raise ValueError(
                    f'Model {model} does not have provided encoding "{encoding}".\n\nValid encoding name for model {model}: \n{_encodingName}\n'
                )

            else:

                raise RuntimeError(
                    f'Unexpected error. Given model "{model}" and encoding name "{encodingName}" resulted in encoding "{_encoding}".\nFor unknown reasons, this encoding doesn\'t match given encoding "{encoding}".\nPlease report this error.'
                )

        else:

            _encoding = encoding

        if _encodingName is None and _encoding is None:

            raise ValueError(
                "Either model, encoding name, or encoding must be provided. Valid models:\n"
                f"{VALID_MODELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
            )

    hasBar = False
    taskName = None

    displayString = f"{string[:30]}..." if len(string) > 33 else string

    if len(_tasks) == 0 and not quiet:

        hasBar = True
        taskName = f'Tokenizing "{displayString}"'
        _InitializeTask(taskName=taskName, total=1, quiet=quiet)

    tokenizedStr = _encoding.encode(text=string)

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f'Done Tokenizing "{displayString}"',
            quiet=quiet,
        )

    if mapTokens:

        tokenizedStr = MapTokens(tokenizedStr, model, encodingName, encoding)

    return tokenizedStr


def GetNumTokenStr(
    string: str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
) -> int:
    """
    Get the number of tokens in a string based on the specified model or encoding.

    Parameters
    ----------
    string : str
        The string to count tokens for.
    model : str or None, optional, default="gpt-4o"
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    quiet : bool, optional
        If True, suppress progress updates (default is False).

    Returns
    -------
    int
        The number of tokens in the string.

    Raises
    ------
    TypeError
        If the types of "string", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.

    Examples
    --------
    >>> from PyTokenCounter import GetNumTokenStr
    >>> numTokens = GetNumTokenStr(string="Hail to the Victors!", model="gpt-4o")
    >>> print(numTokens)
    7
    >>> numTokens = GetNumTokenStr(string="2024 National Champions", model="gpt-4o")
    >>> print(numTokens)
    4
    >>> numTokens = GetNumTokenStr(string="Corum 4 Heisman", encoding=tiktoken.get_encoding("cl100k_base"))
    >>> print(numTokens)
    6
    """

    if not isinstance(string, str):

        raise TypeError(
            f'Unexpected type for parameter "string". Expected type: str. Given type: {type(string)}'
        )

    if model is not None and not isinstance(model, str):

        raise TypeError(
            f'Unexpected type for parameter "model". Expected type: str. Given type: {type(model)}'
        )

    if encodingName is not None and not isinstance(encodingName, str):

        raise TypeError(
            f'Unexpected type for parameter "encodingName". Expected type: str. Given type: {type(encodingName)}'
        )

    if encoding is not None and not isinstance(encoding, tiktoken.Encoding):

        raise TypeError(
            f'Unexpected type for parameter "encoding". Expected type: tiktoken.Encoding. Given type: {type(encoding)}'
        )

    hasBar = False
    taskName = None

    displayString = f"{string[:22]}..." if len(string) > 25 else string

    if len(_tasks) == 0 and not quiet:

        hasBar = True
        taskName = f'Counting Tokens in "{displayString}"'
        _InitializeTask(taskName=taskName, total=1, quiet=quiet)

    tokens = TokenizeStr(
        string=string,
        model=model,
        encodingName=encodingName,
        encoding=encoding,
        quiet=quiet,
    )

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f'Done Counting Tokens in "{displayString}"',
            quiet=quiet,
        )

    return len(tokens)
