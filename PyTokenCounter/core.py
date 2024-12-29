# core.py

"""
PyTokenCounter Core Module
=============================

Provides functions to tokenize and count tokens in strings, files, and directories using specified models or encodings.
Includes utilities for managing model-encoding mappings and validating inputs.

Key Functions
-------------
- "GetModelMappings": Retrieve model to encoding mappings.
- "GetValidModels": List all valid model names.
- "GetValidEncodings": List all valid encoding names.
- "GetModelForEncoding": Get the model associated with a specific encoding.
- "GetEncodingForModel": Get the encoding associated with a specific model.
- "GetEncoding": Obtain the "tiktoken.Encoding" based on a model or encoding name.
- "TokenizeStr": Tokenize a single string into token IDs.
- "GetNumTokenStr": Count the number of tokens in a string.
- "TokenizeFile": Tokenize the contents of a file into token IDs.
- "GetNumTokenFile": Count the number of tokens in a file.
- "TokenizeFiles": Tokenize multiple files into token IDs.
- "GetNumTokenFiles": Count the number of tokens across multiple files.
- "TokenizeDir": Tokenize all files within a directory.
- "GetNumTokenDir": Count the number of tokens within a directory.

"""

from pathlib import Path

import tiktoken

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

VALID_MODELS_STR = "\n".join(VALID_MODELS)
VALID_ENCODINGS_STR = "\n".join(VALID_ENCODINGS)


def GetModelMappings() -> dict:
    """
    Get the mappings between models and their encodings.

    Returns
    -------
    dict
        A dictionary where keys are model names and values are their corresponding encodings.
    """

    return MODEL_MAPPINGS


def GetValidModels() -> list[str]:
    """
    Get a list of valid models.

    Returns
    -------
    list of str
        A list of valid model names.
    """

    return VALID_MODELS


def GetValidEncodings() -> list[str]:
    """
    Get a list of valid encodings.

    Returns
    -------
    list of str
        A list of valid encoding names.
    """

    return VALID_ENCODINGS


def GetModelForEncoding(encodingName: str) -> str:
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
    """

    if encodingName not in VALID_ENCODINGS:

        raise ValueError(
            f"Invalid encoding name: {encodingName}\n\nValid encoding names:\n{VALID_ENCODINGS_STR}"
        )

    else:

        for model, encoding in MODEL_MAPPINGS.items():

            if encoding == encodingName:

                return model


def GetEncodingForModel(modelName: str) -> str:
    """
    Get the encoding for a given model name.

    Parameters
    ----------
    modelName : str
        The name of the model.

    Returns
    -------
    str
        The encoding corresponding to the given model.

    Raises
    ------
    ValueError
        If the model name is not valid.
    """

    if modelName not in VALID_MODELS:

        raise ValueError(
            f"Invalid model: {modelName}\n\nValid models:\n{VALID_MODELS_STR}"
        )
    else:

        return MODEL_MAPPINGS[modelName]


def GetEncoding(
    model: str | None = None, encodingName: str | None = None
) -> tiktoken.Encoding:
    """
    Get the tiktoken Encoding based on the specified model or encoding name.

    Parameters
    ----------
    model : str | None, optional
        The name of the model to retrieve the encoding for. If provided,
        the encoding associated with the model will be used.
    encodingName : str | None, optional
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
            "Either model or encoding must be provided. Valid models:\n{VALID_MODELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
        )

    return tiktoken.get_encoding(encoding_name=_encodingName)


def TokenizeStr(
    string: str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> list[int]:
    """
    Tokenize a string into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    string : str
        The string to tokenize.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

    Returns
    -------
    list[int]
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
                "Either model, encoding name, or encoding must be provided. Valid models:\n{VALID_MODELS_STR}\n\nValid encodings:\n{VALID_ENCODINGS_STR}"
            )

    return _encoding.encode(text=string)


def GetNumTokenStr(
    string: str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> int:
    """
    Get the number of tokens in a string based on the specified model or encoding.

    Parameters
    ----------
    string : str
        The string to count tokens for.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

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

    tokens = TokenizeStr(
        string=string, model=model, encodingName=encodingName, encoding=encoding
    )

    return len(tokens)


def TokenizeFile(
    filePath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> list[int]:
    """
    Tokenize the contents of a file into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    filePath : Path | str
        The path to the file to tokenize.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

    Returns
    -------
    list[int]
        A list of token IDs representing the tokenized file contents.

    Raises
    ------
    TypeError
        If the types of "filePath", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    """

    if not isinstance(filePath, str) and not isinstance(filePath, Path):

        raise TypeError(
            f'Unexpected type for parameter "filePath". Expected type: str or pathlib.Path. Given type: {type(filePath)}'
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

    filePath = Path(filePath)

    fileContents = filePath.read_text()

    return TokenizeStr(
        string=fileContents, model=model, encodingName=encodingName, encoding=encoding
    )


def GetNumTokenFile(
    filePath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> int:
    """
    Get the number of tokens in a file based on the specified model or encoding.

    Parameters
    ----------
    filePath : Path | str
        The path to the file to count tokens for.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

    Returns
    -------
    int
        The number of tokens in the file.

    Raises
    ------
    TypeError
        If the types of "filePath", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    """

    if not isinstance(filePath, str) and not isinstance(filePath, Path):

        raise TypeError(
            f'Unexpected type for parameter "filePath". Expected type: str or pathlib.Path. Given type: {type(filePath)}'
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

    filePath = Path(filePath)

    return len(
        TokenizeFile(
            filePath=filePath, model=model, encodingName=encodingName, encoding=encoding
        )
    )


def TokenizeFiles(
    filePaths: list[Path] | list[str],
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> list[list[int]]:
    """
    Tokenize multiple files into lists of token IDs using the specified model or encoding.

    Parameters
    ----------
    filePaths : list[Path] | list[str]
        A list of paths to the files to tokenize.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

    Returns
    -------
    list[list[int]]
        A list where each element is a list of token IDs representing a tokenized file.

    Raises
    ------
    TypeError
        If the types of "filePaths", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    """

    if not isinstance(filePaths, list):

        raise TypeError(
            f'Unexpected type for parameter "filePaths". Expected type: list. Given type: {type(filePaths)}'
        )

    else:

        if not all(
            isinstance(filePath, str) or isinstance(filePath, Path)
            for filePath in filePaths
        ):

            listTypes = set([type(path) for path in filePaths])

            if len(listTypes) > 1:

                raise TypeError(
                    f'Unexpected type for parameter "filePaths". Expected type: list of str or pathlib.Path. Given list contains types: {listTypes}'
                )

            else:

                raise TypeError(
                    f'Unexpected type for parameter "filePaths". Expected type: list of str or pathlib.Path. Given list contains type: {listTypes}'
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

    tokenizedFiles = []

    for filePath in filePaths:

        tokenizedFiles.append(
            TokenizeFile(
                filePath=filePath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
            )
        )

    return tokenizedFiles


def GetNumTokenFiles(
    filePaths: list[Path] | list[str],
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
) -> int:
    """
    Get the number of tokens in multiple files based on the specified model or encoding.

    Parameters
    ----------
    filePaths : list[Path] | list[str]
        A list of paths to the files to count tokens for.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.

    Returns
    -------
    int
        The total number of tokens across all files.

    Raises
    ------
    TypeError
        If the types of "filePaths", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    """

    if not isinstance(filePaths, list):

        raise TypeError(
            f'Unexpected type for parameter "filePaths". Expected type: list. Given type: {type(filePaths)}'
        )

    else:

        if not all(
            isinstance(filePath, str) or isinstance(filePath, Path)
            for filePath in filePaths
        ):

            listTypes = set([type(path) for path in filePaths])

            if len(listTypes) > 1:

                raise TypeError(
                    f'Unexpected type for parameter "filePaths". Expected type: list of str or pathlib.Path. Given list contains types: {listTypes}'
                )

            else:

                raise TypeError(
                    f'Unexpected type for parameter "filePaths". Expected type: list of str or pathlib.Path. Given list contains type: {listTypes}'
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

    runningTokenCount = 0

    for filePath in filePaths:

        runningTokenCount += GetNumTokenFile(
            filePath=filePath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
        )

    return runningTokenCount


def TokenizeDir(
    dirPath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
) -> list[int | list] | list[int]:
    """
    Tokenize all files in a directory into lists of token IDs using the specified model or encoding.

    Parameters
    ----------
    dirPath : Path | str
        The path to the directory to tokenize.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        Whether to tokenize files in subdirectories recursively.

    Returns
    -------
    list[int | list] | list[int]
        A nested list where each element is either a list of token IDs representing
        a tokenized file or a sublist for a subdirectory.

    Raises
    ------
    TypeError
        If the types of "dirPath", "model", "encodingName", "encoding", or "recursive" are incorrect.
    ValueError
        If the provided "dirPath" is not a directory.
    RuntimeError
        If an unexpected error occurs during tokenization.
    """

    if not isinstance(dirPath, str) and not isinstance(dirPath, Path):

        raise TypeError(
            f'Unexpected type for parameter "dirPath". Expected type: str or pathlib.Path. Given type: {type(dirPath)}'
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

    if not isinstance(recursive, bool):

        raise TypeError(
            f'Unexpected type for parameter "recursive". Expected type: bool. Given type: {type(recursive)}'
        )

    dirPath = Path(dirPath)

    givenDirPath = dirPath

    dirPath = dirPath.resolve()

    if not dirPath.is_dir():

        raise ValueError(f'Given directory path "{givenDirPath}" is not a directory.')

    if recursive:

        tokenizedDir = []

        subDirPaths = []
        dirFilePaths = []

        for entry in dirPath.iterdir():

            if entry.is_dir():

                subDirPaths.append(entry)

            else:

                dirFilePaths.append(entry)

        tokenizedDir.append(
            TokenizeFiles(
                filePaths=dirFilePaths,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
            )
        )

        for subDirPath in subDirPaths:

            tokenizedDir.append(
                TokenizeDir(
                    dirPath=subDirPath,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    recursive=recursive,
                )
            )

        return tokenizedDir

    else:

        dirFilePaths = []

        for entry in dirPath.iterdir():

            if entry.is_dir():

                continue

            else:

                dirFilePaths.append(entry)

        return TokenizeFiles(
            filePaths=dirFilePaths,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
        )


def GetNumTokenDir(
    dirPath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
) -> int:
    """
    Get the number of tokens in all files within a directory based on the specified model or encoding.

    Parameters
    ----------
    dirPath : Path | str
        The path to the directory to count tokens for.
    model : str | None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str | None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding | None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        Whether to count tokens in files in subdirectories recursively.

    Returns
    -------
    int
        The total number of tokens across all files in the directory.

    Raises
    ------
    TypeError
        If the types of "dirPath", "model", "encodingName", "encoding", or "recursive" are incorrect.
    ValueError
        If the provided "dirPath" is not a directory.
    RuntimeError
        If an unexpected error occurs during token counting.
    """

    def FlattenTokenizedDir(tokenizedDir: list[int | list]) -> list[list[int]]:
        """
        Flatten a nested list of token counts.

        Parameters
        ----------
        tokenizedDir : list[int | list]
            The nested list of token counts.

        Returns
        -------
        list[int]
            A flattened list of token counts.
        """

        flattened = []

        for item in tokenizedDir:

            if isinstance(item, list):

                if all([isinstance(elem, int) for elem in item]):

                    flattened.append(item)

                else:

                    flattened.extend(FlattenTokenizedDir(item))

            else:

                flattened.append(item)

        return flattened

    if not isinstance(dirPath, str) and not isinstance(dirPath, Path):

        raise TypeError(
            f'Unexpected type for parameter "dirPath". Expected type: str or pathlib.Path. Given type: {type(dirPath)}'
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

    if not isinstance(recursive, bool):

        raise TypeError(
            f'Unexpected type for parameter "recursive". Expected type: bool. Given type: {type(recursive)}'
        )

    dirPath = Path(dirPath)

    givenDirPath = dirPath

    dirPath = dirPath.resolve()

    if not dirPath.is_dir():

        raise ValueError(f'Given directory path "{givenDirPath}" is not a directory.')

    tokenizedDir = TokenizeDir(
        dirPath=dirPath,
        model=model,
        encodingName=encodingName,
        encoding=encoding,
        recursive=recursive,
    )

    if recursive:

        flatTokenizedDir = FlattenTokenizedDir(tokenizedDir=tokenizedDir)

        runningTokenCount = 0

        for tokenizedFile in flatTokenizedDir:

            runningTokenCount += len(tokenizedFile)

        return runningTokenCount

    else:

        if not all([isinstance(item, int) for item in tokenizedDir]):

            raise RuntimeError(
                f"Unexpected error. Non-recursive tokenizedDir contains non-integer items. Contained types: {set([type(item) for item in tokenizedDir])}"
            )

        else:

            runningTokenCount = 0

            for tokenizedFile in tokenizedDir:

                runningTokenCount += len(tokenizedFile)

            return runningTokenCount
