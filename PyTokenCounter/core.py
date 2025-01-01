# core.py

"""
PyTokenCounter Core Module
===========================

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
- "TokenizeFiles": Tokenize multiple files or a directory into token IDs.
- "GetNumTokenFiles": Count the number of tokens across multiple files or in a directory.
- "TokenizeDir": Tokenize all files within a directory.
- "GetNumTokenDir": Count the number of tokens within a directory.

"""

from pathlib import Path

import tiktoken
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column

from ._utils import ReadTextFile, UnsupportedEncodingError

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


_progressInstance = Progress(
    TextColumn(
        "[bold blue]{task.description}",
        justify="left",
        table_column=Column(width=50),
    ),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(bar_width=None),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    expand=True,
)
_tasks = {}


def _InitializeTask(taskName: str, total: int, quiet: bool = False) -> int | None:
    """
    Internal function to initialize a task in the progress bar.

    Parameters
    ----------
    taskName : str
        The description of the task to display in the progress bar.
    total : int
        The total work required for the task.
    quiet : bool, optional
        If True, suppress the progress bar (default is False).

    Returns
    -------
    int or None
        Task ID for the initialized task, or None if quiet is True.
    """

    if quiet:

        return None

    if not _progressInstance.live.is_started:

        _progressInstance.start()

    if taskName in _tasks:

        return _tasks[taskName]

    taskId = _progressInstance.add_task(taskName, total=total)
    _tasks[taskName] = taskId

    return taskId


def _UpdateTask(
    taskName: str,
    advance: int,
    description: str = None,
    appendDescription: str = None,
    quiet: bool = False,
) -> None:
    """
    Internal function to update a task's progress and optionally its description.

    Parameters
    ----------
    taskName : str
        The name of the task to update.
    advance : int
        The amount of work to add to the task's progress.
    description : str, optional
        A new description for the task (default is None).
    appendDescription : str, optional
        Text to append to the current description of the task (default is None).
    quiet : bool, optional
        If True, suppress the progress bar update (default is False).

    Raises
    ------
    ValueError
        If the specified task name is not found.
    """

    if quiet:

        return

    if taskName not in _tasks:

        raise ValueError(f"Task '{taskName}' not found.")

    currentTask = _progressInstance.tasks[_tasks[taskName]]
    currentDescription = currentTask.description if currentTask.description else ""

    if appendDescription is not None:

        description = f"{currentDescription} {appendDescription}".strip()

    elif description is None:

        description = currentDescription

    _progressInstance.update(_tasks[taskName], advance=advance, description=description)

    if all(task.finished for task in _progressInstance.tasks):

        _progressInstance.stop()
        _tasks.clear()


def _CountDirFiles(dirPath: Path, recursive: bool = True) -> int:
    """
    Count the number of files in a directory.

    This function traverses the specified directory and counts the number of files it contains.
    It can operate recursively to include files in subdirectories if desired.

    Parameters
    ----------
    dirPath : Path
        The path to the directory in which to count files.
    recursive : bool, optional
        Whether to count files in subdirectories recursively (default is True).

    Returns
    -------
    int
        The total number of files in the directory.

    Raises
    ------
    ValueError
        If the provided `dirPath` is not a directory.
    """

    if not dirPath.is_dir():

        raise ValueError(f"Given path '{dirPath}' is not a directory.")

    numFiles = 0

    if recursive:

        for entry in dirPath.iterdir():

            if entry.is_dir():

                numFiles += _CountDirFiles(entry, recursive=recursive)

            else:

                numFiles += 1

    else:

        numFiles = sum(1 for entry in dirPath.iterdir() if entry.is_file())

    return numFiles


def GetModelMappings() -> dict:
    """
    Get the mappings between models and their encodings.

    Returns
    -------
    dict
        A dictionary where keys are model names and values are their corresponding encodings.

    Examples
    --------
    >>> from core import GetModelMappings
    >>> mappings = GetModelMappings()
    >>> print(mappings)
    {'gpt-4': 'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base', ...}
    """

    return MODEL_MAPPINGS


def GetValidModels() -> list[str]:
    """
    Get a list of valid models.

    Returns
    -------
    list of str
        A list of valid model names.

    Examples
    --------
    >>> from core import GetValidModels
    >>> models = GetValidModels()
    >>> print(models)
    ['gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002', ...]
    """

    return VALID_MODELS


def GetValidEncodings() -> list[str]:
    """
    Get a list of valid encodings.

    Returns
    -------
    list of str
        A list of valid encoding names.

    Examples
    --------
    >>> from core import GetValidEncodings
    >>> encodings = GetValidEncodings()
    >>> print(encodings)
    ['cl100k_base', 'p50k_base', 'r50k_base', ...]
    """

    return VALID_ENCODINGS


def GetModelForEncoding(encodingName: str) -> list[str] | str:
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
    >>> from core import GetModelForEncoding
    >>> model = GetModelForEncoding('cl100k_base')
    >>> print(model)
    'gpt-4'
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


def GetEncodingForModel(modelName: str, quiet: bool = False) -> str:
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
    >>> from core import GetEncodingForModel
    >>> encoding = GetEncodingForModel('gpt-3.5-turbo')
    >>> print(encoding)
    'cl100k_base'
    """

    if modelName not in VALID_MODELS:

        raise ValueError(
            f"Invalid model: {modelName}\n\nValid models:\n{VALID_MODELS_STR}"
        )

    else:

        return MODEL_MAPPINGS[modelName]


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
    >>> from core import GetEncoding
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


def TokenizeStr(
    string: str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
) -> list[int]:
    """
    Tokenize a string into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    string : str
        The string to tokenize.
    model : str or None, optional
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
    >>> from core import TokenizeStr
    >>> tokens = TokenizeStr("Hello, world!", model="gpt-3.5-turbo")
    >>> print(tokens)
    [15496, 11, 995]

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding("p50k_base")
    >>> tokens = TokenizeStr("Sample text for tokenization.", encoding=encoding)
    >>> print(tokens)
    [1234, 5678, 91011]
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

    return tokenizedStr


def GetNumTokenStr(
    string: str,
    model: str | None = None,
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
    model : str or None, optional
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
    >>> from core import GetNumTokenStr
    >>> num_tokens = GetNumTokenStr("Hello, world!", model="gpt-4")
    >>> print(num_tokens)
    3

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding("p50k_base")
    >>> num_tokens = GetNumTokenStr("Another example string.", encoding=encoding)
    >>> print(num_tokens)
    5
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


def TokenizeFile(
    filePath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
) -> list[int]:
    """
    Tokenize the contents of a file into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    filePath : Path or str
        The path to the file to tokenize.
    model : str or None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used. Default is None.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model. Default is None.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName. Default is None.
    quiet : bool, optional
        If True, suppress progress updates. Default is False.

    Returns
    -------
    list of int
        A list of token IDs representing the tokenized file contents.

    Raises
    ------
    TypeError
        If the types of `filePath`, `model`, `encodingName`, or `encoding` are incorrect.
    ValueError
        If the provided `model` or `encodingName` is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    UnsupportedEncodingError
        If the file's encoding is not supported (i.e., not UTF-8, ASCII, or another
        text encoding format supported by the chardet package).
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    Tokenizing a file with a specified model:

    >>> tokens = TokenizeFile('example.txt', model='gpt-3.5-turbo')
    >>> print(tokens)
    [15496, 11, 995, ...]

    Tokenizing a file with an existing encoding object:

    >>> from tiktoken import Encoding
    >>> encoding = Encoding.load('gpt-3.5-turbo')
    >>> tokens = TokenizeFile('example.txt', encoding=encoding)
    >>> print(tokens)
    [1234, 5678, 91011, ...]

    Handling a non-existent file:

    >>> TokenizeFile('non_existent.txt')
    Traceback (most recent call last):
        ...
    FileNotFoundError: File not found: /absolute/path/to/non_existent.txt

    Using a mismatched encoding and model:

    >>> TokenizeFile('example.txt', model='gpt-3.5-turbo', encodingName='wrong-encoding')
    Traceback (most recent call last):
        ...
    ValueError: Encoding name "wrong-encoding" does not match the encoding associated with model "gpt-3.5-turbo".
    """

    if not isinstance(filePath, (str, Path)):

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

    fileContents = ReadTextFile(filePath=filePath)

    if not isinstance(fileContents, str):

        raise UnsupportedEncodingError(encoding=fileContents[1], filePath=filePath)

    hasBar = False
    taskName = None

    if len(_tasks) == 0 and not quiet:

        hasBar = True
        taskName = f"Tokenizing {filePath.name}"
        _InitializeTask(taskName=taskName, total=1, quiet=quiet)

    tokens = TokenizeStr(
        string=fileContents,
        model=model,
        encodingName=encodingName,
        encoding=encoding,
        quiet=quiet,
    )

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f"Done Tokenizing {filePath.name}",
            quiet=quiet,
        )

    return tokens


def GetNumTokenFile(
    filePath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
) -> int:
    """
    Get the number of tokens in a file based on the specified model or encoding.

    Parameters
    ----------
    filePath : Path or str
        The path to the file to count tokens for.
    model : str or None, optional
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
        The number of tokens in the file.

    Raises
    ------
    TypeError
        If the types of "filePath", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid, or if there is a
        mismatch between the model and encoding name, or between the provided
        encoding and the derived encoding.
    UnsupportedEncodingError
        If the file's encoding is not supported (i.e., not UTF-8, ASCII, or another
        text encoding format supported by the chardet package).
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> from core import GetNumTokenFile
    >>> num_tokens = GetNumTokenFile('example.txt', model='gpt-4')
    >>> print(num_tokens)
    150

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding('p50k_base')
    >>> num_tokens = GetNumTokenFile('sample.txt', encoding=encoding)
    >>> print(num_tokens)
    200

    >>> # Handling a non-existent file
    >>> GetNumTokenFile('non_existent.txt')
    Traceback (most recent call last):
        ...
    FileNotFoundError: File not found: /absolute/path/to/non_existent.txt

    >>> # Using a mismatched encoding and model
    >>> GetNumTokenFile('example.txt', model='gpt-3.5-turbo', encodingName='wrong-encoding')
    Traceback (most recent call last):
        ...
    ValueError: Encoding name "wrong-encoding" does not match the encoding associated with model "gpt-3.5-turbo".
    """

    if not isinstance(filePath, (str, Path)):

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

    hasBar = False
    taskName = None

    if len(_tasks) == 0 and not quiet:

        hasBar = True
        taskName = f"Counting Tokens in {filePath.name}"
        _InitializeTask(taskName=taskName, total=1, quiet=quiet)

    numTokens = len(
        TokenizeFile(
            filePath=filePath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            quiet=quiet,
        )
    )

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f"Done Counting Tokens in {filePath.name}",
            quiet=quiet,
        )

    return numTokens


def TokenizeDir(
    dirPath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
) -> dict[str, list[int] | dict]:
    """
    Tokenize all files in a directory into lists of token IDs using the specified model or encoding.

    Parameters
    ----------
    dirPath : Path or str
        The path to the directory to tokenize.
    model : str or None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        Whether to tokenize files in subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.

    Returns
    -------
    dict[str, list[int] | dict]
        A nested dictionary where each key is a file or subdirectory name:
        - If the key is a file, its value is a list of token IDs.
        - If the key is a subdirectory, its value is another dictionary following the same structure.

    Raises
    ------
    TypeError
        If the types of "dirPath", "model", "encodingName", "encoding", or "recursive" are incorrect.
    ValueError
        If the provided "dirPath" is not a directory.
    RuntimeError
        If an unexpected error occurs during tokenization.

    Examples
    --------
    >>> from core import TokenizeDir
    >>> tokenized_dir = TokenizeDir('/path/to/directory', model='gpt-4')
    >>> print(tokenized_dir)
    {
        'file1.txt': [15496, 11, 995],
        'file2.txt': [1234, 5678, 91011],
        'subdir': {
            'file3.txt': [2345, 6789]
        }
    }
    """

    if not isinstance(dirPath, (str, Path)):

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

    dirPath = Path(dirPath).resolve()

    if not dirPath.is_dir():

        raise ValueError(f'Given directory path "{dirPath}" is not a directory.')

    numFiles = _CountDirFiles(dirPath=dirPath, recursive=recursive)

    if not quiet:

        taskName = "Tokenizing Directory"
        _InitializeTask(taskName=taskName, total=numFiles, quiet=quiet)

    else:

        taskName = None

    tokenizedDir: dict[str, list[int] | dict] = {}
    subDirPaths: list[Path] = []

    for entry in dirPath.iterdir():

        if entry.is_dir():

            subDirPaths.append(entry)

        else:

            if not quiet:

                _UpdateTask(
                    taskName=taskName,
                    advance=0,
                    description=f"Tokenizing {entry.relative_to(dirPath)}",
                    quiet=quiet,
                )

            try:

                tokenizedFile = TokenizeFile(
                    filePath=entry,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    quiet=quiet,
                )
                tokenizedDir[entry.name] = tokenizedFile

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Done Tokenizing {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

            except UnsupportedEncodingError as e:

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Skipping {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

                continue

    if recursive:

        for subDirPath in subDirPaths:

            tokenizedSubDir = TokenizeDir(
                dirPath=subDirPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
            )

            if tokenizedSubDir:

                tokenizedDir[subDirPath.name] = tokenizedSubDir

    return tokenizedDir


def GetNumTokenDir(
    dirPath: Path | str,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
) -> int:
    """
    Get the number of tokens in all files within a directory based on the specified model or encoding.

    Parameters
    ----------
    dirPath : Path or str
        The path to the directory to count tokens for.
    model : str or None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        Whether to count tokens in files in subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.

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

    Examples
    --------
    >>> from core import GetNumTokenDir
    >>> total_tokens = GetNumTokenDir('/path/to/directory', model='gpt-4')
    >>> print(total_tokens)
    1500

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding('p50k_base')
    >>> total_tokens = GetNumTokenDir('/path/to/directory', encoding=encoding, recursive=False)
    >>> print(total_tokens)
    2000

    >>> # Counting tokens with recursion
    >>> total_tokens = GetNumTokenDir('/path/to/directory', model='gpt-3.5-turbo', recursive=True)
    >>> print(total_tokens)
    3000
    """
    if not isinstance(dirPath, (str, Path)):

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

    dirPath = Path(dirPath).resolve()

    if not dirPath.is_dir():

        raise ValueError(f'Given directory path "{dirPath}" is not a directory.')

    numFiles = _CountDirFiles(dirPath=dirPath, recursive=recursive)

    if not quiet:

        taskName = "Counting Tokens in Directory"
        _InitializeTask(taskName=taskName, total=numFiles, quiet=quiet)

    else:

        taskName = None

    runningTokenTotal = 0
    subDirPaths: list[Path] = []

    for entry in dirPath.iterdir():

        if entry.is_dir():

            subDirPaths.append(entry)

        else:

            if not quiet:

                _UpdateTask(
                    taskName=taskName,
                    advance=0,
                    description=f"Counting Tokens in {entry.relative_to(dirPath)}",
                    quiet=quiet,
                )

            try:

                runningTokenTotal += GetNumTokenFile(
                    filePath=entry,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    quiet=quiet,
                )

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Done Counting Tokens in {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

            except UnsupportedEncodingError:

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Skipping {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

                continue

    for subDirPath in subDirPaths:

        runningTokenTotal += GetNumTokenDir(
            dirPath=subDirPath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            recursive=recursive,
            quiet=quiet,
        )

    return runningTokenTotal


def TokenizeFiles(
    inputPath: Path | str | list[Path | str],
    /,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    exitOnListError: bool = True,
) -> list[int] | dict[str, list[int] | dict]:
    """
    Tokenize multiple files or all files within a directory into lists of token IDs.

    Parameters
    ----------
    inputPath : Path, str, or list of Path or str
        The path to a file or directory, or a list of file paths to tokenize.
    model : str or None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        If inputPath is a directory, whether to tokenize files in subdirectories
        recursively.
    quiet : bool, default False
        If True, suppress progress updates.
    exitOnListError : bool, default True
        If True, stop processing the list upon encountering an error. If False,
        skip files that cause errors.

    Returns
    -------
    list[int] | dict[str, list[int] | dict]
        - If `inputPath` is a file, returns a list of token IDs for that file.
        - If `inputPath` is a list of files, returns a dictionary where each key is
          the file name and the value is the list of token IDs for that file.
        - If `inputPath` is a directory:
          - If `recursive` is True, returns a nested dictionary where each key is a
            file or subdirectory name with corresponding token lists or sub-dictionaries.
          - If `recursive` is False, returns a dictionary with file names as keys and
            their token lists as values.

    Raises
    ------
    TypeError
        If the types of `inputPath`, `model`, `encodingName`, `encoding`, or
        `recursive` are incorrect.
    ValueError
        If any of the provided file paths in a list are not files, or if a provided
        directory path is not a directory.
    UnsupportedEncodingError
        If any of the files to be tokenized have an unsupported encoding.
    RuntimeError
        If the provided `inputPath` is neither a file, a directory, nor a list.

    Examples
    --------
    Tokenizing a list of files with a specified model:

    >>> from core import TokenizeFiles
    >>> tokens = TokenizeFiles(['file1.txt', 'file2.txt'], model='gpt-4')
    >>> print(tokens)
    {
        'file1.txt': [15496, 11, 995],
        'file2.txt': [1234, 5678, 91011]
    }

    Tokenizing a directory with a specific encoding and non-recursive:

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding('p50k_base')
    >>> tokens = TokenizeFiles('/path/to/directory', encoding=encoding, recursive=False)
    >>> print(tokens)
    {
        'file1.txt': [1234, 5678],
        'file2.txt': [91011, 1213],
        ...
    }

    Handling a list with a non-file entry:

    >>> TokenizeFiles(['file1.txt', 'directory'])
    Traceback (most recent call last):
        ...
    ValueError: Given list contains non-file entries: [Path('directory')]

    Tokenizing a directory with recursion:

    >>> tokens = TokenizeFiles('/path/to/directory', model='gpt-3.5-turbo', recursive=True)
    >>> print(tokens)
    {
        'file1.txt': [15496, 11, 995],
        'file2.txt': [1234, 5678, 91011],
        'subdir': {
            'file3.txt': [2345, 6789],
            'file4.txt': [3456, 7890]
        }
    }
    """

    if not isinstance(inputPath, (str, Path, list)):

        raise TypeError(
            f'Unexpected type for parameter "inputPath". Expected type: str, pathlib.Path, or list. Given type: {type(inputPath)}'
        )

    if isinstance(inputPath, list):

        if not all(isinstance(item, (str, Path)) for item in inputPath):

            listTypes = set(type(item) for item in inputPath)

            raise TypeError(
                f'Unexpected type for parameter "inputPath". Expected type: list of str or pathlib.Path. Given list contains types: {listTypes}'
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

    if isinstance(inputPath, list):

        inputPath = [Path(entry) for entry in inputPath]

        if not all(entry.is_file() for entry in inputPath):

            nonFiles = [entry for entry in inputPath if not entry.is_file()]

            raise ValueError(f"Given list contains non-file entries: {nonFiles}")

        else:

            tokenizedFiles: dict[str, list[int]] = dict()
            numFiles = len(inputPath)

            if not quiet:

                _InitializeTask(
                    taskName="Tokenizing File List", total=numFiles, quiet=quiet
                )

            for file in inputPath:

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File List",
                        advance=0,
                        description=f"Tokenizing {file.name}",
                        quiet=quiet,
                    )

                if exitOnListError:

                    tokenizedFiles[file.name] = TokenizeFile(
                        filePath=file,
                        model=model,
                        encodingName=encodingName,
                        encoding=encoding,
                        quiet=quiet,
                    )

                    if not quiet:

                        _UpdateTask(
                            taskName="Tokenizing File List",
                            advance=1,
                            description=f"Done Tokenizing {file.name}",
                            quiet=quiet,
                        )

                else:

                    try:

                        tokenizedFiles[file.name] = TokenizeFile(
                            filePath=file,
                            model=model,
                            encodingName=encodingName,
                            encoding=encoding,
                            quiet=quiet,
                        )

                        if not quiet:

                            _UpdateTask(
                                taskName="Tokenizing File List",
                                advance=1,
                                description=f"Done Tokenizing {file.name}",
                                quiet=quiet,
                            )

                    except UnsupportedEncodingError:

                        if not quiet:

                            _UpdateTask(
                                taskName="Tokenizing File List",
                                advance=1,
                                description=f"Skipping {file.name}",
                                quiet=quiet,
                            )

                        continue

            return tokenizedFiles

    else:

        inputPath = Path(inputPath)

    if inputPath.is_file():

        return TokenizeFile(
            filePath=inputPath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            quiet=quiet,
        )

    elif inputPath.is_dir():

        return TokenizeDir(
            dirPath=inputPath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            recursive=recursive,
            quiet=quiet,
        )

    else:

        raise RuntimeError(
            f'Unexpected error. Given inputPath "{inputPath}" is neither a file, a directory, nor a list.'
        )


def GetNumTokenFiles(
    inputPath: Path | str | list[Path | str],
    /,
    model: str | None = None,
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    exitOnListError: bool = True,
) -> int:
    """
    Get the number of tokens in multiple files or all files within a directory.

    Parameters
    ----------
    inputPath : Path, str, or list of Path or str
        The path to a file or directory, or a list of file paths to count tokens for.
    model : str or None, optional
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        If inputPath is a directory, whether to count tokens in files in
        subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.
    exitOnListError : bool, default True
        If True, stop processing the list upon encountering an error. If False,
        skip files that cause errors.

    Returns
    -------
    int
        The total number of tokens in the specified files or directory.

    Raises
    ------
    TypeError
        If the types of "inputPath", "model", "encodingName", "encoding", or
        "recursive" are incorrect.
    ValueError
        If any of the provided file paths in a list are not files, or if a provided
        directory path is not a directory.
    UnsupportedEncodingError
        If any of the files to be tokenized have an unsupported encoding.
    RuntimeError
        If the provided inputPath is neither a file, a directory, nor a list.

    Examples
    --------
    Tokenizing a list of files with a specified model:

    >>> from core import GetNumTokenFiles
    >>> total_tokens = GetNumTokenFiles(['file1.txt', 'file2.txt'], model='gpt-4')
    >>> print(total_tokens)
    3000

    Tokenizing a directory with a specific encoding and non-recursive:

    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding('p50k_base')
    >>> total_tokens = GetNumTokenFiles('/path/to/directory', encoding=encoding, recursive=False)
    >>> print(total_tokens)
    4000

    Handling a list with a non-file entry:

    >>> GetNumTokenFiles(['file1.txt', 'directory'])
    Traceback (most recent call last):
        ...
    ValueError: Given list contains non-file entries: [Path('directory')]

    Counting tokens in a directory with recursion:

    >>> total_tokens = GetNumTokenFiles('/path/to/directory', model='gpt-3.5-turbo', recursive=True)
    >>> print(total_tokens)
    5000
    """

    if not isinstance(inputPath, (str, Path, list)):

        raise TypeError(
            f'Unexpected type for parameter "inputPath". Expected type: str, pathlib.Path, or list. Given type: {type(inputPath)}'
        )

    if isinstance(inputPath, list):

        if not all(isinstance(item, (str, Path)) for item in inputPath):

            listTypes = set(type(item) for item in inputPath)

            raise TypeError(
                f'Unexpected type for parameter "inputPath". Expected type: list of str or pathlib.Path. Given list contains types: {listTypes}'
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

    if isinstance(inputPath, list):

        inputPath = [Path(entry) for entry in inputPath]

        if not all(entry.is_file() for entry in inputPath):

            nonFiles = [entry for entry in inputPath if not entry.is_file()]

            raise ValueError(f"Given list contains non-file entries: {nonFiles}")

        else:

            runningTokenTotal = 0
            numFiles = len(inputPath)

            if not quiet:

                _InitializeTask(
                    taskName="Counting Tokens in File List", total=numFiles, quiet=quiet
                )

            for file in inputPath:

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File List",
                        advance=0,
                        description=f"Counting Tokens in {file.name}",
                        quiet=quiet,
                    )

                if exitOnListError:

                    runningTokenTotal += GetNumTokenFile(
                        filePath=file,
                        model=model,
                        encodingName=encodingName,
                        encoding=encoding,
                        quiet=quiet,
                    )

                    if not quiet:

                        _UpdateTask(
                            taskName="Counting Tokens in File List",
                            advance=1,
                            description=f"Done Counting Tokens in {file.name}",
                            quiet=quiet,
                        )

                else:

                    try:

                        runningTokenTotal += GetNumTokenFile(
                            filePath=file,
                            model=model,
                            encodingName=encodingName,
                            encoding=encoding,
                            quiet=quiet,
                        )

                        if not quiet:

                            _UpdateTask(
                                taskName="Counting Tokens in File List",
                                advance=1,
                                description=f"Done Counting Tokens in {file.name}",
                                quiet=quiet,
                            )

                    except UnsupportedEncodingError:

                        if not quiet:

                            _UpdateTask(
                                taskName="Counting Tokens in File List",
                                advance=1,
                                description=f"Skipping {file.name}",
                                quiet=quiet,
                            )

                        continue

            return runningTokenTotal

    else:

        inputPath = Path(inputPath)

    if inputPath.is_file():

        return GetNumTokenFile(
            filePath=inputPath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            quiet=quiet,
        )

    elif inputPath.is_dir():

        return GetNumTokenDir(
            dirPath=inputPath,
            model=model,
            encodingName=encodingName,
            encoding=encoding,
            recursive=recursive,
            quiet=quiet,
        )

    else:

        raise RuntimeError(
            f'Unexpected error. Given inputPath "{inputPath}" is neither a file, a directory, nor a list.'
        )
