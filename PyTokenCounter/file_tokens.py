from collections import OrderedDict
from pathlib import Path

import tiktoken

from .encoding_utils import ReadTextFile, UnsupportedEncodingError
from .progress import _InitializeTask, _UpdateTask, _tasks
from .core import BINARY_EXTENSIONS, TokenizeStr

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


def _ComputeTotalTokens(structure: any) -> int:
    """
    Compute the total number of tokens from a nested token structure.

    This helper recursively processes a token structure that may be:
    - an integer (in which case it is returned directly),
    - a list of tokens (returning its length), or
    - a dictionary (in which case it sums the token counts of its values).
    If the dictionary has a "numTokens" key, that value is returned.

    Parameters
    ----------
    structure : any
        The token structure which can be an int, list, or dict.

    Returns
    -------
    int
        The total number of tokens represented in the structure.
    """

    if isinstance(structure, int):

        return structure

    elif isinstance(structure, list):

        return len(structure)

    elif isinstance(structure, dict):

        if "numTokens" in structure:

            return structure["numTokens"]

        total = 0

        for value in structure.values():

            total += _ComputeTotalTokens(value)

        return total

    else:

        return 0

def TokenizeFile(
    filePath: Path | str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
    mapTokens: bool = True,
) -> list[int] | OrderedDict[str, list[int] | OrderedDict]:
    """
    Tokenize the contents of a file into a list of token IDs using the specified model or encoding.

    Parameters
    ----------
    filePath : Path or str
        The path to the file to tokenize.
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
    mapTokens : bool, optional, default=True
        If True, returns the tokenized result as an OrderedDict with the file name as the key.
        The associated value is itself an OrderedDict with two keys:
            - "numTokens": the number of tokens in the file,
            - "tokens": the list of token IDs.
        Otherwise, returns a list of token IDs.

    Returns
    -------
    list[int] or OrderedDict[str, list[int] | OrderedDict]
        If mapTokens is False, a list of token IDs representing the tokenized file contents.
        If mapTokens is True, an OrderedDict with the file name as the key and as its value another OrderedDict




        with keys "numTokens" and "tokens" corresponding to the token count and token list, respectively.





    Raises
    ------
    TypeError
        If the types of `filePath`, `model`, `encodingName`, or `encoding` are incorrect.
    UnsupportedEncodingError
        If the file's encoding is not supported.
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    Tokenizing a file with a specified model:

    >>> from pathlib import Path
    >>> from PyTokenCounter import TokenizeFile
    >>> filePath = Path("./PyTokenCounter/Tests/Input/TestFile1.txt")
    >>> tokens = TokenizeFile(filePath=filePath, model="gpt-4o")
    >>> print(tokens)
    OrderedDict({
        "TestFile1.txt": OrderedDict({
            "numTokens": 221,
            "tokens": [2305, 290, 7334, 132491, 11, 290, ..., 11526, 13]
        })
    })

    Tokenizing a file with an existing encoding object:

    >>> from pathlib import Path
    >>> from PyTokenCounter import TokenizeFile
    >>> import tiktoken
    >>> encoding = tiktoken.get_encoding("p50k_base")
    >>> filePath = Path("./PyTokenCounter/Tests/Input/TestFile2.txt")
    >>> tokens = TokenizeFile(filePath=filePath, encoding=encoding)
    >>> print(tokens)
    OrderedDict({
        "TestFile2.txt": OrderedDict({
            "numTokens": 213,
            "tokens": [976, 13873, 10377, 472, 261, ..., 3333, 13]
        })
    })
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
        mapTokens=mapTokens,
    )

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f"Done Tokenizing {filePath.name}",
            quiet=quiet,
        )

    if mapTokens:

        # Return an OrderedDict with the file name as key and a nested OrderedDict
        # containing "numTokens" and "tokens" as keys.

        return OrderedDict(
            {filePath.name: OrderedDict({"numTokens": len(tokens), "tokens": tokens})}
        )

    else:

        return tokens


def GetNumTokenFile(
    filePath: Path | str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    quiet: bool = False,
    mapTokens: bool = False,
) -> int | OrderedDict[str, int]:
    """
    Get the number of tokens in a file based on the specified model or encoding.

    Parameters
    ----------
    filePath : Path or str
        The path to the file to count tokens for.
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
    mapTokens : bool, optional, default=False
        If True, returns the token count as an OrderedDict with the file name as the key.
        Otherwise, returns an integer token count.

    Returns
    -------
    int or OrderedDict[str, int]
        If mapTokens is False, the number of tokens in the file.
        If mapTokens is True, an OrderedDict mapping the file name to its token count.

    Raises
    ------
    TypeError
        If the types of "filePath", "model", "encodingName", or "encoding" are incorrect.
    ValueError
        If the provided "model" or "encodingName" is invalid.
    UnsupportedEncodingError
        If the file's encoding is not supported.
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> from PyTokenCounter import GetNumTokenFile
    >>> from pathlib import Path
    >>> filePath = Path("./PyTokenCounter/Tests/Input/TestFile1.txt")
    >>> numTokens = GetNumTokenFile(filePath=filePath, model="gpt-4o")
    >>> print(numTokens)
    221
    >>> filePath = Path("./PyTokenCounter/Tests/Input/TestFile2.txt")
    >>> numTokens = GetNumTokenFile(filePath=filePath, model="gpt-4o")
    >>> print(numTokens)
    213
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

    tokens = TokenizeFile(
        filePath=filePath,
        model=model,
        encodingName=encodingName,
        encoding=encoding,
        quiet=quiet,
        mapTokens=False,
    )

    count = len(tokens)

    if hasBar:

        _UpdateTask(
            taskName=taskName,
            advance=1,
            description=f"Done Counting Tokens in {filePath.name}",
            quiet=quiet,
        )

    if mapTokens:

        return OrderedDict({filePath.name: count})

    else:

        return count


def TokenizeDir(
    dirPath: Path | str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    mapTokens: bool = True,
    excludeBinary: bool = True,
    includeHidden: bool = False,
) -> OrderedDict[str, list[int] | OrderedDict]:
    """
    Tokenize all files in a directory into lists of token IDs using the specified model or encoding.

    Parameters
    ----------
    dirPath : Path or str
        The path to the directory to tokenize.
    model : str or None, optional, default="gpt-4o"
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
    mapTokens : bool, default True
        If True, returns the tokenized result as a nested OrderedDict with file or directory names as keys.
        For files, the value is an OrderedDict with the file name as key and the token list as the value.
        For directories, the value is an OrderedDict with two keys:
            - "numTokens": the total number of tokens in that directory and its subdirectories,
            - "tokens": the nested OrderedDict mapping of tokenized contents.
    excludeBinary : bool, default True
        Excludes any binary files by skipping over them.
    includeHidden : bool, default False
        Skips over hidden files and directories, including subdirectories and files of a hidden directory.

    Returns
    -------
    OrderedDict[str, list[int] | OrderedDict]
        A nested OrderedDictionary where each key is a file or subdirectory name:
        - For a file, if mapTokens is True, the value is an OrderedDict with the file name as key and the token list as the value;




          if mapTokens is False, the value is the list of token IDs.




        - For a subdirectory, if mapTokens is True, the value is an OrderedDict with keys "numTokens" (total tokens in the directory)
          and "tokens" (the nested token structure); if mapTokens is False, the value is the nested token structure.

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
    >>> from PyTokenCounter import TokenizeDir
    >>> from pathlib import Path
    >>> dirPath = Path("./PyTokenCounter/Tests/Input/TestDirectory")
    >>> tokenizedDir = TokenizeDir(dirPath=dirPath, model='gpt-4o')
    >>> print(tokenizedDir)
    {
        'TestDir1.txt': OrderedDict({'TestDir1.txt': [976, 19458, 5831, 23757, 306, 290, ..., 26321, 13]}),
        'TestSubDir': OrderedDict({
            'numTokens': 132,
            'tokens': {
                'TestDir2.txt': OrderedDict({'TestDir2.txt': [976, 5030, 45940, 295, 483, ..., 1665, 4717, 13]})
            }
        })
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

    # Skip processing if the directory itself is hidden and hidden files are not to be included.

    if not includeHidden and dirPath.name.startswith("."):

        return OrderedDict()

    if not dirPath.is_dir():

        raise ValueError(f'Given directory path "{dirPath}" is not a directory.')

    numFiles = _CountDirFiles(dirPath=dirPath, recursive=recursive)

    if not quiet:

        taskName = "Tokenizing Directory"
        _InitializeTask(taskName=taskName, total=numFiles, quiet=quiet)

    else:

        taskName = None

    tokenizedDir: OrderedDict[str, list[int] | OrderedDict] = OrderedDict()
    subDirPaths: list[Path] = []

    for entry in dirPath.iterdir():

        # Skip hidden files and directories if includeHidden is False.

        if not includeHidden and entry.name.startswith("."):

            continue

        if entry.is_dir():

            subDirPaths.append(entry)

        else:

            # Skip binary files if excludeBinary is True.

            if excludeBinary and entry.suffix.lower() in BINARY_EXTENSIONS:

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Skipping binary file {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

                continue

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
                    mapTokens=mapTokens,
                )

            except UnicodeDecodeError:

                if excludeBinary:

                    if not quiet:

                        _UpdateTask(
                            taskName=taskName,
                            advance=1,
                            description=f"Skipping binary file {entry.relative_to(dirPath)}",
                            quiet=quiet,
                        )

                    continue

                else:

                    raise

            if mapTokens:

                # TokenizeFile returns an OrderedDict with the file name as key.
                tokenizedDir.update(tokenizedFile)

            else:

                tokenizedDir[entry.name] = tokenizedFile

            if not quiet:

                _UpdateTask(
                    taskName=taskName,
                    advance=1,
                    description=f"Done Tokenizing {entry.relative_to(dirPath)}",
                    quiet=quiet,
                )

    if recursive:

        for subDirPath in subDirPaths:

            subStructure = TokenizeDir(
                dirPath=subDirPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
                excludeBinary=excludeBinary,
                includeHidden=includeHidden,
                mapTokens=mapTokens,
            )

            if mapTokens:

                totalTokens = _ComputeTotalTokens(subStructure)
                tokenizedDir[subDirPath.name] = OrderedDict(
                    {"numTokens": totalTokens, "tokens": subStructure}
                )

            else:

                tokenizedDir[subDirPath.name] = subStructure

    return tokenizedDir


def GetNumTokenDir(
    dirPath: Path | str,
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    excludeBinary: bool = True,
    includeHidden: bool = False,
    mapTokens: bool = False,
) -> int | OrderedDict[str, int | OrderedDict]:
    """
    Get the number of tokens in all files within a directory (and its subdirectories if recursive is True).

    When mapTokens is True, the function returns an OrderedDict with two keys:
        - "numTokens": total number of tokens in the directory (and its subdirectories).
        - "tokens": an OrderedDict mapping each file or subdirectory name to its token count (or nested OrderedDict).
    When mapTokens is False, the function returns an integer representing the total token count.

    Parameters
    ----------
    dirPath : Path or str
        The path to the directory to count tokens for.
    model : str or None, optional
        The model to use for encoding (default is "gpt-4o").
    encodingName : str or None, optional
        The name of the encoding to use.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use.
    recursive : bool, default True
        Whether to process subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.
    excludeBinary : bool, default True
        If True, binary files are skipped.
    includeHidden : bool, default False
        If False, hidden files and directories are skipped.
    mapTokens : bool, default False
        If True, returns a detailed mapping (OrderedDict) with token counts;
        otherwise, returns only the total token count as an integer.

    Returns
    -------
    int or OrderedDict[str, int | OrderedDict]
        An integer total token count if mapTokens is False, or an OrderedDict with keys:
            - "numTokens": total token count (int)
            - "tokens": OrderedDict mapping file or subdirectory names to token counts or nested mappings

    Raises
    ------
    ValueError
        If the provided dirPath is not a directory.
    TypeError
        If any parameter is of an unexpected type.

    Examples
    --------
    >>> from pathlib import Path
    >>> dirPath = Path("./PyTokenCounter/Tests/Input/TestDirectory")
    >>> tokenDir = GetNumTokenDir(
    ...     dirPath=dirPath,
    ...     model="gpt-4o",
    ...     recursive=True,
    ...     mapTokens=True
    ... )
    >>> print(tokenDir)
    OrderedDict({
        "numTokens": 1500,
        "tokens": OrderedDict({
            "TestFile1.txt": 221,
            "TestSubDir": OrderedDict({
                "numTokens": 132,
                "tokens": OrderedDict({
                    "TestFile2.txt": 132
                })
            })
        })
    })
    """
    dirPath = Path(dirPath).resolve()

    if not includeHidden and dirPath.name.startswith("."):

        return (
            OrderedDict([("numTokens", 0), ("tokens", OrderedDict())])
            if mapTokens
            else 0
        )

    if not dirPath.is_dir():

        raise ValueError(f'Given path "{dirPath}" is not a directory.')

    numFiles = _CountDirFiles(dirPath=dirPath, recursive=recursive)

    if not quiet:

        taskName = "Counting Tokens in Directory"
        _InitializeTask(taskName=taskName, total=numFiles, quiet=quiet)

    else:

        taskName = None

    if mapTokens:

        tokensMapping = OrderedDict()
        totalTokens = 0

    else:

        runningTokenTotal = 0

    subDirPaths: list[Path] = []

    for entry in dirPath.iterdir():

        if not includeHidden and entry.name.startswith("."):

            continue

        if entry.is_dir():

            if recursive:

                subDirPaths.append(entry)

        else:

            if excludeBinary and entry.suffix.lower() in BINARY_EXTENSIONS:

                if not quiet:

                    _UpdateTask(
                        taskName=taskName,
                        advance=1,
                        description=f"Skipping binary file {entry.relative_to(dirPath)}",
                        quiet=quiet,
                    )

                continue

            if not quiet:

                _UpdateTask(
                    taskName=taskName,
                    advance=0,
                    description=f"Counting Tokens in {entry.relative_to(dirPath)}",
                    quiet=quiet,
                )

            try:

                count = GetNumTokenFile(
                    filePath=entry,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    quiet=quiet,
                    mapTokens=False,
                )

            except UnicodeDecodeError:

                if excludeBinary:

                    if not quiet:

                        _UpdateTask(
                            taskName=taskName,
                            advance=1,
                            description=f"Skipping binary file {entry.relative_to(dirPath)}",
                            quiet=quiet,
                        )

                    continue

                else:

                    raise

            if mapTokens:

                tokensMapping[entry.name] = count
                totalTokens += count

            else:

                runningTokenTotal += count

            if not quiet:

                _UpdateTask(
                    taskName=taskName,
                    advance=1,
                    description=f"Done Counting Tokens in {entry.relative_to(dirPath)}",
                    quiet=quiet,
                )

    for subDir in subDirPaths:

        if mapTokens:

            subMapping = GetNumTokenDir(
                dirPath=subDir,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
                excludeBinary=excludeBinary,
                includeHidden=includeHidden,
                mapTokens=True,
            )
            tokensMapping[subDir.name] = subMapping
            subTotal = subMapping.get("numTokens", 0)
            totalTokens += subTotal

        else:

            subTotal = GetNumTokenDir(
                dirPath=subDir,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
                excludeBinary=excludeBinary,
                includeHidden=includeHidden,
                mapTokens=False,
            )
            runningTokenTotal += subTotal

    if mapTokens:

        result = OrderedDict([("numTokens", totalTokens), ("tokens", tokensMapping)])

        return result

    else:

        return runningTokenTotal


def TokenizeFiles(
    inputPath: Path | str | list[Path | str],
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    exitOnListError: bool = True,
    mapTokens: bool = True,
    excludeBinary: bool = True,
    includeHidden: bool = False,
) -> list[int] | OrderedDict[str, list[int] | OrderedDict]:
    """
    Tokenize multiple files or all files within a directory into lists of token IDs using the specified model or encoding.

    Parameters
    ----------
    inputPath : Path, str, or list of Path or str
        The path to a file or directory, or a list of file/directory paths to tokenize.
    model : str or None, optional, default="gpt-4o"
        The name of the model to use for encoding. If provided, the encoding
        associated with the model will be used.
    encodingName : str or None, optional
        The name of the encoding to use. If provided, it must match the encoding
        associated with the specified model.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use for tokenization. If provided,
        it must match the encoding derived from the model or encodingName.
    recursive : bool, default True
        If inputPath is a directory, whether to tokenize files in subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.
    exitOnListError : bool, default True
        If True, stop processing the list upon encountering an error. If False,
        skip files/directories that cause errors.
    excludeBinary : bool, default True
        Excludes any binary files by skipping over them.
    includeHidden : bool, default False
        Skips over hidden files and directories, including subdirectories and files of a hidden directory.
    mapTokens : bool, default True
        If True, returns the tokenized result as an OrderedDict with file or directory names as keys.
        For files, the value is an OrderedDict with the file name as key and the token list as the value.
        For directories, the value is an OrderedDict with two keys:
            - "numTokens": the total number of tokens in that directory (including subdirectories if recursive is True),
            - "tokens": the nested OrderedDict mapping file/directory names to their tokenized contents.
        If False and inputPath is a file, returns a list of token IDs.

    Returns
    -------
    list[int] or OrderedDict[str, list[int] | OrderedDict]
        - If `inputPath` is a file, returns a list of token IDs for that file if mapTokens is False,
          or an OrderedDict with the file name as the key and the token list as the value if mapTokens is True.
        - If `inputPath` is a list, returns an OrderedDict mapping each file or directory name to its tokenized output.
          For directories (when mapTokens is True) the value includes both "numTokens" and "tokens" keys.
        - If `inputPath` is a directory:
          - If `recursive` is True, returns a nested OrderedDict where for each subdirectory the value is an OrderedDict




            with keys "numTokens" (total tokens in that directory and its subdirectories) and "tokens" (the nested token structure).




          - If `recursive` is False, returns an OrderedDict with file names as keys and their token lists as values.

    Raises
    ------
    TypeError
        If the types of `inputPath`, `model`, `encodingName`, `encoding`, or `recursive` are incorrect.
    ValueError
        If any of the provided paths in a list are neither files nor directories.
    UnsupportedEncodingError
        If any of the files to be tokenized have an unsupported encoding.
    RuntimeError
        If the provided `inputPath` is neither a file, a directory, nor a list.

    Examples
    --------
    Tokenizing a mixed list of files and directories:
    >>> from PyTokenCounter import TokenizeFiles
    >>> from pathlib import Path
    >>> paths = [Path("./TestFile1.txt"), Path("./TestDirectory")]
    >>> tokens = TokenizeFiles(inputPath=paths, model='gpt-4o', recursive=True)
    >>> print(tokens)
    OrderedDict({
        'TestFile1.txt': OrderedDict({'TestFile1.txt': [2305, 290, 7334, ...]}),
        'TestDirectory': OrderedDict({
            'numTokens': 657,
            'tokens': { ... nested structure ... }
        })
    })
    """
    # If inputPath is a list, allow files and directories.

    if isinstance(inputPath, list):

        inputPath = [Path(entry) for entry in inputPath]
        tokenizedResults: OrderedDict[str, any] = OrderedDict()
        numEntries = len(inputPath)

        if not quiet:

            _InitializeTask(
                taskName="Tokenizing File/Directory List", total=numEntries, quiet=quiet
            )

        for entry in inputPath:

            if not includeHidden and entry.name.startswith("."):

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File/Directory List",
                        advance=1,
                        description=f"Skipping hidden entry {entry.name}",
                        quiet=quiet,
                    )

                continue

            if entry.is_file():

                if excludeBinary and entry.suffix.lower() in BINARY_EXTENSIONS:

                    if not quiet:

                        _UpdateTask(
                            taskName="Tokenizing File/Directory List",
                            advance=1,
                            description=f"Skipping binary file {entry.name}",
                            quiet=quiet,
                        )

                    continue

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File/Directory List",
                        advance=0,
                        description=f"Tokenizing file {entry.name}",
                        quiet=quiet,
                    )

                try:

                    result = TokenizeFile(
                        filePath=entry,
                        model=model,
                        encodingName=encodingName,
                        encoding=encoding,
                        quiet=quiet,
                        mapTokens=mapTokens,
                    )

                except UnicodeDecodeError:

                    if excludeBinary:

                        if not quiet:

                            _UpdateTask(
                                taskName="Tokenizing File/Directory List",
                                advance=1,
                                description=f"Skipping binary file {entry.name}",
                                quiet=quiet,
                            )

                        continue

                    else:

                        raise

                if mapTokens:

                    tokenizedResults.update(result)

                else:

                    tokenizedResults[entry.name] = result

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File/Directory List",
                        advance=1,
                        description=f"Done tokenizing file {entry.name}",
                        quiet=quiet,
                    )

            elif entry.is_dir():

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File/Directory List",
                        advance=0,
                        description=f"Tokenizing directory {entry.name}",
                        quiet=quiet,
                    )
                subMapping = TokenizeDir(
                    dirPath=entry,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    recursive=recursive,
                    quiet=quiet,
                    excludeBinary=excludeBinary,
                    includeHidden=includeHidden,
                    mapTokens=mapTokens,
                )

                if mapTokens:

                    totalTokens = _ComputeTotalTokens(subMapping)
                    tokenizedResults[entry.name] = OrderedDict(
                        {"numTokens": totalTokens, "tokens": subMapping}
                    )

                else:

                    tokenizedResults[entry.name] = subMapping

                if not quiet:

                    _UpdateTask(
                        taskName="Tokenizing File/Directory List",
                        advance=1,
                        description=f"Done tokenizing directory {entry.name}",
                        quiet=quiet,
                    )

            else:

                raise ValueError(f"Entry '{entry}' is neither a file nor a directory.")

        return tokenizedResults

    # Not a list: if inputPath is a file or directory, process as before.

    else:

        inputPath = Path(inputPath)

        if inputPath.is_file():

            if not includeHidden and inputPath.name.startswith("."):

                return [] if not mapTokens else OrderedDict()

            if excludeBinary and inputPath.suffix.lower() in BINARY_EXTENSIONS:

                return [] if not mapTokens else OrderedDict()

            return TokenizeFile(
                filePath=inputPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                quiet=quiet,
                mapTokens=mapTokens,
            )

        elif inputPath.is_dir():

            return TokenizeDir(
                dirPath=inputPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
                excludeBinary=excludeBinary,
                includeHidden=includeHidden,
                mapTokens=mapTokens,
            )

        else:

            raise RuntimeError(
                f'Unexpected error. Given inputPath "{inputPath}" is neither a file, a directory, nor a list.'
            )


def GetNumTokenFiles(
    inputPath: Path | str | list[Path | str],
    model: str | None = "gpt-4o",
    encodingName: str | None = None,
    encoding: tiktoken.Encoding | None = None,
    recursive: bool = True,
    quiet: bool = False,
    exitOnListError: bool = True,
    excludeBinary: bool = True,
    includeHidden: bool = False,
    mapTokens: bool = False,
) -> int | OrderedDict[str, int]:
    """
    Get the number of tokens in multiple files or all files within a directory based on the
    specified model or encoding.

    Parameters
    ----------
    inputPath : Path, str, or list of Path or str
        The path to a file or directory, or a list of file/directory paths to count tokens for.
    model : str or None, optional
        The name of the model to use for encoding (default is "gpt-4o").
    encodingName : str or None, optional
        The name of the encoding to use.
    encoding : tiktoken.Encoding or None, optional
        An existing tiktoken.Encoding object to use.
    recursive : bool, default True
        If inputPath is a directory, whether to count tokens in files in subdirectories recursively.
    quiet : bool, default False
        If True, suppress progress updates.
    exitOnListError : bool, default True
        If True, stop processing the list upon encountering an error.
    excludeBinary : bool, default True
        If True, binary files are skipped.
    includeHidden : bool, default False
        If False, hidden files and directories are skipped.
    mapTokens : bool, default False
        If True, returns an OrderedDict mapping file/directory names to token counts (or nested
        OrderedDict for directories). If False, returns an integer total token count.

    Returns
    -------
    int or OrderedDict[str, int]
        If mapTokens is False, the total number of tokens.
        If mapTokens is True, an OrderedDict mapping file/directory names to token counts.

    Raises
    ------
    TypeError
        If the types of parameters are incorrect.
    ValueError
        If an entry is neither a file nor a directory.
    RuntimeError
        If the inputPath is not a file, directory, or list.

    Examples
    --------
    >>> from pathlib import Path
    >>> paths = [Path("./TestFile1.txt"), Path("./TestDirectory")]
    >>> totalTokens = GetNumTokenFiles(
    ...     inputPath=paths,
    ...     model="gpt-4o",
    ...     recursive=True,
    ...     mapTokens=True
    ... )
    >>> print(totalTokens)
    OrderedDict({
        'TestFile1.txt': 221,
        'TestDirectory': OrderedDict({
            "numTokens": 657,
            "tokens": OrderedDict({ ... })
        })
    })
    """

    if isinstance(inputPath, list):

        inputPath = [Path(entry) for entry in inputPath]

        if mapTokens:

            result: OrderedDict[str, int | OrderedDict] = OrderedDict()

        else:

            runningTokenTotal = 0

        numEntries = len(inputPath)

        if not quiet:

            _InitializeTask(
                taskName="Counting Tokens in File/Directory List",
                total=numEntries,
                quiet=quiet,
            )

        for entry in inputPath:

            if not includeHidden and entry.name.startswith("."):

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File/Directory List",
                        advance=1,
                        description=f"Skipping hidden entry {entry.name}",
                        quiet=quiet,
                    )

                continue

            if entry.is_file():

                if excludeBinary and entry.suffix.lower() in BINARY_EXTENSIONS:

                    if not quiet:

                        _UpdateTask(
                            taskName="Counting Tokens in File/Directory List",
                            advance=1,
                            description=f"Skipping binary file {entry.name}",
                            quiet=quiet,
                        )

                    continue

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File/Directory List",
                        advance=0,
                        description=f"Counting tokens in file {entry.name}",
                        quiet=quiet,
                    )

                try:

                    # For files, always get an integer count.
                    count = GetNumTokenFile(
                        filePath=entry,
                        model=model,
                        encodingName=encodingName,
                        encoding=encoding,
                        quiet=quiet,
                        mapTokens=False,
                    )

                except UnicodeDecodeError:

                    if excludeBinary:

                        if not quiet:

                            _UpdateTask(
                                taskName="Counting Tokens in File/Directory List",
                                advance=1,
                                description=f"Skipping binary file {entry.name}",
                                quiet=quiet,
                            )

                        continue

                    else:

                        raise

                if mapTokens:

                    result[entry.name] = count

                else:

                    runningTokenTotal += count

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File/Directory List",
                        advance=1,
                        description=f"Done counting tokens in file {entry.name}",
                        quiet=quiet,
                    )

            elif entry.is_dir():

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File/Directory List",
                        advance=0,
                        description=f"Counting tokens in directory {entry.name}",
                        quiet=quiet,
                    )
                subMapping = GetNumTokenDir(
                    dirPath=entry,
                    model=model,
                    encodingName=encodingName,
                    encoding=encoding,
                    recursive=recursive,
                    quiet=quiet,
                    excludeBinary=excludeBinary,
                    includeHidden=includeHidden,
                    mapTokens=mapTokens,
                )

                if mapTokens:

                    result[entry.name] = subMapping

                else:

                    runningTokenTotal += subMapping.get("numTokens", 0)

                if not quiet:

                    _UpdateTask(
                        taskName="Counting Tokens in File/Directory List",
                        advance=1,
                        description=f"Done counting tokens in directory {entry.name}",
                        quiet=quiet,
                    )

            else:

                raise ValueError(f"Entry '{entry}' is neither a file nor a directory.")

        return result if mapTokens else runningTokenTotal

    else:

        inputPath = Path(inputPath)

        if inputPath.is_file():

            if not includeHidden and inputPath.name.startswith("."):

                return OrderedDict() if mapTokens else 0

            if excludeBinary and inputPath.suffix.lower() in BINARY_EXTENSIONS:

                return OrderedDict() if mapTokens else 0

            return GetNumTokenFile(
                filePath=inputPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                quiet=quiet,
                mapTokens=False,
            )

        elif inputPath.is_dir():

            dirMapping = GetNumTokenDir(
                dirPath=inputPath,
                model=model,
                encodingName=encodingName,
                encoding=encoding,
                recursive=recursive,
                quiet=quiet,
                excludeBinary=excludeBinary,
                includeHidden=includeHidden,
                mapTokens=mapTokens,
            )

            return dirMapping if mapTokens else dirMapping.get("numTokens", 0)

        else:

            raise RuntimeError(
                f'Unexpected error. Given inputPath "{inputPath}" is neither a file, a directory, nor a list.'
            )

