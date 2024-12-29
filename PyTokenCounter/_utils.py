"""
_utils.py

Utilities for file operations, including reading text files with UTF-8 encoding.
Provides a custom exception for unsupported encodings.
"""

from pathlib import Path

import chardet


class UnsupportedEncodingError(Exception):
    """
    Exception raised when a file's encoding is not UTF-8 or ASCII.

    Attributes
    ----------
    encoding : str | None
        The detected encoding of the file.
    filePath : pathlib.Path or str
        The path to the file that caused the error.
    message : str
        A message explaining the error.
    """

    def __init__(
        self,
        encoding: str | None,
        filePath: Path | str,
        message: str = "File encoding is not supported",
    ):
        self.encoding = encoding
        self.filePath = filePath
        self.message = (
            f"{message}. Detected encoding: {encoding}. File path: {filePath}"
        )
        super().__init__(self.message)


def ReadTextFile(filePath: Path | str) -> str | tuple[None, str | None]:
    """
    Reads a text file if it is UTF-8 or ASCII encoded. If the file is not UTF-8 or ASCII encoded,
    returns a tuple containing `None` and the detected encoding.

    Parameters
    ----------
    filePath : pathlib.Path or str
        The path to the file to be read. Can be provided as a string or a `Path` object.

    Returns
    -------
    str
        The content of the file as a string if it is UTF-8 or ASCII encoded.
    tuple[None, str | None]
        A tuple `(None, encoding)` if the file is not UTF-8 or ASCII encoded, where `encoding`
        is the detected encoding or `None` if the encoding could not be detected.

    Raises
    ------
    TypeError
        If the input `filePath` is not of type `str` or `pathlib.Path`.
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    Reading a valid UTF-8 encoded file:

    >>> from utils import ReadTextFile
    >>> content = ReadTextFile('example.txt')
    >>> print(content)

    Handling a file with a non-UTF-8 encoding:

    >>> result = ReadTextFile('non_utf8.txt')
    >>> print(result)
    (None, 'ISO-8859-1')

    Handling a non-existent file:

    >>> ReadTextFile('non_existent.txt')
    Traceback (most recent call last):
        ...
    FileNotFoundError: File not found: /absolute/path/to/non_existent.txt

    Ensuring type safety:

    >>> ReadTextFile(123)
    Traceback (most recent call last):
        ...
    TypeError: Unexpected type for parameter "filePath". Expected type: str or pathlib.Path. Given type: <class 'int'>
    """

    if not isinstance(filePath, str) and not isinstance(filePath, Path):

        raise TypeError(
            f'Unexpected type for parameter "filePath". Expected type: str or pathlib.Path. Given type: {type(filePath)}'
        )

    file = Path(filePath).resolve()

    if not file.exists():

        raise FileNotFoundError(f"File not found: {file}")

    with file.open("rb") as binaryFile:

        detection = chardet.detect(binaryFile.read())
        encoding = detection["encoding"]

    if encoding and encoding.lower() == "utf-8":

        return file.read_text(encoding="utf-8")

    elif encoding and encoding.lower() == "ascii":

        return file.read_text(encoding="ascii")

    else:

        return (None, encoding)
