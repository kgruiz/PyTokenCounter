# PyTokenCounter/_utils.py

"""
Utilities for file operations, including reading text files with UTF-8 encoding.
Provides a custom exception for unsupported encodings.

Key Classes
-----------
- ``UnsupportedEncodingError`` : Custom exception for unsupported file encodings.

Key Functions
-------------
- ``ReadTextFile`` : Reads a text file using its detected encoding.
"""


from pathlib import Path

import chardet
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# Custom exception for unsupported file encodings
class UnsupportedEncodingError(Exception):
    """
    Exception raised when a file's encoding is not UTF-8 or ASCII.
    """

    def __init__(
        self,
        encoding: str | None,
        filePath: Path | str,
        message: str = "File encoding is not supported",
    ):
        self.encoding = encoding
        self.filePath = filePath

        # Build a rich formatted error message
        errorText = Text()
        errorText.append(f"{message}", style="bold red")
        errorText.append("\n\n")
        errorText.append("Detected encoding: ", style="green")
        errorText.append(f"{encoding}", style="bold")
        errorText.append("\n")
        errorText.append("File path: ", style="green")
        errorText.append(f"{filePath}", style="bold blue")

        panel = Panel(
            errorText, title="Encoding Error", title_align="left", border_style="red"
        )

        console = Console(width=80, color_system="truecolor", record=True)

        with console.capture() as capture:

            console.print("")  # Add a new line before the panel
            console.print(panel)
        captured = capture.get()

        # Store the formatted panel; pass a plain message to the base Exception
        self.message = captured
        super().__init__(message)

        # Flag to ensure the rich panel is output only once
        self._printed = False

    def __str__(self) -> str:
        # Return the rich formatted error message only the first time __str__ is called

        if not self._printed:

            self._printed = True

            return self.message

        return ""


# Set module for correct traceback display
UnsupportedEncodingError.__module__ = "PyTokenCounter"


def ReadTextFile(filePath: Path | str) -> str:
    """
    Reads a text file using its detected encoding.
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

    if encoding:

        actualEncoding = encoding
        encoding = "utf-8"

        try:

            return file.read_text(encoding=encoding)

        except UnicodeDecodeError:

            raise UnsupportedEncodingError(encoding=actualEncoding, filePath=filePath)

    else:

        raise UnsupportedEncodingError(encoding=encoding, filePath=filePath)
