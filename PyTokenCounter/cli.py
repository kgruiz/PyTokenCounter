# PyTokenCounter/cli.py

"""
CLI Module for PyTokenCounter
=============================

This module provides a Command-Line Interface (CLI) for tokenizing strings, files, and directories
using specified models or encodings. It leverages the functionality defined in the "core.py" module.

Usage:
    After installing the package, use the "tokencount" command followed by the desired subcommand
    and options.

Subcommands:
    tokenize-str   Tokenize a provided string.
    tokenize-file  Tokenize the contents of a file.
    tokenize-files Tokenize the contents of multiple files or a directory.
    tokenize-dir   Tokenize all files in a directory.
    count-str      Count tokens in a provided string.
    count-file     Count tokens in a file.
    count-files    Count tokens in multiple files or a directory.
    count-dir      Count tokens in all files within a directory.
    get-model      Retrieves the model name from the provided encoding.
    get-encoding   Retrieves the encoding name from the provided model.
    map-tokens     Map a list of token integers to their decoded strings.

Options:
    -m, --model          Model to use for encoding. (default: gpt-4o)
    -e, --encoding       Encoding to use directly.
    -nr, --no-recursive  Do not tokenize files in subdirectories if a directory is given.
    -q, --quiet          Silence progress bars and minimize output.
    -M, --mapTokens      Output mapped tokens (or nested structure with 'numTokens' and 'tokens' keys) instead of raw token integers.
    -o, --output         Specify an output JSON file to save the results.
    -b, --include-binary Include binary files in processing. (default: binary files are excluded)
    -H, --include-hidden Include hidden files and directories in processing. (default: hidden files and directories are skipped)

For detailed help on each subcommand, use:

    tokencount <subcommand> -h

Example:
    tokencount tokenize-str "Hello, world!" -m gpt-4o
    tokencount tokenize-files ./file1.txt ./file2.txt -m gpt-4o
    tokencount tokenize-files ./myDirectory -m gpt-4o -nr
    tokencount tokenize-dir ./myDirectory -m gpt-4o -nr
    tokencount count-files ./myDirectory -m gpt-4o
    tokencount count-dir ./myDirectory -m gpt-4o
    tokencount get-model cl100k_base
    tokencount get-encoding gpt-4o
    tokencount map-tokens 123 456 789 -m gpt-4o
    tokencount map-tokens 123,456,789 -m gpt-4o
    tokencount map-tokens 123,456 789 -m gpt-4o
    tokencount tokenize-files ./file1.txt,./file2.txt -m gpt-4o -o tokens.json
    tokencount map-tokens 123,456,789 -m gpt-4o -o mappedTokens.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from colorlog import ColoredFormatter

from .core import (
    VALID_ENCODINGS,
    VALID_MODELS,
    GetEncoding,
    GetModelForEncodingName,
    GetNumTokenDir,
    GetNumTokenFiles,
    GetNumTokenStr,
    MapTokens,
    TokenizeDir,
    TokenizeFiles,
    TokenizeStr,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation


if not logger.handlers:

    # Log format
    logFormat = (
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - "
        "%(funcName)s:%(lineno)d - %(message)s"
    )
    dateFormat = "%Y-%m-%d %H:%M:%S"

    # Color scheme
    colorScheme = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    # Formatter

    formatter = ColoredFormatter(
        logFormat,
        datefmt=dateFormat,
        log_colors=colorScheme,
        reset=True,
        style="%",
    )

    # Console handler
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)

    # Add handler
    logger.addHandler(consoleHandler)


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    Custom formatter combining defaults and raw text.

    Parameters
    ----------
    None
    """

    pass


def FormatChoices(choices: list[str]) -> str:
    """
    Format a list of choices into multiple rows and columns.

    Parameters
    ----------
    choices : list[str]
        List of choices.

    Returns
    -------
    str
        Formatted string with choices in columns.
    """
    numColumns = 4 if len(choices) >= 4 else len(choices)
    rows = [choices[i : i + numColumns] for i in range(0, len(choices), numColumns)]
    colWidths = [0] * numColumns

    for row in rows:

        for colIndex, item in enumerate(row):

            itemWidth = len("- " + item)

            if itemWidth > colWidths[colIndex]:

                colWidths[colIndex] = itemWidth

    formattedRows = []

    for row in rows:

        formattedItems = []

        for colIndex, item in enumerate(row):

            text = f"- {item}"

            if colIndex < numColumns - 1:

                formattedItems.append(text.ljust(colWidths[colIndex] + 2))

            else:

                formattedItems.append(text)

        formattedRows.append("".join(formattedItems))

    return "\n".join(formattedRows)


def AddCommonArgs(subParser: argparse.ArgumentParser) -> None:
    """
    Add common arguments to a subparser.

    Parameters
    ----------
    subParser : argparse.ArgumentParser
        Subparser object.
    """
    modelHelp = (
        "Model to use for encoding.\nValid options are:\n"
        + FormatChoices(VALID_MODELS)
        + "\n(default: gpt-4o)"
    )
    encodingHelp = "Encoding to use directly.\nValid options are:\n" + FormatChoices(
        VALID_ENCODINGS
    )
    subParser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=VALID_MODELS,
        metavar="MODEL",
        help=modelHelp,
        default="gpt-4o",
    )
    subParser.add_argument(
        "-e",
        "--encoding",
        type=str,
        choices=VALID_ENCODINGS,
        metavar="ENCODING",
        help=encodingHelp,
    )
    subParser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Silence progress bars and minimize output.",
    )
    subParser.add_argument(
        "-M",
        "--mapTokens",
        action="store_true",
        help="Output mapped tokens (or nested structure with 'numTokens' and 'tokens' keys) instead of raw token integers.",
    )
    subParser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="OUTPUT_FILE",
        help="Specify an output JSON file to save the results.",
    )
    subParser.add_argument(
        "-b",
        "--include-binary",
        action="store_true",
        dest="includeBinary",
        default=False,
        help="Include binary files in processing. (Default: binary files are excluded.)",
    )
    subParser.add_argument(
        "-H",
        "--include-hidden",
        action="store_true",
        dest="includeHidden",
        default=False,
        help="Include hidden files and directories in processing. (Default: hidden files and directories are skipped.)",
    )


def ParseFiles(fileArgs: list[str]) -> list[str]:
    """
    Parse file arguments, allowing comma-separated values.

    Parameters
    ----------
    fileArgs : list[str]
        Raw file arguments.

    Returns
    -------
    list[str]
        List of file paths.

    Raises
    ------
    ValueError
        If a path does not exist.
    """
    files = []

    for arg in fileArgs:

        parts = arg.split(",")

        for part in parts:

            part = part.strip()

            if part:

                path = Path(part)

                if not path.exists():

                    raise ValueError(f"File or directory '{part}' does not exist.")

                files.append(str(path))

    return files


def ParseTokens(tokenArgs: list[str]) -> list[int]:
    """
    Parse token arguments, allowing comma-separated values.

    Parameters
    ----------
    tokenArgs : list[str]
        Raw token arguments.

    Returns
    -------
    list[int]
        List of integer tokens.

    Raises
    ------
    ValueError
        If a token is not an integer.
    """
    tokens = []

    for arg in tokenArgs:

        parts = arg.split(",")

        for part in parts:

            part = part.strip()

            if part:

                try:

                    token = int(part)
                    tokens.append(token)

                except ValueError:

                    raise ValueError(
                        f"Invalid token '{part}'. Tokens must be integers."
                    )

    return tokens


def SaveOutput(data: any, outputFile: str) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    data : any
        Data to save.
    outputFile : str
        Output file path.

    Raises
    ------
    IOError
        If writing fails.
    """

    try:

        with open(outputFile, "w", encoding="utf-8") as f:

            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Output successfully saved to '{outputFile}'.")

    except IOError as e:

        raise IOError(f"Failed to write to '{outputFile}': {e}")


def main() -> None:
    """
    CLI entry point. Parses arguments and calls tokenization/counting functions.

    Raises
    ------
    SystemExit
        If an error occurs.
    """
    parser = argparse.ArgumentParser(
        description="Tokenize strings, files, or directories using specified models or encodings.",
        formatter_class=CustomFormatter,
    )
    subParsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Tokenize string
    parserTokenizeStr = subParsers.add_parser(
        "tokenize-str",
        help="Tokenize a provided string.",
        description="Tokenize a given string into a list of token IDs or mapped tokens using the specified model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserTokenizeStr)
    parserTokenizeStr.add_argument("string", type=str, help="The string to tokenize.")

    # Tokenize file
    parserTokenizeFile = subParsers.add_parser(
        "tokenize-file",
        help="Tokenize the contents of a file.",
        description="Tokenize the contents of a specified file into a list of token IDs or mapped tokens using the given model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserTokenizeFile)
    parserTokenizeFile.add_argument(
        "file",
        type=str,
        help="Path to the file to tokenize. Multiple files can be separated by commas.",
    )

    # Tokenize multiple files or directory
    parserTokenizeFiles = subParsers.add_parser(
        "tokenize-files",
        help="Tokenize the contents of multiple files or a directory.",
        description="Tokenize the contents of multiple specified files or all files within a directory into lists of token IDs or mapped tokens using the given model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserTokenizeFiles)
    parserTokenizeFiles.add_argument(
        "input",
        type=str,
        nargs="+",
        help="""\
Paths to the files to tokenize or a directory path.
Multiple files can be separated by spaces or commas.
""",
    )
    parserTokenizeFiles.add_argument(
        "-nr",
        "--no-recursive",
        action="store_true",
        help="Do not tokenize files in subdirectories if a directory is given.",
    )

    # Tokenize directory
    parserTokenizeDir = subParsers.add_parser(
        "tokenize-dir",
        help="Tokenize all files in a directory.",
        description="Tokenize all files within a specified directory into lists of token IDs or mapped tokens using the chosen model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserTokenizeDir)
    parserTokenizeDir.add_argument(
        "directory",
        type=str,
        help="Path to the directory to tokenize.",
    )
    parserTokenizeDir.add_argument(
        "-nr",
        "--no-recursive",
        action="store_true",
        help="Do not tokenize files in subdirectories.",
    )

    # Count tokens in string
    parserCountStr = subParsers.add_parser(
        "count-str",
        help="Count tokens in a provided string.",
        description="Count the number of tokens in a given string using the specified model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserCountStr)
    parserCountStr.add_argument(
        "string", type=str, help="The string to count tokens for."
    )

    # Count tokens in file
    parserCountFile = subParsers.add_parser(
        "count-file",
        help="Count tokens in a file.",
        description="Count the number of tokens in a specified file using the given model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserCountFile)
    parserCountFile.add_argument(
        "file",
        type=str,
        help="Path to the file to count tokens for. Multiple files can be separated by commas.",
    )

    # Count tokens in multiple files or directory
    parserCountFiles = subParsers.add_parser(
        "count-files",
        help="Count tokens in multiple files or a directory.",
        description="Count the number of tokens in multiple specified files or all files within a directory using the given model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserCountFiles)
    parserCountFiles.add_argument(
        "input",
        type=str,
        nargs="+",
        help="""\
Paths to the files to count tokens for or a directory path.
Multiple files can be separated by spaces or commas.
""",
    )
    parserCountFiles.add_argument(
        "-nr",
        "--no-recursive",
        action="store_true",
        help="Do not count tokens in subdirectories if a directory is given.",
    )

    # Count tokens in directory
    parserCountDir = subParsers.add_parser(
        "count-dir",
        help="Count tokens in all files within a directory.",
        description="Count the total number of tokens across all files in a specified directory using the chosen model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserCountDir)
    parserCountDir.add_argument(
        "directory",
        type=str,
        help="Path to the directory to count tokens for.",
    )
    parserCountDir.add_argument(
        "-nr",
        "--no-recursive",
        action="store_true",
        help="Do not count tokens in subdirectories.",
    )

    # Get model from encoding
    parserGetModel = subParsers.add_parser(
        "get-model",
        help="Retrieves the model name from the provided encoding.",
        description="Retrieves the model name(s) associated with the specified encoding.",
        formatter_class=CustomFormatter,
    )
    parserGetModel.add_argument(
        "encoding",
        type=str,
        choices=VALID_ENCODINGS,
        metavar="ENCODING",
        help="Encoding to get the model for.\nValid options are:\n"
        + FormatChoices(VALID_ENCODINGS),
    )

    # Get encoding from model
    parserGetEncoding = subParsers.add_parser(
        "get-encoding",
        help="Retrieves the encoding name from the provided model.",
        description="Retrieves the encoding name associated with the specified model.",
        formatter_class=CustomFormatter,
    )
    parserGetEncoding.add_argument(
        "model",
        type=str,
        choices=VALID_MODELS,
        metavar="MODEL",
        help="Model to get the encoding for.\nValid options are:\n"
        + FormatChoices(VALID_MODELS),
    )

    # Map tokens
    parserMapTokens = subParsers.add_parser(
        "map-tokens",
        help="Map a list of token integers to their decoded strings.",
        description="Map a provided list of integer tokens to their corresponding decoded strings using the specified model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserMapTokens)
    parserMapTokens.add_argument(
        "tokens",
        type=str,
        nargs="+",
        help="List of integer tokens to map. Tokens can be separated by spaces or commas.",
    )

    if len(sys.argv) == 1:

        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:

        encoding = None

        if args.model and args.encoding:

            encoding = GetEncoding(model=args.model, encodingName=args.encoding)

        elif args.model:

            encoding = GetEncoding(model=args.model)

        elif args.encoding:

            encoding = GetEncoding(encodingName=args.encoding)

        else:

            encoding = GetEncoding(model="gpt-4o")

        if args.command == "tokenize-str":

            tokens = TokenizeStr(
                string=args.string,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                quiet=args.quiet,
                mapTokens=args.mapTokens,
            )

            if args.output:

                SaveOutput(tokens, args.output)

            else:

                print(json.dumps(tokens, ensure_ascii=False, indent=4))

        elif args.command == "tokenize-file":

            files = ParseFiles([args.file])
            tokens = TokenizeFiles(
                files,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                quiet=args.quiet,
                mapTokens=args.mapTokens,
                excludeBinary=not args.includeBinary,
                includeHidden=args.includeHidden,
            )

            if args.output:

                SaveOutput(tokens, args.output)

            else:

                print(json.dumps(tokens, ensure_ascii=False, indent=4))

        elif args.command == "tokenize-files":

            inputPaths = ParseFiles(args.input)

            if len(inputPaths) == 1 and Path(inputPaths[0]).is_dir():

                tokenLists = TokenizeFiles(
                    inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
                    excludeBinary=not args.includeBinary,
                    includeHidden=args.includeHidden,
                )

            else:

                tokenLists = TokenizeFiles(
                    inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
                    excludeBinary=not args.includeBinary,
                    includeHidden=args.includeHidden,
                )

            if args.output:

                SaveOutput(tokenLists, args.output)

            else:

                print(json.dumps(tokenLists, ensure_ascii=False, indent=4))

        elif args.command == "tokenize-dir":

            tokenizedDir = TokenizeDir(
                dirPath=args.directory,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                recursive=not args.no_recursive,
                quiet=args.quiet,
                mapTokens=args.mapTokens,
                excludeBinary=not args.includeBinary,
                includeHidden=args.includeHidden,
            )

            if args.output:

                SaveOutput(tokenizedDir, args.output)

            else:

                print(json.dumps(tokenizedDir, ensure_ascii=False, indent=4))

        elif args.command == "count-str":

            count = GetNumTokenStr(
                string=args.string,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                quiet=args.quiet,
            )
            print(count)

        elif args.command == "count-file":

            files = ParseFiles([args.file])
            count = GetNumTokenFiles(
                files,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                quiet=args.quiet,
                mapTokens=args.mapTokens,
                excludeBinary=not args.includeBinary,
                includeHidden=args.includeHidden,
            )
            print(count)

        elif args.command == "count-files":

            inputPaths = ParseFiles(args.input)

            if len(inputPaths) == 1 and Path(inputPaths[0]).is_dir():

                totalCount = GetNumTokenFiles(
                    inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
                    excludeBinary=not args.includeBinary,
                    includeHidden=args.includeHidden,
                )

            else:

                totalCount = GetNumTokenFiles(
                    inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
                    excludeBinary=not args.includeBinary,
                    includeHidden=args.includeHidden,
                )
            print(totalCount)

        elif args.command == "count-dir":

            count = GetNumTokenDir(
                dirPath=args.directory,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                recursive=not args.no_recursive,
                quiet=args.quiet,
                mapTokens=args.mapTokens,
                excludeBinary=not args.includeBinary,
                includeHidden=args.includeHidden,
            )
            print(count)

        elif args.command == "get-model":

            modelName = GetModelForEncodingName(encodingName=args.encoding)
            print(modelName)

        elif args.command == "get-encoding":

            encodingName = GetEncoding(model=args.model).name
            print(encodingName)

        elif args.command == "map-tokens":

            tokens = ParseTokens(args.tokens)
            mapped = MapTokens(
                tokens,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
            )

            if args.output:

                SaveOutput(mapped, args.output)

            else:

                print(json.dumps(mapped, ensure_ascii=False, indent=4))

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":

    main()
