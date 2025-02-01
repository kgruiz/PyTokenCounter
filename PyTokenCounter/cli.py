# cli.py

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
    -M, --mapTokens      Output mapped tokens instead of raw token integers.
    -o, --output         Specify an output JSON file to save the results.

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
logger.propagate = False  # Prevent logs from being propagated to the root logger

if not logger.handlers:
    # Define log format with color support
    logFormat = (
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - "
        "%(funcName)s:%(lineno)d - %(message)s"
    )
    dateFormat = "%Y-%m-%d %H:%M:%S"

    # Define color scheme for different log levels
    colorScheme = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    # Create ColoredFormatter
    formatter = ColoredFormatter(
        logFormat,
        datefmt=dateFormat,
        log_colors=colorScheme,
        reset=True,
        style="%",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(console_handler)


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    Custom formatter to combine ArgumentDefaultsHelpFormatter and RawTextHelpFormatter.
    This allows for multiline help messages and inclusion of default values.
    """

    pass


def FormatChoices(choices: list[str]) -> str:
    """
    Formats a list of choices into a bulleted list.

    Parameters
    ----------
    choices : list[str]
        The list of choices to format.

    Returns
    -------
    str
        A formatted string with each choice on a new line, preceded by a bullet.
    """
    return "\n".join(f"  - {choice}" for choice in choices)


def AddCommonArgs(subParser: argparse.ArgumentParser) -> None:
    """
    Adds common arguments to a subparser.

    Parameters
    ----------
    subParser : argparse.ArgumentParser
        The subparser to which the arguments will be added.
    """
    model_help = (
        "Model to use for encoding.\nValid options are:\n"
        + FormatChoices(VALID_MODELS)
        + "\n(default: gpt-4o)"
    )
    encoding_help = "Encoding to use directly.\nValid options are:\n" + FormatChoices(
        VALID_ENCODINGS
    )

    subParser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=VALID_MODELS,
        metavar="MODEL",
        help=model_help,
        default="gpt-4o",  # Set default model here
    )
    subParser.add_argument(
        "-e",
        "--encoding",
        type=str,
        choices=VALID_ENCODINGS,
        metavar="ENCODING",
        help=encoding_help,
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
        help="Output mapped tokens instead of raw token integers.",
    )
    subParser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="OUTPUT_FILE",
        help="Specify an output JSON file to save the results.",
    )


def ParseFiles(file_args: list[str]) -> list[str]:
    """
    Parses a list of file arguments, allowing for comma-separated values.

    Parameters
    ----------
    file_args : list[str]
        The raw file arguments from the command line.

    Returns
    -------
    list[str]
        The list of parsed file paths.

    Raises
    ------
    ValueError
        If any file path is invalid or does not exist.
    """

    files = []

    for arg in file_args:

        # Split each argument by commas
        parts = arg.split(",")

        for part in parts:

            part = part.strip()

            if part:

                path = Path(part)

                if not path.exists():

                    raise ValueError(f"File or directory '{part}' does not exist.")

                files.append(str(path))

    return files


def ParseTokens(token_args: list[str]) -> list[int]:
    """
    Parses a list of token arguments, allowing for comma-separated values.

    Parameters
    ----------
    token_args : list[str]
        The raw token arguments from the command line.

    Returns
    -------
    list[int]
        The list of parsed integer tokens.

    Raises
    ------
    ValueError
        If any token cannot be converted to an integer.
    """

    tokens = []

    for arg in token_args:

        # Split each argument by commas
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
    Saves the provided data to a JSON file.

    Parameters
    ----------
    data : any
        The data to save.
    outputFile : str
        The path to the output JSON file.

    Raises
    ------
    IOError
        If the file cannot be written.
    """

    try:

        with open(outputFile, "w", encoding="utf-8") as f:

            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Output successfully saved to '{outputFile}'.")

    except IOError as e:

        raise IOError(f"Failed to write to '{outputFile}': {e}")


def main() -> None:
    """
    Entry point for the CLI. Parses command-line arguments and invokes the appropriate
    tokenization or counting functions based on the provided subcommand.

    Raises
    ------
    SystemExit
        Exits the program with a status code of 1 if an error occurs.
    """

    parser = argparse.ArgumentParser(
        description="Tokenize strings, files, or directories using specified models or encodings.",
        formatter_class=CustomFormatter,
    )

    subParsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Subparser for tokenizing a string
    parserTokenizeStr = subParsers.add_parser(
        "tokenize-str",
        help="Tokenize a provided string.",
        description="Tokenize a given string into a list of token IDs or mapped tokens using the specified model or encoding.",
        formatter_class=CustomFormatter,
    )
    AddCommonArgs(parserTokenizeStr)
    parserTokenizeStr.add_argument("string", type=str, help="The string to tokenize.")

    # Subparser for tokenizing a file
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

    # Subparser for tokenizing multiple files or a directory
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

    # Subparser for tokenizing a directory
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

    # Subparser for counting tokens in a string
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

    # Subparser for counting tokens in a file
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

    # Subparser for counting tokens in multiple files or a directory
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

    # Subparser for counting tokens in a directory
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

    # Subparser for getting the model from an encoding
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

    # Subparser for getting the encoding from a model
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

    # Subparser for mapping tokens
    parserMapTokens = subParsers.add_parser(
        "map-tokens",
        help="Map a list of token integers to their decoded strings.",
        description="Map a provided list of integer tokens to their corresponding decoded strings using the specified model or encoding.",
        formatter_class=CustomFormatter,
    )
    # Common arguments for encoding/model
    AddCommonArgs(parserMapTokens)
    # Positional arguments: list of token integers (allow comma-separated)
    parserMapTokens.add_argument(
        "tokens",
        type=str,
        nargs="+",
        help="List of integer tokens to map. Tokens can be separated by spaces or commas.",
    )

    # Parse the arguments

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
            encoding = GetEncoding(model="gpt-4o")  # Default model

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
            )

            if args.output:
                SaveOutput(tokens, args.output)
            else:
                print(json.dumps(tokens, ensure_ascii=False, indent=4))

        elif args.command == "tokenize-files":

            # Handle both multiple files and directory
            # Split inputs by commas and flatten the list
            inputPaths = ParseFiles(args.input)

            # Check if the input is a single directory
            if len(inputPaths) == 1 and Path(inputPaths[0]).is_dir():

                tokenLists = TokenizeFiles(
                    inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
                )

            else:

                tokenLists = TokenizeFiles(
                    inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    quiet=args.quiet,
                    mapTokens=args.mapTokens,
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
            )

            print(count)

        elif args.command == "count-files":

            # Split inputs by commas and flatten the list
            inputPaths = ParseFiles(args.input)

            if len(inputPaths) == 1 and Path(inputPaths[0]).is_dir():

                totalCount = GetNumTokenFiles(
                    inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                    quiet=args.quiet,
                )

            else:

                totalCount = GetNumTokenFiles(
                    inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    quiet=args.quiet,
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
            )

            print(count)

        elif args.command == "get-model":
            modelName = GetModelForEncodingName(encodingName=args.encoding)
            print(modelName)

        elif args.command == "get-encoding":
            encodingName = GetEncoding(model=args.model).name
            print(encodingName)

        elif args.command == "map-tokens":

            # Parse tokens allowing for comma-separated inputs
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
