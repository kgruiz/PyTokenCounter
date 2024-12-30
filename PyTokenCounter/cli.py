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

For detailed help on each subcommand, use:
    tokencount <subcommand> -h

Example:
    tokencount tokenize-str "Hello, world!" -m gpt-4
    tokencount tokenize-files ./file1.txt ./file2.txt -m gpt-4
    tokencount tokenize-files ./my_directory -m gpt-4 -nr
    tokencount tokenize-dir ./my_directory -m gpt-4 -nr
    tokencount count-files ./my_directory -m gpt-4
    tokencount count-dir ./my_directory -m gpt-4
"""

import argparse
import logging
import sys
from pathlib import Path

from .core import (
    VALID_ENCODINGS,
    VALID_MODELS,
    GetEncoding,
    GetNumTokenDir,
    GetNumTokenFiles,
    GetNumTokenStr,
    TokenizeDir,
    TokenizeFiles,
    TokenizeStr,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def AddCommonArgs(subParser: argparse.ArgumentParser) -> None:
    """
    Adds common arguments to a subparser.

    Parameters
    ----------
    subParser : argparse.ArgumentParser
        The subparser to which the arguments will be added.
    """

    subParser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=VALID_MODELS,
        help="Model to use for encoding.",
    )
    subParser.add_argument(
        "-e",
        "--encoding",
        type=str,
        choices=VALID_ENCODINGS,
        help="Encoding to use directly.",
    )


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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subParsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Subparser for tokenizing a string
    parserTokenizeStr = subParsers.add_parser(
        "tokenize-str",
        help="Tokenize a provided string.",
        description="Tokenize a given string into a list of token IDs using the specified model or encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    AddCommonArgs(parserTokenizeStr)
    parserTokenizeStr.add_argument("string", type=str, help="The string to tokenize.")

    # Subparser for tokenizing a file
    parserTokenizeFile = subParsers.add_parser(
        "tokenize-file",
        help="Tokenize the contents of a file.",
        description="Tokenize the contents of a specified file into a list of token IDs using the given model or encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    AddCommonArgs(parserTokenizeFile)
    parserTokenizeFile.add_argument(
        "file",
        type=str,
        help="Path to the file to tokenize.",
    )

    # Subparser for tokenizing multiple files or a directory
    parserTokenizeFiles = subParsers.add_parser(
        "tokenize-files",
        help="Tokenize the contents of multiple files or a directory.",
        description="Tokenize the contents of multiple specified files or all files within a directory into lists of token IDs using the given model or encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    AddCommonArgs(parserTokenizeFiles)
    parserTokenizeFiles.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Paths to the files to tokenize or a directory path.",
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
        description="Tokenize all files within a specified directory into lists of token IDs using the chosen model or encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    AddCommonArgs(parserCountFile)
    parserCountFile.add_argument(
        "file",
        type=str,
        help="Path to the file to count tokens for.",
    )

    # Subparser for counting tokens in multiple files or a directory
    parserCountFiles = subParsers.add_parser(
        "count-files",
        help="Count tokens in multiple files or a directory.",
        description="Count the number of tokens in multiple specified files or all files within a directory using the given model or encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    AddCommonArgs(parserCountFiles)
    parserCountFiles.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Paths to the files to count tokens for or a directory path.",
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Parse the arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:

        encoding = GetEncoding(model=args.model, encodingName=args.encoding)

        if args.command == "tokenize-str":

            tokens = TokenizeStr(
                string=args.string,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
            )
            print(tokens)

        elif args.command == "tokenize-file":
            tokens = TokenizeFiles(
                inputPath=args.file,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
            )

            print(tokens)

        elif args.command == "tokenize-files":
            # Handle both multiple files and directory
            inputPaths = [Path(p) for p in args.input]
            if len(inputPaths) == 1 and inputPaths[0].is_dir():
                tokenLists = TokenizeFiles(
                    inputPath=inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                )
            else:
                tokenLists = TokenizeFiles(
                    inputPath=inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                )
            print(tokenLists)

        elif args.command == "tokenize-dir":
            tokenizedDir = TokenizeDir(
                dirPath=args.directory,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                recursive=not args.no_recursive,
            )

            print(tokenizedDir)

        elif args.command == "count-str":

            count = GetNumTokenStr(
                string=args.string,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
            )
            print(count)

        elif args.command == "count-file":

            count = GetNumTokenFiles(
                inputPath=args.file,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
            )

            print(count)
        elif args.command == "count-files":

            inputPaths = [Path(p) for p in args.input]

            if len(inputPaths) == 1 and inputPaths[0].is_dir():
                totalCount = GetNumTokenFiles(
                    inputPath=inputPaths[0],
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                    recursive=not args.no_recursive,
                )
            else:
                totalCount = GetNumTokenFiles(
                    inputPath=inputPaths,
                    model=args.model,
                    encodingName=args.encoding,
                    encoding=encoding,
                )
            print(totalCount)

        elif args.command == "count-dir":

            count = GetNumTokenDir(
                dirPath=args.directory,
                model=args.model,
                encodingName=args.encoding,
                encoding=encoding,
                recursive=not args.no_recursive,
            )

            print(count)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":

    main()
