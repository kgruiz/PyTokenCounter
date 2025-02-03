# PyTokenCounter/__init__.py

"""
PyTokenCounter
==============

A package for tokenizing and counting tokens in strings, files, and directories
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
- ``TokenizeStr`` : Tokenize a single string into token IDs.
- ``GetNumTokenStr`` : Count the number of tokens in a string.
- ``TokenizeFile`` : Tokenize the contents of a file into token IDs.
- ``GetNumTokenFile`` : Count the number of tokens in a file.
- ``TokenizeFiles`` : Tokenize multiple files or a directory into token IDs.
- ``GetNumTokenFiles`` : Count the number of tokens across multiple files or in a directory.
- ``TokenizeDir`` : Tokenize all files within a directory.
- ``GetNumTokenDir`` : Count the number of tokens within a directory.
"""

from PyTokenCounter._utils import UnsupportedEncodingError
from PyTokenCounter.core import (
    GetEncoding,
    GetEncodingForModel,
    GetEncodingNameForModel,
    GetModelForEncoding,
    GetModelForEncodingName,
    GetModelMappings,
    GetNumTokenDir,
    GetNumTokenFile,
    GetNumTokenFiles,
    GetNumTokenStr,
    GetValidEncodings,
    GetValidModels,
    TokenizeDir,
    TokenizeFile,
    TokenizeFiles,
    TokenizeStr,
)

# Define the public API of the package
__all__ = [
    "GetModelMappings",
    "GetValidModels",
    "GetEncodingForModel",
    "GetValidEncodings",
    "GetModelForEncodingName",
    "GetModelForEncoding",
    "GetEncodingNameForModel",
    "GetEncoding",
    "TokenizeStr",
    "GetNumTokenStr",
    "TokenizeFile",
    "GetNumTokenFile",
    "TokenizeFiles",
    "GetNumTokenFiles",
    "TokenizeDir",
    "GetNumTokenDir",
    "UnsupportedEncodingError",
]
