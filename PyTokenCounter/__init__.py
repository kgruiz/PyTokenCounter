# PyTokenCounter/__init__.py

from PyTokenCounter._utils import UnsupportedEncodingError
from PyTokenCounter.core import (
    GetEncoding,
    GetEncodingForModel,
    GetModelForEncoding,
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
    "GetValidEncodings",
    "GetModelForEncoding",
    "GetEncodingForModel",
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
