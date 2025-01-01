# PyTokenCounter/__init__.py

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
