import inspect
import json
import os
from pathlib import Path

import tiktoken

import PyTokenCounter as tc

# Define the paths to the test input and answers directories
testInputDir = Path(
    "/Users/kadengruizenga/Development/Packages/Python/PyTokenCounter/Tests/Input"
)
testAnswersDir = Path(
    "/Users/kadengruizenga/Development/Packages/Python/PyTokenCounter/Tests/Answers"
)


def RaiseTestAssertion(message: str):
    """
    Helper function to raise AssertionError with test suite file and line number.
    """
    currentFrame = inspect.currentframe()
    frameInfo = inspect.getframeinfo(currentFrame.f_back)
    fileName = Path(frameInfo.filename).resolve()
    lineNo = frameInfo.lineno
    fullMessage = f"{fileName}:{lineNo} -\n{message}"
    raise AssertionError(fullMessage)


def compareTokenDicts(expected, actual, path=""):
    """
    Recursively compare two nested dictionaries containing token lists.

    Parameters
    ----------
    expected : dict
        The expected token dictionary. Keys are filenames with extensions or subdirectory names.
        If the key is a filename, its value is a dict with 'numTokens' and 'tokens'.
        If the key is a subdirectory, its value is another dict following the same structure.
    actual : dict
        The actual token dictionary returned by the tokenization function.
        Structure mirrors that of 'expected'.
    path : str, optional
        The current path in the directory structure for error reporting.

    Raises
    ------
    AssertionError
        If any mismatch is found between expected and actual tokens.
    """
    for key in expected:
        expectedPath = f"{path}/{key}" if path else key
        if key not in actual:
            RaiseTestAssertion(f"Missing key '{expectedPath}' in actual tokenization.")

        expectedEntry = expected[key]
        actualEntry = actual[key]

        if isinstance(expectedEntry, dict):
            if "tokens" in expectedEntry:
                # It's a file
                expectedTokens = expectedEntry["tokens"]
                actualTokens = actualEntry

                if not isinstance(actualTokens, list):
                    RaiseTestAssertion(
                        f"Expected '{expectedPath}' to be a list of tokens, but got {type(actualTokens).__name__}."
                    )

                if expectedTokens != actualTokens:
                    incorrectTokens = {}
                    min_length = min(len(expectedTokens), len(actualTokens))
                    for i in range(min_length):
                        if expectedTokens[i] != actualTokens[i]:
                            incorrectTokens[f"{i}"] = {
                                "actual": actualTokens[i],
                                "expected": expectedTokens[i],
                            }

                    # Check for extra tokens
                    if len(expectedTokens) != len(actualTokens):
                        if len(expectedTokens) > len(actualTokens):
                            for i in range(min_length, len(expectedTokens)):
                                incorrectTokens[f"{i}"] = {
                                    "actual": None,
                                    "expected": expectedTokens[i],
                                }
                        else:
                            for i in range(min_length, len(actualTokens)):
                                incorrectTokens[f"{i}"] = {
                                    "actual": actualTokens[i],
                                    "expected": None,
                                }

                    RaiseTestAssertion(
                        f"Tokenization mismatch for file '{expectedPath}'.\n"
                        f"Incorrect Tokens: {json.dumps(incorrectTokens, indent=4)}"
                    )
            else:
                # It's a subdirectory
                if not isinstance(actualEntry, dict):
                    RaiseTestAssertion(
                        f"Expected '{expectedPath}' to be a directory, but got {type(actualEntry).__name__}."
                    )
                # Recurse into the subdirectory
                compareTokenDicts(expectedEntry, actualEntry, path=expectedPath)
        else:
            RaiseTestAssertion(
                f"Unexpected structure for key '{expectedPath}' in expected data."
            )

    # Check for unexpected keys in actual
    for key in actual:
        expectedPath = f"{path}/{key}" if path else key
        if key not in expected:
            RaiseTestAssertion(
                f"Unexpected key '{expectedPath}' found in actual tokenization."
            )


def TestTokenizeDirectory():
    """
    Test tokenization of a directory with multiple files and subdirectories.
    """

    dirPath = Path(testInputDir, "TestDirectory")
    answerPath = Path(testAnswersDir, "TestDirectory.json")

    # Load expected results
    with answerPath.open("r") as file:
        expected = json.load(file)

    tokenizedDir = tc.TokenizeDir(
        dirPath=dirPath,
        model="gpt-4o",
        recursive=True,
        quiet=True,
    )

    if not isinstance(tokenizedDir, dict):
        RaiseTestAssertion(
            f"Expected TokenizeDir to return a dict for directory '{dirPath}', got {type(tokenizedDir).__name__}."
        )

    compareTokenDicts(expected, tokenizedDir)


def TestTokenizeFilesWithDirectory():
    """
    Test tokenization of a directory using TokenizeFiles function.
    """
    dirPath = Path(testInputDir, "TestDirectory")
    answerPath = Path(testAnswersDir, "TestDirectory.json")

    # Load expected results
    with answerPath.open("r") as file:
        expected = json.load(file)

    tokenizedFiles = tc.TokenizeFiles(dirPath, model="gpt-4o", quiet=True)

    if not isinstance(tokenizedFiles, dict):
        RaiseTestAssertion(
            f"Expected TokenizeFiles to return a dict for directory '{dirPath}', got {type(tokenizedFiles).__name__}."
        )

    compareTokenDicts(expected, tokenizedFiles)


def TestTokenizeFilesMultiple():
    """
    Test tokenization of multiple files using TokenizeFiles function.
    """
    inputFiles = [
        Path(testInputDir, "TestFile1.txt"),
        Path(testInputDir, "TestFile2.txt"),
    ]
    answerFiles = [
        Path(testAnswersDir, "TestFile1.json"),
        Path(testAnswersDir, "TestFile2.json"),
    ]

    expectedTokenLists = {}
    for inputFile, answerFile in zip(inputFiles, answerFiles):
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists[inputFile.name] = answer["tokens"]

    tokenizedFiles = tc.TokenizeFiles(inputFiles, model="gpt-4o", quiet=True)

    if not isinstance(tokenizedFiles, dict):
        RaiseTestAssertion(
            f"Expected TokenizeFiles to return a dict for list of files, got {type(tokenizedFiles).__name__}."
        )

    # Build expected structure
    expectedStructure = {
        key: {"tokens": value} for key, value in expectedTokenLists.items()
    }

    compareTokenDicts(expectedStructure, tokenizedFiles)


def TestTokenizeFilesExitOnListErrorFalse():
    """
    Test TokenizeFiles function with exitOnListError=False to ensure it continues on encountering errors.
    """
    inputList = [
        Path(testInputDir, "TestFile1.txt"),
        Path(
            testInputDir, "TestImg.jpg"
        ),  # Assuming this file has unsupported encoding
        Path(testInputDir, "TestFile2.txt"),
    ]

    answerFiles = [
        Path(testAnswersDir, "TestFile1.json"),
        Path(testAnswersDir, "TestFile2.json"),
    ]

    expectedTokenLists = {}
    for inputFile, answerFile in zip([inputList[0], inputList[2]], answerFiles):
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists[inputFile.name] = answer["tokens"]

    tokenizedFiles = tc.TokenizeFiles(
        inputList, model="gpt-4o", quiet=True, exitOnListError=False
    )

    if not isinstance(tokenizedFiles, dict):
        RaiseTestAssertion(
            f"Expected TokenizeFiles to return a dict when exitOnListError=False, got {type(tokenizedFiles).__name__}."
        )

    # Build expected structure (only successful files)
    expectedStructure = {
        key: {"tokens": value} for key, value in expectedTokenLists.items()
    }

    compareTokenDicts(expectedStructure, tokenizedFiles)


def TestTokenizeDirectoryNoRecursion():
    """
    Test tokenization of a directory without recursion to ensure subdirectories are not tokenized.
    """
    dirPath = Path(testInputDir, "TestDirectory")
    answerPath = Path(testAnswersDir, "TestDirectoryNoRecursion.json")

    # Load expected results (only top-level files)
    with answerPath.open("r") as file:
        expected = json.load(file)

    tokenizedDir = tc.TokenizeDir(
        dirPath=dirPath,
        model="gpt-4o",
        recursive=False,
        quiet=True,
    )

    if not isinstance(tokenizedDir, dict):
        RaiseTestAssertion(
            f"Expected TokenizeDir to return a dict for directory '{dirPath}', got {type(tokenizedDir).__name__}."
        )

    compareTokenDicts(expected, tokenizedDir)

    # Ensure subdirectories are not included
    for entry in dirPath.iterdir():
        if entry.is_dir():
            if entry.name in tokenizedDir:
                RaiseTestAssertion(
                    f"Subdirectory '{entry.name}' should not be tokenized when recursion is disabled."
                )


def TestTokenizeFilesWithInvalidInput():
    """
    Test TokenizeFiles function with invalid input types.
    """
    invalidInput = 67890  # Neither str, Path, nor list

    try:
        tc.TokenizeFiles(invalidInput)
    except TypeError as e:
        expectedMessage = 'Unexpected type for parameter "inputPath". Expected type: str, pathlib.Path, or list.'
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for invalid inputPath type in TokenizeFiles.\n"
                f"Provided Type: '{type(invalidInput).__name__}'\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            "Test Failed: No error was raised for invalid inputPath type in TokenizeFiles."
        )


def TestTokenizeFilesListQuietFalse():
    """
    Test TokenizeFiles function with a list of files and quiet=False to ensure progress is displayed.
    """
    inputFiles = [
        Path(testInputDir, "TestFile1.txt"),
        Path(testInputDir, "TestFile2.txt"),
    ]
    answerFiles = [
        Path(testAnswersDir, "TestFile1.json"),
        Path(testAnswersDir, "TestFile2.json"),
    ]

    expectedTokenLists = {}
    for inputFile, answerFile in zip(inputFiles, answerFiles):
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists[inputFile.name] = answer["tokens"]

    # Capture stdout to verify that progress is displayed when quiet=False
    import io
    import sys

    capturedOutput = io.StringIO()
    sysStdout = sys.stdout
    sys.stdout = capturedOutput

    try:
        tokenizedFiles = tc.TokenizeFiles(inputFiles, model="gpt-4o", quiet=False)
    finally:
        sys.stdout = sysStdout  # Restore original stdout

    # Check if any progress messages were printed
    output = capturedOutput.getvalue()
    if not output.strip():
        RaiseTestAssertion(
            "Expected progress messages to be printed when quiet=False, but no output was captured."
        )

    # Verify tokenization results
    if not isinstance(tokenizedFiles, dict):
        RaiseTestAssertion(
            f"Expected TokenizeFiles to return a dict for list of files, got {type(tokenizedFiles).__name__}."
        )

    # Build expected structure
    expectedStructure = {
        key: {"tokens": value} for key, value in expectedTokenLists.items()
    }

    compareTokenDicts(expectedStructure, tokenizedFiles)


def TestGetModelMappings():
    """
    Test retrieval of model to encoding mappings.
    """
    expectedMappings = {
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

    actualMappings = tc.GetModelMappings()

    if actualMappings != expectedMappings:
        RaiseTestAssertion(
            f"Model mappings mismatch.\n"
            f"Expected Mappings:\n{json.dumps(expectedMappings, indent=4)}\n"
            f"Actual Mappings:\n{json.dumps(actualMappings, indent=4)}"
        )


def TestGetValidModels():
    """
    Test retrieval of valid model names.
    """
    expectedModels = [
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

    actualModels = tc.GetValidModels()

    if set(actualModels) != set(expectedModels):
        missing = set(expectedModels) - set(actualModels)
        extra = set(actualModels) - set(expectedModels)
        message = "Valid models mismatch.\n"
        if missing:
            message += f"Missing Models: {missing}\n"
        if extra:
            message += f"Unexpected Models: {extra}\n"
        RaiseTestAssertion(message)


def TestGetValidEncodings():
    """
    Test retrieval of valid encoding names.
    """
    expectedEncodings = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base"]

    actualEncodings = tc.GetValidEncodings()

    if set(actualEncodings) != set(expectedEncodings):
        missing = set(expectedEncodings) - set(actualEncodings)
        extra = set(actualEncodings) - set(expectedEncodings)
        message = "Valid encodings mismatch.\n"
        if missing:
            message += f"Missing Encodings: {missing}\n"
        if extra:
            message += f"Unexpected Encodings: {extra}\n"
        RaiseTestAssertion(message)


def TestGetModelForEncoding():
    """
    Test retrieval of the correct model(s) for each encoding.
    """
    encodingModelPairs = {
        "o200k_base": ["gpt-4o", "gpt-4o-mini"],
        "cl100k_base": [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ],
        "p50k_base": ["Codex models", "text-davinci-002", "text-davinci-003"],
        "r50k_base": "GPT-3 models like davinci",
    }

    for encodingName, expectedModel in encodingModelPairs.items():
        actualModel = tc.GetModelForEncoding(encodingName=encodingName)
        if isinstance(expectedModel, list):
            if not isinstance(actualModel, list):
                RaiseTestAssertion(
                    f"Expected GetModelForEncoding('{encodingName}') to return a list, but got {type(actualModel).__name__}."
                )
            sortedExpected = sorted(expectedModel)
            sortedActual = sorted(actualModel)
            if sortedActual != sortedExpected:
                RaiseTestAssertion(
                    f"Models for encoding '{encodingName}' mismatch.\n"
                    f"Expected: {sortedExpected}\n"
                    f"Got: {sortedActual}"
                )
        elif isinstance(expectedModel, str):
            if not isinstance(actualModel, str):
                RaiseTestAssertion(
                    f"Expected GetModelForEncoding('{encodingName}') to return a string, but got {type(actualModel).__name__}."
                )
            if actualModel != expectedModel:
                RaiseTestAssertion(
                    f"Model for encoding '{encodingName}' mismatch.\n"
                    f"Expected: '{expectedModel}'\n"
                    f"Got: '{actualModel}'"
                )
        else:
            RaiseTestAssertion(
                f"Invalid expectedModel type for encoding '{encodingName}'. Must be list or string."
            )


def TestGetEncodingForModel():
    """
    Test retrieval of the correct encoding for each model.
    """
    modelEncodingPairs = {
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

    for modelName, expectedEncoding in modelEncodingPairs.items():
        actualEncoding = tc.GetEncodingForModel(modelName=modelName)
        if actualEncoding != expectedEncoding:
            RaiseTestAssertion(
                f"Encoding for model '{modelName}' mismatch.\n"
                f"Expected: '{expectedEncoding}'\n"
                f"Got: '{actualEncoding}'"
            )


def TestGetEncoding():
    """
    Test retrieval of tiktoken.Encoding based on model or encoding name.
    """
    # Test with model
    encoding = tc.GetEncoding(model="gpt-3.5-turbo")
    expectedEncodingName = "cl100k_base"
    if encoding.name != expectedEncodingName:
        RaiseTestAssertion(
            f"Encoding retrieval by model mismatch.\n"
            f"Model: 'gpt-3.5-turbo'\n"
            f"Expected: '{expectedEncodingName}'\n"
            f"Got: '{encoding.name}'"
        )

    # Test with encodingName
    encoding = tc.GetEncoding(encodingName="p50k_base")
    expectedEncodingName = "p50k_base"
    if encoding.name != expectedEncodingName:
        RaiseTestAssertion(
            f"Encoding retrieval by encodingName mismatch.\n"
            f"Encoding Name: 'p50k_base'\n"
            f"Expected: '{expectedEncodingName}'\n"
            f"Got: '{encoding.name}'"
        )

    # Test with both model and encodingName (matching)
    encoding = tc.GetEncoding(model="gpt-4-turbo", encodingName="cl100k_base")
    expectedEncodingName = "cl100k_base"
    if encoding.name != expectedEncodingName:
        RaiseTestAssertion(
            f"Encoding retrieval with both model and encodingName mismatch.\n"
            f"Model: 'gpt-4-turbo', Encoding Name: 'cl100k_base'\n"
            f"Expected: '{expectedEncodingName}'\n"
            f"Got: '{encoding.name}'"
        )

    # Test with both model and encodingName (mismatched)
    try:
        tc.GetEncoding(model="gpt-3.5-turbo", encodingName="p50k_base")
    except ValueError as e:
        expectedMessage = f"Model gpt-3.5-turbo does not have encoding name p50k_base"
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for mismatched model and encodingName.\n"
                f"Model: 'gpt-3.5-turbo', Encoding Name: 'p50k_base'\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            "Test Failed: No error was raised for mismatched model and encodingName."
        )


def TestGetEncodingError():
    """
    Test that GetEncoding raises an error when neither model nor encodingName is provided.
    """
    try:
        tc.GetEncoding()
    except ValueError as e:
        expectedMessage = "Either model or encoding must be provided."
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message when neither model nor encodingName is provided.\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            "Test Failed: No error was raised when neither model nor encodingName was provided."
        )


def TestTokenizeFileWithUnsupportedEncoding():
    """
    Test tokenization of a file with unsupported encoding.
    """
    unsupportedFilePath = Path(
        testInputDir, "TestImg.jpg"
    )  # Assuming this file has unsupported encoding

    try:
        tc.TokenizeFile(filePath=unsupportedFilePath, model="gpt-4o", quiet=True)
    except tc.UnsupportedEncodingError:
        pass  # Expected exception
    except Exception as e:
        RaiseTestAssertion(
            f"Test Failed: Unexpected error type raised for file '{unsupportedFilePath}' - {type(e).__name__}"
        )
    else:
        RaiseTestAssertion(
            f"Test Failed: No error was raised for unsupported encoding in file '{unsupportedFilePath}'."
        )


def TestTokenizeFileErrorType():
    """
    Test that TokenizeFile raises TypeError when filePath is not str or Path.
    """
    invalidFilePath = 54321  # Invalid type

    try:
        tc.TokenizeFile(filePath=invalidFilePath, model="gpt-4o", quiet=True)
    except TypeError as e:
        expectedMessage = 'Unexpected type for parameter "filePath". Expected type: str or pathlib.Path.'
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for invalid filePath type in TokenizeFile.\n"
                f"Provided Type: '{type(invalidFilePath).__name__}'\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            f"Test Failed: No error was raised for invalid filePath type '{type(invalidFilePath).__name__}' in TokenizeFile."
        )


def TestStr():
    """
    Test string tokenization.
    """

    expectedStrings = {
        "Hail to the Victors!": [39, 663, 316, 290, 16566, 914, 0],
        "2024 National Champions": [1323, 19, 6743, 40544],
        "Corum 4 Heisman": [11534, 394, 220, 19, 1679, 107107],
    }

    for string, expectedTokens in expectedStrings.items():

        actualTokens = tc.TokenizeStr(string=string, model="gpt-4o", quiet=True)

        if actualTokens != expectedTokens:
            incorrectTokens = {}

            for i, actualToken in enumerate(actualTokens):
                if i < len(expectedTokens):
                    if actualToken != expectedTokens[i]:
                        incorrectTokens[f"{i}"] = {
                            "actual": actualToken,
                            "expected": expectedTokens[i],
                        }

            RaiseTestAssertion(
                f"Tokenization mismatch for string '{string}'.\n"
                f"Compared Model: 'gpt-4o'\n"
                f"Incorrect Tokens: {json.dumps(incorrectTokens, indent=4)}"
            )

        expectedCount = len(expectedTokens)
        actualCount = tc.GetNumTokenStr(string=string, model="gpt-4o", quiet=True)
        if expectedCount != actualCount:
            RaiseTestAssertion(
                f"Token count mismatch for string '{string}'.\n"
                f"Compared Model: 'gpt-4o'\n"
                f"Expected Count: {expectedCount}, "
                f"Got Count: {actualCount}"
            )


def TestFile(inputName, answerName):
    """
    Test file tokenization.
    """

    answerPath = Path(testAnswersDir, answerName)
    with answerPath.open("r") as file:
        expected = json.load(file)

    expectedLen = expected["numTokens"]
    expectedTokens = expected["tokens"]

    filePath = Path(testInputDir, inputName)

    actualTokens = tc.TokenizeFile(filePath=filePath, model="gpt-4o", quiet=True)

    incorrectTokens = {}

    for i, actualToken in enumerate(actualTokens):

        if i < len(expectedTokens):

            if actualToken != expectedTokens[i]:

                incorrectTokens[f"{i}/{len(actualTokens) - 1}"] = {
                    "actual": actualToken,
                    "expected": expectedTokens[i],
                }

    # Check for extra tokens
    if len(expectedTokens) != len(actualTokens):
        min_length = min(len(expectedTokens), len(actualTokens))
        if len(expectedTokens) > len(actualTokens):
            for i in range(min_length, len(expectedTokens)):
                incorrectTokens[f"{i}"] = {
                    "actual": None,
                    "expected": expectedTokens[i],
                }
        else:
            for i in range(min_length, len(actualTokens)):
                incorrectTokens[f"{i}"] = {
                    "actual": actualTokens[i],
                    "expected": None,
                }

    if len(incorrectTokens) > 0:

        print(json.dumps(incorrectTokens, indent=4))
        print()

        RaiseTestAssertion(
            f"Tokenization mismatch for file '{filePath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Incorrect Tokens: {json.dumps(incorrectTokens, indent=4)}"
        )

    expectedLenCount = expectedLen
    actualLenCount = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o", quiet=True)
    if expectedLenCount != actualLenCount:

        RaiseTestAssertion(
            f"Token count mismatch for file '{filePath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Expected Count: {expectedLenCount}, "
            f"Got Count: {actualLenCount}"
        )


def TestFileError(imgPath):
    """
    Test tokenization of a file with unsupported encoding.
    """

    try:

        tc.TokenizeFile(filePath=imgPath, model="gpt-4o", quiet=True)

    except tc.UnsupportedEncodingError:

        pass  # Expected exception

    except Exception as e:

        RaiseTestAssertion(
            f"Test Failed: Unexpected error type raised for file '{imgPath}' - {type(e).__name__}"
        )

    else:

        RaiseTestAssertion(
            f"Test Failed: No error was raised for unsupported encoding in file '{imgPath}'."
        )


if __name__ == "__main__":

    # Existing Tests
    TestStr()
    TestFile(answerName="TestFile1.json", inputName="TestFile1.txt")
    TestFile(answerName="TestFile2.json", inputName="TestFile2.txt")
    TestFileError(imgPath=Path(testInputDir, "TestImg.jpg"))

    # Additional Tests
    TestGetModelMappings()
    TestGetValidModels()
    TestGetValidEncodings()
    TestGetModelForEncoding()
    TestGetEncodingForModel()
    TestGetEncoding()
    TestTokenizeDirectory()
    TestTokenizeDirectoryNoRecursion()
    TestTokenizeFilesMultiple()
    TestTokenizeFilesExitOnListErrorFalse()
    TestTokenizeFilesWithDirectory()
    TestTokenizeFilesListQuietFalse()
    TestTokenizeFileWithUnsupportedEncoding()
    TestTokenizeFileErrorType()

    print("All tests passed successfully!")
