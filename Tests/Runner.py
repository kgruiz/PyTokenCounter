import inspect
import json
import os
from pathlib import Path

import tiktoken

import PyTokenCounter as tc

# Define the paths to the test input and answers directories
TEST_INPUT_DIR = Path(
    "/Users/kadengruizenga/Development/Packages/Python/PyTokenCounter/Tests/Input"
)
TEST_ANSWERS_DIR = Path(
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
                        incorrectTokens[f"{i}/{len(actualTokens) - 1}"] = {
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

    answerPath = Path(TEST_ANSWERS_DIR, answerName)
    with answerPath.open("r") as file:
        expected = json.load(file)

    expectedLen = expected["numTokens"]
    expectedTokens = expected["tokens"]

    filePath = Path(TEST_INPUT_DIR, inputName)

    actualTokens = tc.TokenizeFile(filePath=filePath, model="gpt-4o", quiet=True)

    incorrectTokens = {}

    for i, actualToken in enumerate(actualTokens):

        if i < len(expectedTokens):

            if actualToken != expectedTokens[i]:

                incorrectTokens[f"{i}/{len(actualTokens) - 1}"] = {
                    "actual": actualToken,
                    "expected": expectedTokens[i],
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
    Test retrieval of the correct model for each encoding.
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
        if actualModel != expectedModel:
            RaiseTestAssertion(
                f"Model for encoding '{encodingName}' mismatch.\n"
                f"Expected: {expectedModel}\n"
                f"Got: {actualModel}"
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


def TestTokenizeDirectory():
    """
    Test tokenization of a directory with multiple files.
    """
    dirPath = Path(TEST_INPUT_DIR, "TestDirectory")
    answerPath = Path(TEST_ANSWERS_DIR, "TestDirectory.json")

    # Load expected results
    with answerPath.open("r") as file:
        expected = json.load(file)

    tokenizedDir = tc.TokenizeDir(
        dirPath=dirPath,
        model="gpt-4o",
        recursive=False,  # Assuming no subdirectories within TestDirectory
        quiet=True,
    )

    # tokenizedDir is a list of token lists corresponding to each file in the directory
    # Assuming the order of files is consistent
    expectedFiles = sorted(expected.keys())

    expectedFiles = [f"{fileName}.txt" for fileName in expectedFiles]

    actualFiles = sorted(
        entry.name
        for entry in dirPath.iterdir()
        if entry.is_file() and entry.suffix != ".jpg"
    )

    if len(tokenizedDir) != len(expectedFiles):
        RaiseTestAssertion(
            f"Number of tokenized files mismatch in directory '{dirPath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Expected: {len(expectedFiles)} files\n"
            f"Got: {len(tokenizedDir)} tokenized files"
        )

    incorrectTokens = {}

    for idx, fileName in enumerate(expectedFiles):

        expectedTokens = expected[fileName.replace(".txt", "")]["tokens"]
        actualTokens = tokenizedDir[fileName]

        incorrectTokens[fileName] = dict()

        if actualTokens != expectedTokens:

            for i, actualToken in enumerate(actualTokens):

                if i < len(expectedTokens):

                    if actualToken != expectedTokens[i]:

                        incorrectTokens[fileName][f"{i}/{len(actualTokens) - 1}"] = {
                            "actual": actualToken,
                            "expected": expectedTokens[i],
                        }

    if any(
        [
            len(incorrectFileTokens) > 0
            for incorrectFileTokens in incorrectTokens.values()
        ]
    ):

        RaiseTestAssertion(
            f"Tokenization mismatch for file '{fileName}' in directory '{dirPath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Incorrect Tokens: {json.dumps(incorrectTokens, indent=4)}"
        )


def TestGetNumTokenDirectory():
    """
    Test counting the number of tokens in a directory.
    """
    dirPath = Path(TEST_INPUT_DIR, "TestDirectory")
    answerPath = Path(TEST_ANSWERS_DIR, "TestDirectory.json")

    # Load expected results
    with answerPath.open("r") as file:
        expected = json.load(file)

    expectedTotalTokens = sum(item["numTokens"] for item in expected.values())

    actualTotalTokens = tc.GetNumTokenDir(
        dirPath=dirPath,
        model="gpt-4o",
        recursive=False,  # Assuming no subdirectories within TestDirectory
        quiet=True,
    )

    if actualTotalTokens != expectedTotalTokens:
        RaiseTestAssertion(
            f"Total token count mismatch for directory '{dirPath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Expected Total Tokens: {expectedTotalTokens}\n"
            f"Got Total Tokens: {actualTotalTokens}"
        )


def TestTokenizeFilesMultiple():
    """
    Test tokenization of multiple files using TokenizeFiles function.
    """
    inputFiles = [
        Path(TEST_INPUT_DIR, "TestFile1.txt"),
        Path(TEST_INPUT_DIR, "TestFile2.txt"),
    ]
    answerFiles = [
        Path(TEST_ANSWERS_DIR, "TestFile1.json"),
        Path(TEST_ANSWERS_DIR, "TestFile2.json"),
    ]

    expectedTokenLists = []
    for answerFile in answerFiles:
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists.append(answer["tokens"])

    tokenizedFiles = tc.TokenizeFiles(inputFiles, model="gpt-4o", quiet=True)

    if tokenizedFiles != expectedTokenLists:
        # Identify mismatches
        mismatches = {}
        for idx, (expected, actual) in enumerate(
            zip(expectedTokenLists, tokenizedFiles)
        ):
            if expected != actual:
                incorrectTokens = {}
                for i, (exp_tok, act_tok) in enumerate(zip(expected, actual)):
                    if exp_tok != act_tok:
                        incorrectTokens[f"{i}/{len(actual)}"] = {
                            "actual": act_tok,
                            "expected": exp_tok,
                        }
                mismatches[f"File '{inputFiles[idx]}'"] = incorrectTokens

        RaiseTestAssertion(
            f"Tokenization mismatch for multiple files.\n"
            f"Mismatches:\n{json.dumps(mismatches, indent=4)}"
        )


def TestGetNumTokenFilesMultiple():
    """
    Test counting the number of tokens in multiple files using GetNumTokenFiles function.
    """
    inputFiles = [
        Path(TEST_INPUT_DIR, "TestFile1.txt"),
        Path(TEST_INPUT_DIR, "TestFile2.txt"),
    ]
    answerFiles = [
        Path(TEST_ANSWERS_DIR, "TestFile1.json"),
        Path(TEST_ANSWERS_DIR, "TestFile2.json"),
    ]

    expectedTotalTokens = 0
    for answerFile in answerFiles:
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTotalTokens += answer["numTokens"]

    actualTotalTokens = tc.GetNumTokenFiles(inputFiles, model="gpt-4o", quiet=True)

    if actualTotalTokens != expectedTotalTokens:
        RaiseTestAssertion(
            f"Total token count mismatch for multiple files.\n"
            f"Expected Total Tokens: {expectedTotalTokens}\n"
            f"Got Total Tokens: {actualTotalTokens}"
        )


def TestInvalidModel():
    """
    Test behavior when an invalid model name is provided.
    """
    invalidModel = "invalid-model"

    try:
        tc.TokenizeStr(string="Test string", model=invalidModel)
    except ValueError as e:
        expectedMessage = f"Invalid model: {invalidModel}"
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for invalid model.\n"
                f"Model: '{invalidModel}'\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            f"Test Failed: No error was raised for invalid model '{invalidModel}'."
        )


def TestInvalidEncodingName():
    """
    Test behavior when an invalid encoding name is provided.
    """
    invalidEncodingName = "invalid-encoding"

    try:
        tc.TokenizeStr(
            string="Test string", encodingName=invalidEncodingName, quiet=True
        )
    except ValueError as e:
        expectedMessage = f"Invalid encoding name: {invalidEncodingName}"
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for invalid encoding name.\n"
                f"Encoding Name: '{invalidEncodingName}'\n"
                f"Expected to contain: '{expectedMessage}'\n"
                f"Got: '{e}'"
            )
    else:
        RaiseTestAssertion(
            f"Test Failed: No error was raised for invalid encoding name '{invalidEncodingName}'."
        )


def TestMismatchedModelAndEncoding():
    """
    Test behavior when the provided model and encodingName do not match.
    """
    mismatchedModel = "gpt-3.5-turbo"
    mismatchedEncodingName = "p50k_base"  # Incorrect encoding for this model

    try:
        tc.TokenizeStr(
            string="Test string",
            model=mismatchedModel,
            encodingName=mismatchedEncodingName,
            quiet=True,
        )
    except ValueError as e:
        expectedMessage = f"Model {mismatchedModel} does not have encoding name {mismatchedEncodingName}"
        if expectedMessage not in str(e):
            RaiseTestAssertion(
                f"Unexpected error message for mismatched model and encodingName.\n"
                f"Model: '{mismatchedModel}', Encoding Name: '{mismatchedEncodingName}'\n"
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


def TestTokenizeFilesWithDirectory():
    """
    Test tokenization of a directory using TokenizeFiles function.
    """
    dirPath = Path(TEST_INPUT_DIR, "TestDirectory")
    answerPath = Path(TEST_ANSWERS_DIR, "TestDirectory.json")

    # Load expected results
    with answerPath.open("r") as file:
        expected = json.load(file)

    expectedFiles = sorted(expected.keys())

    expectedFiles = [f"{fileName}.txt" for fileName in expectedFiles]

    tokenizedDir = tc.TokenizeFiles(dirPath, model="gpt-4o", quiet=True)

    actualFiles = sorted(
        entry.name
        for entry in dirPath.iterdir()
        if entry.is_file() and entry.suffix != ".jpg"
    )

    if len(tokenizedDir) != len(expectedFiles):
        RaiseTestAssertion(
            f"Number of tokenized files mismatch in directory '{dirPath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Expected: {len(expectedFiles)} files\n"
            f"Got: {len(tokenizedDir)} tokenized files"
        )

    incorrectTokens = {}

    for idx, fileName in enumerate(expectedFiles):

        expectedTokens = expected[fileName.replace(".txt", "")]["tokens"]
        actualTokens = tokenizedDir[fileName]

        incorrectTokens[fileName] = dict()

        if actualTokens != expectedTokens:

            for i, actualToken in enumerate(actualTokens):

                if i < len(expectedTokens):

                    if actualToken != expectedTokens[i]:

                        incorrectTokens[fileName][f"{i}/{len(actualTokens) - 1}"] = {
                            "actual": actualToken,
                            "expected": expectedTokens[i],
                        }

    if any(
        [
            len(incorrectFileTokens) > 0
            for incorrectFileTokens in incorrectTokens.values()
        ]
    ):

        RaiseTestAssertion(
            f"Tokenization mismatch for file '{fileName}' in directory '{dirPath}'.\n"
            f"Answer Path: '{answerPath}'\n"
            f"Incorrect Tokens: {json.dumps(incorrectTokens, indent=4)}"
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
        Path(TEST_INPUT_DIR, "TestFile1.txt"),
        Path(TEST_INPUT_DIR, "TestFile2.txt"),
    ]
    answerFiles = [
        Path(TEST_ANSWERS_DIR, "TestFile1.json"),
        Path(TEST_ANSWERS_DIR, "TestFile2.json"),
    ]

    expectedTokenLists = []
    for answerFile in answerFiles:
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists.append(answer["tokens"])

    tokenizedFiles = tc.TokenizeFiles(inputFiles, model="gpt-4o", quiet=False)

    if tokenizedFiles != expectedTokenLists:
        # Identify mismatches
        mismatches = {}
        for idx, (expected, actual) in enumerate(
            zip(expectedTokenLists, tokenizedFiles)
        ):
            if expected != actual:
                incorrectTokens = {}
                for i, (exp_tok, act_tok) in enumerate(zip(expected, actual)):
                    if exp_tok != act_tok:
                        incorrectTokens[f"{i}/{len(actual)}"] = {
                            "actual": act_tok,
                            "expected": exp_tok,
                        }
                mismatches[f"File '{inputFiles[idx]}'"] = incorrectTokens

        RaiseTestAssertion(
            f"Tokenization mismatch for multiple files with quiet=False.\n"
            f"Mismatches:\n{json.dumps(mismatches, indent=4)}"
        )


def TestTokenizeFilesExitOnListErrorFalse():
    """
    Test TokenizeFiles function with exitOnListError=False to ensure it continues on encountering errors.
    """
    inputList = [
        Path(TEST_INPUT_DIR, "TestFile1.txt"),
        Path(
            TEST_INPUT_DIR, "TestImg.jpg"
        ),  # Assuming this file has unsupported encoding
        Path(TEST_INPUT_DIR, "TestFile2.txt"),
    ]

    answerFiles = [
        Path(TEST_ANSWERS_DIR, "TestFile1.json"),
        Path(TEST_ANSWERS_DIR, "TestFile2.json"),
    ]

    expectedTokenLists = []
    for answerFile in answerFiles:
        with answerFile.open("r") as file:
            answer = json.load(file)
            expectedTokenLists.append(answer["tokens"])

    tokenizedFiles = tc.TokenizeFiles(
        inputList, model="gpt-4o", quiet=True, exitOnListEror=False
    )

    # The unsupported encoding file should be skipped
    if tokenizedFiles != expectedTokenLists:
        # Identify mismatches
        mismatches = {}
        for idx, (expected, actual) in enumerate(
            zip(expectedTokenLists, tokenizedFiles)
        ):
            if expected != actual:
                incorrectTokens = {}
                for i, (exp_tok, act_tok) in enumerate(zip(expected, actual)):
                    if exp_tok != act_tok:
                        incorrectTokens[f"{i}/{len(actual)}"] = {
                            "actual": act_tok,
                            "expected": exp_tok,
                        }
                mismatches[f"File '{inputList[idx]}'"] = incorrectTokens

        RaiseTestAssertion(
            f"TokenizeFiles with exitOnListError=False mismatch.\n"
            f"Mismatches:\n{json.dumps(mismatches, indent=4)}"
        )


def TestTokenizeFileWithUnsupportedEncoding():
    """
    Test tokenization of a file with unsupported encoding.
    """
    unsupportedFilePath = Path(
        TEST_INPUT_DIR, "TestImg.jpg"
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


if __name__ == "__main__":

    # Existing Tests
    TestStr()
    TestFile(answerName="TestFile1.json", inputName="TestFile1.txt")
    TestFile(answerName="TestFile2.json", inputName="TestFile2.txt")
    TestFileError(imgPath=Path(TEST_INPUT_DIR, "TestImg.jpg"))

    # Additional Tests
    TestGetModelMappings()
    TestGetValidModels()
    TestGetValidEncodings()
    TestGetModelForEncoding()
    TestGetEncodingForModel()
    TestGetEncoding()
    TestTokenizeDirectory()
    TestGetNumTokenDirectory()
    TestTokenizeFilesMultiple()
    TestGetNumTokenFilesMultiple()
    TestInvalidModel()
    TestInvalidEncodingName()
    TestMismatchedModelAndEncoding()
    TestGetEncodingError()
    TestTokenizeFilesWithDirectory()
    TestTokenizeFilesListQuietFalse()
    TestTokenizeFilesExitOnListErrorFalse()
    TestTokenizeFileWithUnsupportedEncoding()
    TestTokenizeFileErrorType()

    print("All tests passed successfully!")
