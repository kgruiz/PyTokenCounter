import json
import unittest
from pathlib import Path

import tiktoken
from box import Box

import PyTokenCounter as tc

TEST_INPUT_DIR = Path(
    "/Users/kadengruizenga/Development/Packages/Python/PyTokenCounter/Tests/Input"
)

TEST_ANSWERS_DIR = Path(
    "/Users/kadengruizenga/Development/Packages/Python/PyTokenCounter/Tests/Answers"
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
            raise AssertionError(
                f"Tokenization mismatch for string '{string}'.\nExpected: {expectedTokens}, Got: {actualTokens}"
            )

        if len(expectedTokens) != tc.GetNumTokenStr(
            string=string, model="gpt-4o", quiet=True
        ):
            raise AssertionError(
                f"Token count mismatch for string '{string}'.\nExpected: {len(expectedTokens)}, Got: {tc.GetNumTokenStr(string=string, model='gpt-4o', quiet=True)}"
            )


def TestFile():
    """
    Test file tokenization.
    """

    with Path(TEST_ANSWERS_DIR, "TestFile1.json").open("r") as file:

        expected = Box(json.load(file))

    expectedLen = expected.numTokens
    expectedTokens = expected.tokens

    filePath = Path(TEST_INPUT_DIR, "TestFile1.txt")

    actualTokens = tc.TokenizeFile(filePath=filePath, model="gpt-4o", quiet=True)

    incorrectTokens = dict()

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

    if expectedLen != tc.GetNumTokenFile(filePath=filePath, model="gpt-4o", quiet=True):

        raise AssertionError(
            f"Token count mismatch for file '{filePath}'.\nExpected: {expectedLen}, Got: {tc.GetNumTokenFile(filePath=filePath, model='gpt-4o', quiet=True)}"
        )


if __name__ == "__main__":

    TestStr()

    TestFile()
