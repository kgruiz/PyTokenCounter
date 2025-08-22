import json
from pathlib import Path
import PyTokenCounter as tc
from PyTokenCounter.encoding_utils import ReadTextFile
from PyTokenCounter.cli import ParseFiles

TEST_DIR = Path(__file__).resolve().parent.parent / "Tests"
INPUT_DIR = TEST_DIR / "Input"
ANSWERS_DIR = TEST_DIR / "Answers"


def test_parse_files_glob():
    pattern = "Input/TestFile*.txt"
    expected = {
        "Input/TestFile1.txt",
        "Input/TestFile2.txt",
    }
    result = set(ParseFiles([pattern]))
    assert result == expected


def test_parse_files_glob_recursive():
    pattern = "Input/**/*.txt"
    expected = {
        "Input/TestFile1.txt",
        "Input/TestFile2.txt",
        "Input/TestDirectory/TestDir1.txt",
        "Input/TestDirectory/TestDir2.txt",
        "Input/TestDirectory/TestDir3.txt",
        "Input/TestDirectory/TestSubDir/TestDir4.txt",
        "Input/TestDirectory/TestSubDir/TestDir5.txt",
    }
    result = set(ParseFiles([pattern]))
    assert result == expected


def test_read_text_file_windows1252():
    file_path = INPUT_DIR / "TestFile1252.txt"
    expected = "Café – résumé naïve fiancé"
    result = ReadTextFile(file_path)
    assert result == expected

