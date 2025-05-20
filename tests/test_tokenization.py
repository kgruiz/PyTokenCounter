import json
from pathlib import Path
import io
import sys
import PyTokenCounter as tc
from PyTokenCounter.cli import ParseFiles

# Paths to test resources
TEST_DIR = Path(__file__).resolve().parent.parent / "Tests"
INPUT_DIR = TEST_DIR / "Input"
ANSWERS_DIR = TEST_DIR / "Answers"


def load_answer(name: str):
    with (ANSWERS_DIR / name).open("r") as f:
        return json.load(f)

def test_tokenize_directory():
    expected = load_answer("TestDirectory.json")
    result = tc.TokenizeDir(
        dirPath=INPUT_DIR / "TestDirectory",
        model="gpt-4o",
        recursive=True,
        quiet=True,
    )
    assert isinstance(result, dict)
    assert result == expected


def test_tokenize_files_with_directory():
    expected = load_answer("TestDirectory.json")
    result = tc.TokenizeFiles(INPUT_DIR / "TestDirectory", model="gpt-4o", quiet=True)
    assert isinstance(result, dict)
    assert result == expected


def test_tokenize_files_multiple():
    files = [
        INPUT_DIR / "TestFile1.txt",
        INPUT_DIR / "TestImg.jpg",
        INPUT_DIR / "TestFile2.txt",
    ]
    answers = [load_answer("TestFile1.json"), load_answer("TestFile2.json")]
    result = tc.TokenizeFiles(files, model="gpt-4o", quiet=True)
    expected = {
        "TestFile1.txt": {"tokens": answers[0]["tokens"]},
        "TestFile2.txt": {"tokens": answers[1]["tokens"]},
    }
    assert isinstance(result, dict)
    assert result == expected


def test_tokenize_files_exit_on_list_error_false():
    files = [
        INPUT_DIR / "TestFile1.txt",
        INPUT_DIR / "TestImg.jpg",
        INPUT_DIR / "TestFile2.txt",
    ]
    answers = [load_answer("TestFile1.json"), load_answer("TestFile2.json")]
    result = tc.TokenizeFiles(files, model="gpt-4o", quiet=True, exitOnListError=False)
    expected = {
        "TestFile1.txt": {"tokens": answers[0]["tokens"]},
        "TestFile2.txt": {"tokens": answers[1]["tokens"]},
    }
    assert isinstance(result, dict)
    assert result == expected


def test_tokenize_directory_no_recursion():
    expected = load_answer("TestDirectoryNoRecursion.json")
    result = tc.TokenizeDir(
        dirPath=INPUT_DIR / "TestDirectory",
        model="gpt-4o",
        recursive=False,
        quiet=True,
    )
    assert isinstance(result, dict)
    assert result == expected
    # ensure subdirectories not included
    for entry in (INPUT_DIR / "TestDirectory").iterdir():
        if entry.is_dir():
            assert entry.name not in result


def test_tokenize_files_with_invalid_input():
    with pytest.raises(TypeError):
        tc.TokenizeFiles(67890)


def test_tokenize_files_list_quiet_false(capsys):
    files = [
        INPUT_DIR / "TestFile1.txt",
        INPUT_DIR / "TestImg.jpg",
        INPUT_DIR / "TestFile2.txt",
    ]
    answers = [load_answer("TestFile1.json"), load_answer("TestFile2.json")]
    result = None
    captured = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = tc.TokenizeFiles(files, model="gpt-4o", quiet=False)
    finally:
        sys.stdout = sys_stdout
    output = captured.getvalue()
    assert "Skipping binary file TestImg.jpg" in output
    expected = {
        "TestFile1.txt": {"tokens": answers[0]["tokens"]},
        "TestFile2.txt": {"tokens": answers[1]["tokens"]},
    }
    assert isinstance(result, dict)
    assert result == expected


import pytest

@pytest.mark.parametrize(
    "input_name,answer_name",
    [
        ("TestFile1.txt", "TestFile1.json"),
        ("TestFile2.txt", "TestFile2.json"),
    ],
)
def test_tokenize_file(input_name, answer_name):
    answer = load_answer(answer_name)
    result = tc.TokenizeFile(INPUT_DIR / input_name, model="gpt-4o", quiet=True)
    assert result == answer["tokens"]
    count = tc.GetNumTokenFile(INPUT_DIR / input_name, model="gpt-4o", quiet=True)
    assert count == answer["numTokens"]


def test_tokenize_file_error():
    with pytest.raises(tc.UnsupportedEncodingError):
        tc.TokenizeFile(INPUT_DIR / "TestImg.jpg", model="gpt-4o", quiet=True)


def test_tokenize_file_error_type():
    with pytest.raises(TypeError):
        tc.TokenizeFile(54321, model="gpt-4o", quiet=True)


def test_tokenize_file_with_unsupported_encoding():
    path = INPUT_DIR / "TestImg.jpg"
    with pytest.raises(tc.UnsupportedEncodingError) as exc:
        tc.TokenizeFile(filePath=path, model="gpt-4o", quiet=True)
    assert str(path) in str(exc.value)
    assert "encoding" in str(exc.value)


def test_tokenize_str():
    expected_strings = {
        "Hail to the Victors!": [39, 663, 316, 290, 16566, 914, 0],
        "2024 National Champions": [1323, 19, 6743, 40544],
        "Corum 4 Heisman": [11534, 394, 220, 19, 1679, 107107],
    }
    for text, tokens in expected_strings.items():
        result = tc.TokenizeStr(string=text, model="gpt-4o", quiet=True)
        assert result == tokens
        count = tc.GetNumTokenStr(string=text, model="gpt-4o", quiet=True)
        assert count == len(tokens)

