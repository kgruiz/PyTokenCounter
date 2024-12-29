# TokenCounter

TokenCounter is a Python library designed to simplify text tokenization and token counting. It supports various encoding schemes, with a focus on those used by OpenAI models, and leverages the `tiktoken` library for efficient processing. This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [CLI](#cli)
- [API](#api)
- [Maintainers](#maintainers)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Background

The development of TokenCounter was driven by the need for a user-friendly and efficient way to handle text tokenization in Python, particularly for applications that interact with OpenAI's language models. Tokenization, the process of converting text into a sequence of tokens, is a fundamental step in natural language processing.

This library offers an intuitive interface for tokenizing strings, files, and directories. It also allows for counting the number of tokens based on different encoding schemes. With support for various OpenAI models and their associated encodings, TokenCounter is versatile enough to be used in a wide range of applications.

## Install

Install TokenCounter using `pip`:

```bash
pip install TokenCounter
```

## Usage

Here are a few examples to get you started with TokenCounter:

```python
from pathlib import Path

import TokenCounter as tc
import tiktoken

# Count tokens in a string
numTokens = tc.GetNumTokenStr(
    string="This is a test string.", model="gpt-3.5-turbo"
)
print(f"Number of tokens: {numTokens}")

# Count tokens in a file
numTokensFile = tc.GetNumTokenFile(
    filePath=Path("./test_file.txt"), model="gpt-4"
)
print(f"Number of tokens in file: {numTokensFile}")

# Count tokens in a directory
numTokensDir = tc.GetNumTokenDir(
    dirPath=Path("./test_dir"), model="gpt-4o", recursive=True
)
print(f"Number of tokens in directory: {numTokensDir}")

# Get the encoding for a model
encoding = tc.GetEncoding(model="gpt-3.5-turbo")

# Tokenize a string using a specific encoding
tokens = tc.TokenizeStr(string="This is another test.", encoding=encoding)
print(f"Token IDs: {tokens}")
```

### CLI

TokenCounter can also be used as a command-line tool:

```bash
# Example usage for counting tokens in a string
poetry run python TokenCounter/main.py --string "This is a test string." --model gpt-3.5-turbo

# Example usage for counting tokens in a file
poetry run python TokenCounter/main.py --file test_file.txt --model gpt-4

# Example usage for counting tokens in a directory
poetry run python TokenCounter/main.py --dir test_dir --model gpt-4o --recursive
```

## API

### `GetModelMappings() -> dict`

Retrieves the mappings between models and their corresponding encodings.

### `GetValidModels() -> list[str]`

Returns a list of valid model names.

### `GetValidEncodings() -> list[str]`

Returns a list of valid encoding names.

### `GetModelForEncoding(encodingName: str) -> str`

Determines the model name associated with a given encoding.

### `GetEncodingForModel(modelName: str) -> str`

Retrieves the encoding associated with a given model name.

### `GetEncoding(model: str | None = None, encodingName: str | None = None) -> tiktoken.Encoding`

Obtains the `tiktoken` encoding based on the specified model or encoding name.

### `TokenizeStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes a string into a list of token IDs.

### `GetNumTokenStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens in a string.

### `TokenizeFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes the contents of a file into a list of token IDs.

### `GetNumTokenFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens in a file.

### `TokenizeFiles(filePaths: list[Path] | list[str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[list[int]]`

Tokenizes multiple files into lists of token IDs.

### `GetNumTokenFiles(filePaths: list[Path] | list[str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens across multiple files.

### `TokenizeDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> list[int | list] | list[int]`

Tokenizes all files within a directory into lists of token IDs.

### `GetNumTokenDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens in all files within a directory.

## Maintainers

- [Kaden Gruizenga](https://github.com/kgruiz)

## Acknowledgements

- This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Contributing

Contributions are welcome! Feel free to [open an issue](https://github.com/kgruiz/TokenCounter/issues/new) or submit a pull request.

TokenCounter follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) Code of Conduct.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.