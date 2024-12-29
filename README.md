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
# Example usage for tokenizing a string
tokencount tokenize-str "This is a test string." --model gpt-3.5-turbo

# Example usage for tokenizing a file
tokencount tokenize-file test_file.txt --model gpt-4

# Example usage for tokenizing a directory
tokencount tokenize-dir test_dir --model gpt-4o --no-recursive

# Example usage for counting tokens in a string
tokencount count-str "This is a test string." --model gpt-3.5-turbo

# Example usage for counting tokens in a file
tokencount count-file test_file.txt --model gpt-4

# Example usage for counting tokens in a directory
tokencount count-dir test_dir --model gpt-4o --no-recursive
```

**CLI Usage Details:**

The `tokencount` CLI provides several subcommands for tokenizing and counting tokens in strings, files, and directories.

**Subcommands:**

-   `tokenize-str`: Tokenizes a provided string.
    -   `tokencount tokenize-str "Your string here" --model gpt-3.5-turbo`
-   `tokenize-file`: Tokenizes the contents of a file.
    -   `tokencount tokenize-file path/to/your/file.txt --model gpt-4`
-   `tokenize-dir`: Tokenizes all files in a directory.
    -   `tokencount tokenize-dir path/to/your/directory --model gpt-4o --no-recursive`
-   `count-str`: Counts the number of tokens in a provided string.
    -   `tokencount count-str "Your string here" --encoding cl100k_base`
-   `count-file`: Counts the number of tokens in a file.
    -   `tokencount count-file path/to/your/file.txt --encoding cl100k_base`
-   `count-dir`: Counts the total number of tokens in all files within a directory.
    -   `tokencount count-dir path/to/your/directory --encoding cl100k_base --no-recursive`

**Options:**

-   `-m`, `--model`: Specifies the model to use for encoding.
-   `-e`, `--encoding`: Specifies the encoding to use directly.
-   `-nr`, `--no-recursive`: When used with `tokenize-dir` or `count-dir`, it prevents the tool from processing subdirectories recursively.

**Note:** For detailed help on each subcommand, use `tokencount <subcommand> -h`.

## API

Here's a detailed look at the TokenCounter API:

### `GetModelMappings() -> dict`

Retrieves the mappings between models and their corresponding encodings.

**Returns:**

-   `dict`: A dictionary where keys are model names and values are their corresponding encodings.

**Example:**

```python
import TokenCounter as tc

modelMappings = tc.GetModelMappings()
print(modelMappings)
```

---

### `GetValidModels() -> list[str]`

Returns a list of valid model names.

**Returns:**

-   `list[str]`: A list of valid model names.

**Example:**

```python
import TokenCounter as tc

validModels = tc.GetValidModels()
print(validModels)
```

---

### `GetValidEncodings() -> list[str]`

Returns a list of valid encoding names.

**Returns:**

-   `list[str]`: A list of valid encoding names.

**Example:**

```python
import TokenCounter as tc

validEncodings = tc.GetValidEncodings()
print(validEncodings)
```

---

### `GetModelForEncoding(encodingName: str) -> str`

Determines the model name associated with a given encoding.

**Parameters:**

-   `encodingName` (`str`): The name of the encoding.

**Returns:**

-   `str`: The model name corresponding to the given encoding.

**Raises:**

-   `ValueError`: If the encoding name is not valid.

**Example:**

```python
import TokenCounter as tc

modelName = tc.GetModelForEncoding(encodingName="cl100k_base")
print(modelName)
```

---

### `GetEncodingForModel(modelName: str) -> str`

Retrieves the encoding associated with a given model name.

**Parameters:**

-   `modelName` (`str`): The name of the model.

**Returns:**

-   `str`: The encoding corresponding to the given model name.

**Raises:**

-   `ValueError`: If the model name is not valid.

**Example:**

```python
import TokenCounter as tc

encodingName = tc.GetEncodingForModel(modelName="gpt-3.5-turbo")
print(encodingName)
```

---

### `GetEncoding(model: str | None = None, encodingName: str | None = None) -> tiktoken.Encoding`

Obtains the `tiktoken` encoding based on the specified model or encoding name.

**Parameters:**

-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.

**Returns:**

-   `tiktoken.Encoding`: The `tiktoken` encoding object.

**Raises:**

-   `ValueError`: If neither model nor encoding is provided, or if the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
import tiktoken

encoding = tc.GetEncoding(model="gpt-4")
print(type(encoding))

encoding = tc.GetEncoding(encodingName="cl100k_base")
print(type(encoding))
```

---

### `TokenizeStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes a string into a list of token IDs.

**Parameters:**

-   `string` (`str`): The string to tokenize.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.

**Returns:**

-   `list[int]`: A list of token IDs.

**Raises:**

-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc

tokens = tc.TokenizeStr(string="This is a test string.", model="gpt-3.5-turbo")
print(tokens)

encoding = tc.GetEncoding(encodingName="cl100k_base")
tokens = tc.TokenizeStr(string="This is another test.", encoding=encoding)
print(tokens)
```

---

### `GetNumTokenStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens in a string.

**Parameters:**

-   `string` (`str`): The string to count tokens in.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.

**Returns:**

-   `int`: The number of tokens in the string.

**Raises:**

-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc

numTokens = tc.GetNumTokenStr(string="This is a test string.", model="gpt-4")
print(numTokens)
```

---

### `TokenizeFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`
Tokenizes the contents of a file into a list of token IDs.

**Parameters:**

-   `filePath` (`Path | str`): The path to the file to tokenize.
-   `model` (`str`, optional): The name of the model to use for encoding.
-   `encodingName` (`str`, optional): The name of the encoding to use.
-   `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.

**Returns:**

-   `list[int]`: A list of token IDs representing the tokenized file contents.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'test_file.txt' exists with some content
filePath = Path("./test_file.txt")
tokens = tc.TokenizeFile(filePath=filePath, model="gpt-3.5-turbo")
print(tokens)
```

---

### `GetNumTokenFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`
Counts the number of tokens in a file based on the specified model or encoding.

**Parameters:**

-   `filePath` (`Path | str`): The path to the file to count tokens for.
-   `model` (`str`, optional): The name of the model to use for encoding.
-   `encodingName` (`str`, optional): The name of the encoding to use.
-   `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.

**Returns:**

-   `int`: The number of tokens in the file.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'test_file.txt' exists
filePath = Path("./test_file.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4")
print(numTokens)
```

---

### `TokenizeFiles(filePaths: list[Path] | list[str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[list[int]]`
Tokenizes multiple files into lists of token IDs.

**Parameters:**

-   `filePaths` (`list[Path] | list[str]`): A list of paths to the files to tokenize.
-   `model` (`str`, optional): The name of the model to use for encoding.
-   `encodingName` (`str`, optional): The name of the encoding to use.
-   `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.

**Returns:**

-   `list[list[int]]`: A list where each element is a list of token IDs representing a tokenized file.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'file1.txt' and 'file2.txt' exist
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
tokenizedFiles = tc.TokenizeFiles(filePaths=filePaths, model="gpt-3.5-turbo")
print(tokenizedFiles)
```

---

### `GetNumTokenFiles(filePaths: list[Path] | list[str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens across multiple files.

**Parameters:**

-   `filePaths` (`list[Path] | list[str]`): A list of paths to the files.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.

**Returns:**

-   `int`: The total number of tokens across all files.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'file1.txt' and 'file2.txt' exist
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
numTokens = tc.GetNumTokenFiles(filePaths=filePaths, model="gpt-3.5-turbo")
print(numTokens)
```

---

### `TokenizeDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> list[int | list] | list[int]`

Tokenizes all files within a directory into lists of token IDs.

**Parameters:**

-   `dirPath` (`Path | str`): The path to the directory.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.
-   `recursive` (`bool`, optional): Whether to tokenize subdirectories recursively. Defaults to `True`.

**Returns:**

-   `list[int | list] | list[int]`: A nested list of token IDs for each file (recursive) or a list of token IDs (non-recursive).

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'test_dir' exists and contains some files
dirPath = Path("./test_dir")
tokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(tokenizedDir)
```

---

### `GetNumTokenDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens in all files within a directory.

**Parameters:**

-   `dirPath` (`Path | str`): The path to the directory.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.
-   `recursive` (`bool`, optional): Whether to count tokens in subdirectories recursively. Defaults to `True`.

**Returns:**

-   `int`: The total number of tokens in the directory.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import TokenCounter as tc
from pathlib import Path

# Assuming 'test_dir' exists and contains some files
dirPath = Path("./test_dir")
numTokens = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(numTokens)
```

## Maintainers

- [Kaden Gruizenga](https://github.com/kgruiz)

## Acknowledgements

- This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Contributing

Contributions are welcome! Feel free to [open an issue](https://github.com/kgruiz/TokenCounter/issues/new) or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.