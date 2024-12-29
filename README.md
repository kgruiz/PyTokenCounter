# PyTokenCounter

PyTokenCounter is a Python library designed to simplify text tokenization and token counting. It supports various encoding schemes, with a focus on those used by **Large Language Models (LLMs)**, particularly those developed by OpenAI. Leveraging the `tiktoken` library for efficient processing, PyTokenCounter facilitates seamless integration with LLM workflows. This project is based on the [`tiktoken` library](https://github.com/openai/tiktoken) created by [OpenAI](https://github.com/openai/tiktoken).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [CLI](#cli)
- [API](#api)
  - [Utility Functions](#utility-functions)
  - [String Tokenization and Counting](#string-tokenization-and-counting)
  - [File and Directory Tokenization and Counting](#file-and-directory-tokenization-and-counting)
- [Maintainers](#maintainers)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Background

The development of PyTokenCounter was driven by the need for a user-friendly and efficient way to handle text tokenization in Python, particularly for applications that interact with **Large Language Models (LLMs)** like OpenAI's language models. **LLMs process text by breaking it down into tokens**, which are the fundamental units of input and output for these models. Tokenization, the process of converting text into a sequence of tokens, is a fundamental step in natural language processing and essential for optimizing interactions with LLMs.

Understanding and managing token counts is crucial when working with LLMs because it directly impacts aspects such as **API usage costs**, **prompt length limitations**, and **response generation**. PyTokenCounter addresses these needs by providing an intuitive interface for tokenizing strings, files, and directories, as well as counting the number of tokens based on different encoding schemes. With support for various OpenAI models and their associated encodings, PyTokenCounter is versatile enough to be used in a wide range of applications involving LLMs, such as prompt engineering, cost estimation, and monitoring usage.

## Install

Install PyTokenCounter using `pip`:

```bash
pip install PyTokenCounter
```

## Usage

Here are a few examples to get you started with PyTokenCounter, especially in the context of **LLMs**:

```python
from pathlib import Path

import PyTokenCounter as tc
import tiktoken

# Count tokens in a string for an LLM model
numTokens = tc.GetNumTokenStr(
    string="This is a test string.", model="gpt-3.5-turbo"
)
print(f"Number of tokens: {numTokens}")

# Count tokens in a file intended for LLM processing
numTokensFile = tc.GetNumTokenFile(
    filePath=Path("./test_file.txt"), model="gpt-4"
)
print(f"Number of tokens in file: {numTokensFile}")

# Count tokens in a directory of documents for batch processing with an LLM
numTokensDir = tc.GetNumTokenDir(
    dirPath=Path("./test_dir"), model="gpt-4", recursive=True
)
print(f"Number of tokens in directory: {numTokensDir}")

# Get the encoding for a specific LLM model
encoding = tc.GetEncoding(model="gpt-3.5-turbo")

# Tokenize a string using a specific encoding for LLM input
tokens = tc.TokenizeStr(string="This is another test.", encoding=encoding)
print(f"Token IDs: {tokens}")
```

### CLI

PyTokenCounter can also be used as a command-line tool, making it convenient to integrate into scripts and workflows that involve **LLMs**:

```bash
# Example usage for tokenizing a string for an LLM
tokencount tokenize-str "This is a test string." --model gpt-3.5-turbo

# Example usage for tokenizing a file for an LLM
tokencount tokenize-file test_file.txt --model gpt-4

# Example usage for tokenizing multiple files for an LLM
tokencount tokenize-files test_file1.txt test_file2.txt --model gpt-4

# Example usage for tokenizing a directory of files for an LLM
tokencount tokenize-files test_dir --model gpt-4 --no-recursive

# Example usage for tokenizing a directory of files for an LLM (alternative)
tokencount tokenize-dir test_dir --model gpt-4 --no-recursive

# Example usage for counting tokens in a string for an LLM
tokencount count-str "This is a test string." --model gpt-3.5-turbo

# Example usage for counting tokens in a file for an LLM
tokencount count-file test_file.txt --model gpt-4

# Example usage for counting tokens in multiple files for an LLM
tokencount count-files test_file1.txt test_file2.txt --model gpt-4

# Example usage for counting tokens in a directory for an LLM
tokencount count-files test_dir --model gpt-4 --no-recursive

# Example usage for counting tokens in a directory for an LLM (alternative)
tokencount count-dir test_dir --model gpt-4 --no-recursive
```

**CLI Usage Details:**

The `tokencount` CLI provides several subcommands for tokenizing and counting tokens in strings, files, and directories, tailored for use with **LLMs**.

**Subcommands:**

- `tokenize-str`: Tokenizes a provided string.
  - `tokencount tokenize-str "Your string here" --model gpt-3.5-turbo`
- `tokenize-file`: Tokenizes the contents of a file.
  - `tokencount tokenize-file path/to/your/file.txt --model gpt-4`
- `tokenize-files`: Tokenizes the contents of multiple specified files or all files within a directory.
    - `tokencount tokenize-files path/to/your/file1.txt path/to/your/file2.txt --model gpt-4`
    - `tokencount tokenize-files path/to/your/directory --model gpt-4 --no-recursive`
- `tokenize-dir`: Tokenizes all files within a specified directory into lists of token IDs.
    - `tokencount tokenize-dir path/to/your/directory --model gpt-4 --no-recursive`
- `count-str`: Counts the number of tokens in a provided string.
  - `tokencount count-str "Your string here" --model gpt-3.5-turbo`
- `count-file`: Counts the number of tokens in a file.
  - `tokencount count-file path/to/your/file.txt --model gpt-4`
- `count-files`: Counts the number of tokens in multiple specified files or all files within a directory.
  - `tokencount count-files path/to/your/file1.txt path/to/your/file2.txt --model gpt-4`
  - `tokencount count-files path/to/your/directory --model gpt-4 --no-recursive`
- `count-dir`: Counts the total number of tokens across all files in a specified directory.
  - `tokencount count-dir path/to/your/directory --model gpt-4 --no-recursive`

**Options:**

- `-m`, `--model`: Specifies the model to use for encoding, aligning with **LLM** specifications.
- `-e`, `--encoding`: Specifies the encoding to use directly.
- `-nr`, `--no-recursive`: When used with `tokenize-files`, `tokenize-dir`, `count-files` or `count-dir` for a directory, it prevents the tool from processing subdirectories recursively.

**Note:** For detailed help on each subcommand, use `tokencount <subcommand> -h`.

## API

Here's a detailed look at the PyTokenCounter API, designed to integrate seamlessly with **LLM** workflows:

### Utility Functions

#### `GetModelMappings() -> dict`

Retrieves the mappings between models and their corresponding encodings, essential for selecting the correct tokenization strategy for different **LLMs**.

**Returns:**

- `dict`: A dictionary where keys are model names and values are their corresponding encodings.

**Example:**

```python
import PyTokenCounter as tc

modelMappings = tc.GetModelMappings()
print(modelMappings)
```

---

#### `GetValidModels() -> list[str]`

Returns a list of valid model names supported by PyTokenCounter, primarily focusing on **LLMs**.

**Returns:**

- `list[str]`: A list of valid model names.

**Example:**

```python
import PyTokenCounter as tc

validModels = tc.GetValidModels()
print(validModels)
```

---

#### `GetValidEncodings() -> list[str]`

Returns a list of valid encoding names, ensuring compatibility with various **LLMs**.

**Returns:**

- `list[str]`: A list of valid encoding names.

**Example:**

```python
import PyTokenCounter as tc

validEncodings = tc.GetValidEncodings()
print(validEncodings)
```

---

#### `GetModelForEncoding(encodingName: str) -> str`

Determines the model name associated with a given encoding, facilitating the selection of appropriate **LLMs**.

**Parameters:**

- `encodingName` (`str`): The name of the encoding.

**Returns:**

- `str`: The model name corresponding to the given encoding.

**Raises:**

- `ValueError`: If the encoding name is not valid.

**Example:**

```python
import PyTokenCounter as tc

modelName = tc.GetModelForEncoding(encodingName="cl100k_base")
print(modelName)
```

---

#### `GetEncodingForModel(modelName: str) -> str`

Retrieves the encoding associated with a given model name, ensuring accurate tokenization for the selected **LLM**.

**Parameters:**

- `modelName` (`str`): The name of the model.

**Returns:**

- `str`: The encoding corresponding to the given model name.

**Raises:**

- `ValueError`: If the model name is not valid.

**Example:**

```python
import PyTokenCounter as tc

encodingName = tc.GetEncodingForModel(modelName="gpt-3.5-turbo")
print(encodingName)
```

---

#### `GetEncoding(model: str | None = None, encodingName: str | None = None) -> tiktoken.Encoding`

Obtains the `tiktoken` encoding based on the specified model or encoding name, tailored for **LLM** usage.

**Parameters:**

- `model` (`str`, optional): The name of the model.
- `encodingName` (`str`, optional): The name of the encoding.

**Returns:**

- `tiktoken.Encoding`: The `tiktoken` encoding object.

**Raises:**

- `ValueError`: If neither model nor encoding is provided, or if the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
import tiktoken

encoding = tc.GetEncoding(model="gpt-4")
print(type(encoding))

encoding = tc.GetEncoding(encodingName="cl100k_base")
print(type(encoding))
```

---

### String Tokenization and Counting

#### `TokenizeStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes a string into a list of token IDs, preparing text for input into an **LLM**.

**Parameters:**

- `string` (`str`): The string to tokenize.
- `model` (`str`, optional): The name of the model.
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.

**Returns:**

- `list[int]`: A list of token IDs.

**Raises:**

- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc

tokens = tc.TokenizeStr(string="This is a test string.", model="gpt-3.5-turbo")
print(tokens)

encoding = tc.GetEncoding(encodingName="cl100k_base")
tokens = tc.TokenizeStr(string="This is another test.", encoding=encoding)
print(tokens)
```

---

#### `GetNumTokenStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens in a string.

**Parameters:**

- `string` (`str`): The string to count tokens in.
- `model` (`str`, optional): The name of the model.
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.

**Returns:**

- `int`: The number of tokens in the string.

**Raises:**

- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc

numTokens = tc.GetNumTokenStr(string="This is a test string.", model="gpt-4")
print(numTokens)
```

---

### File and Directory Tokenization and Counting

#### `TokenizeFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes the contents of a file into a list of token IDs.

**Parameters:**

- `filePath` (`Path | str`): The path to the file to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.

**Returns:**

- `list[int]`: A list of token IDs representing the tokenized file contents.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Assuming 'test_file.txt' exists with some content
filePath = Path("./test_file.txt")
tokens = tc.TokenizeFile(filePath=filePath, model="gpt-3.5-turbo")
print(tokens)
```

---

#### `GetNumTokenFile(filePath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens in a file based on the specified model or encoding.

**Parameters:**

- `filePath` (`Path | str`): The path to the file to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.

**Returns:**

- `int`: The number of tokens in the file.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Assuming 'test_file.txt' exists
filePath = Path("./test_file.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4")
print(numTokens)
```

---

#### `TokenizeFiles(inputPath: Path | str | list[Path | str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> list[int | list] | list[list[int]] | list[int]`

**This is the main function used for tokenizing files** because it can tokenize single files, multiple files, or entire directories. Tokenizes multiple files or all files within a directory into lists of token IDs.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, default: `True`): If `inputPath` is a directory, whether to tokenize files in subdirectories recursively.

**Returns:**

- `list[int | list] | list[list[int]] | list[int]`:
  - If `inputPath` is a file, returns a list of token IDs for that file.
  - If `inputPath` is a list of files, returns a list where each element is a list of token IDs for each file.
  - If `inputPath` is a directory:
    - If `recursive` is `True`, returns a nested list where each element is either a list of token IDs representing a tokenized file or a sublist for a subdirectory.
    - If `recursive` is `False`, returns a list of token IDs for each file in the directory.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory, or if the provided model or encoding is invalid.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Tokenize a single file
filePath = Path("./test_file.txt")
tokens = tc.TokenizeFiles(inputPath=filePath, model="gpt-3.5-turbo")
print(tokens)

# Tokenize multiple files
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
tokenizedFiles = tc.TokenizeFiles(inputPath=filePaths, model="gpt-3.5-turbo")
print(tokenizedFiles)

# Tokenize a directory recursively
dirPath = Path("./test_dir")
tokenizedDir = tc.TokenizeFiles(inputPath=dirPath, model="gpt-4", recursive=True)
print(tokenizedDir)

# Tokenize a directory non-recursively
tokenizedDirNonRecursive = tc.TokenizeFiles(inputPath=dirPath, model="gpt-4", recursive=False)
print(tokenizedDirNonRecursive)
```

---

#### `GetNumTokenFiles(inputPath: Path | str | list[Path | str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

**This is the main function used for counting tokens in files** because it can count tokens in single files, multiple files, or entire directories. Counts the number of tokens across multiple files or in all files within a directory.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, default: `True`): If `inputPath` is a directory, whether to count tokens in files in subdirectories recursively.

**Returns:**

- `int`: The total number of tokens in the specified files or directory.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory, or if the provided model or encoding is invalid.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Count tokens in a single file
filePath = Path("./test_file.txt")
numTokens = tc.GetNumTokenFiles(inputPath=filePath, model="gpt-3.5-turbo")
print(numTokens)

# Count tokens in multiple files
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
numTokensFiles = tc.GetNumTokenFiles(inputPath=filePaths, model="gpt-3.5-turbo")
print(numTokensFiles)

# Count tokens in a directory recursively
dirPath = Path("./test_dir")
numTokensDir = tc.GetNumTokenFiles(inputPath=dirPath, model="gpt-4", recursive=True)
print(numTokensDir)

# Count tokens in a directory non-recursively
numTokensDirNonRecursive = tc.GetNumTokenFiles(inputPath=dirPath, model="gpt-4", recursive=False)
print(numTokensDirNonRecursive)
```

---

#### `TokenizeDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> list[int | list] | list[int]`

Tokenizes all files within a directory into lists of token IDs.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory.
- `model` (`str`, optional): The name of the model.
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.
- `recursive` (`bool`, optional): Whether to tokenize subdirectories recursively. Defaults to `True`.

**Returns:**

- `list[int | list] | list[int]`: A nested list of token IDs for each file (recursive) or a list of token IDs (non-recursive).

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Tokenize a directory recursively
dirPath = Path("./test_dir")
tokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4", recursive=True)
print(tokenizedDir)

# Tokenize a directory non-recursively
tokenizedDirNonRecursive = tc.TokenizeDir(dirPath=dirPath, model="gpt-4", recursive=False)
print(tokenizedDirNonRecursive)
```

---

#### `GetNumTokenDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens in all files within a directory.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory.
- `model` (`str`, optional): The name of the model.
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.
- `recursive` (`bool`, optional): Whether to count tokens in subdirectories recursively. Defaults to `True`.

**Returns:**

- `int`: The total number of tokens in the directory.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

# Count tokens in a directory recursively
dirPath = Path("./test_dir")
numTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4", recursive=True)
print(numTokensDir)

# Count tokens in a directory non-recursively
numTokensDirNonRecursive = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4", recursive=False)
print(numTokensDirNonRecursive)
```

---

## Maintainers

- [Kaden Gruizenga](https://github.com/kgruiz)

## Acknowledgements

- This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Contributing

Contributions are welcome! Feel free to [open an issue](https://github.com/kgruiz/PyTokenCounter/issues/new) or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
