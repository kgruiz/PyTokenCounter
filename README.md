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
    string="This is a test string.", model="gpt-4o"
)
print(f"Number of tokens: {numTokens}")

# Count tokens in a file intended for LLM processing
filePath = Path("./TestFile.txt")
numTokensFile = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o")
print(f"Number of tokens in file: {numTokensFile}")

# Count tokens in a directory of documents for batch processing with an LLM
dirPath = Path("./TestDir")
numTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=True)
print(f"Number of tokens in directory: {numTokensDir}")

# Get the encoding for a specific LLM model
encoding = tc.GetEncoding(model="gpt-4o")

# Tokenize a string using a specific encoding for LLM input
tokens = tc.TokenizeStr(string="This is another test.", encoding=encoding)
print(f"Token IDs: {tokens}")
```

### CLI

PyTokenCounter can also be used as a command-line tool, making it convenient to integrate into scripts and workflows that involve **LLMs**:

```bash
# Example usage for tokenizing a string for an LLM
tokencount tokenize-str "Hello, world!" --model gpt-4o

# Example usage for tokenizing a file for an LLM
tokencount tokenize-file TestFile.txt --model gpt-4o

# Example usage for tokenizing multiple files for an LLM
tokencount tokenize-files TestFile1.txt TestFile2.txt --model gpt-4o

# Example usage for tokenizing a directory of files for an LLM
tokencount tokenize-files MyDirectory -m gpt-4o -nr

# Example usage for tokenizing a directory of files for an LLM (alternative)
tokencount tokenize-dir MyDirectory -m gpt-4o -nr

# Example usage for counting tokens in a string for an LLM
tokencount count-str "This is a test string." --model gpt-4o

# Example usage for counting tokens in a file for an LLM
tokencount count-file TestFile.txt --model gpt-4o

# Example usage for counting tokens in multiple files for an LLM
tokencount count-files TestFile1.txt TestFile2.txt --model gpt-4o

# Example usage for counting tokens in a directory for an LLM
tokencount count-files TestDir --model gpt-4o --no-recursive

# Example usage for counting tokens in a directory for an LLM (alternative)
tokencount count-dir TestDir --model gpt-4o --no-recursive

# Example to get the model associated with an encoding
tokencount get-model cl100k_base

# Example to get the encoding associated with a model
tokencount get-encoding gpt-4o
```

**CLI Usage Details:**

The `tokencount` CLI provides several subcommands for tokenizing and counting tokens in strings, files, and directories, tailored for use with **LLMs**.

**Subcommands:**

- `tokenize-str`: Tokenizes a provided string.
  - `tokencount tokenize-str "Your string here" --model gpt-4o`
- `tokenize-file`: Tokenizes the contents of a file.
  - `tokencount tokenize-file Path/To/Your/File.txt --model gpt-4o`
- `tokenize-files`: Tokenizes the contents of multiple specified files or all files within a directory.
    - `tokencount tokenize-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
    - `tokencount tokenize-files Path/To/Your/Directory --model gpt-4o --no-recursive`
- `tokenize-dir`: Tokenizes all files within a specified directory into lists of token IDs.
    - `tokencount tokenize-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
- `count-str`: Counts the number of tokens in a provided string.
  - `tokencount count-str "Your string here" --model gpt-4o`
- `count-file`: Counts the number of tokens in a file.
  - `tokencount count-file Path/To/Your/File.txt --model gpt-4o`
- `count-files`: Counts the number of tokens in multiple specified files or all files within a directory.
  - `tokencount count-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
  - `tokencount count-files Path/To/Your/Directory --model gpt-4o --no-recursive`
- `count-dir`: Counts the total number of tokens across all files in a specified directory.
  - `tokencount count-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
- `get-model`: Retrieves the model name from the provided encoding.
  - `tokencount get-model cl100k_base`
- `get-encoding`: Retrieves the encoding name from the provided model.
  - `tokencount get-encoding gpt-4o`

**Options:**

- `-m`, `--model`: Specifies the model to use for encoding, aligning with **LLM** specifications.
- `-e`, `--encoding`: Specifies the encoding to use directly.
- `-nr`, `--no-recursive`: When used with `tokenize-files`, `tokenize-dir`, `count-files` or `count-dir` for a directory, it prevents the tool from processing subdirectories recursively.
- `-q`, `--quiet`: When used with any of the above commands, it prevents the tool from showing the progress bar.

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

#### `GetModelForEncoding(encoding: tiktoken.Encoding) -> list[str] | str`

Determines the model name(s) associated with a given encoding, facilitating the selection of appropriate **LLMs**.

**Parameters:**

- `encoding` (`tiktoken.Encoding`): The encoding to get the model for.

**Returns:**

- `str`: The model name or a list of models corresponding to the given encoding.

**Raises:**

- `ValueError`: If the encoding name is not valid.

**Example:**

```python
import PyTokenCounter as tc
import tiktoken

encoding = tiktoken.get_encoding('cl100k_base')
model = tc.GetModelForEncoding(encoding=encoding)
print(model)
```

---

#### `GetModelForEncodingName(encodingName: str) -> str`

Determines the model name associated with a given encoding name, facilitating the selection of appropriate **LLMs**.

**Parameters:**

- `encodingName` (`str`): The name of the encoding.

**Returns:**

- `str`: The model name or a list of models corresponding to the given encoding.

**Raises:**

- `ValueError`: If the encoding name is not valid.

**Example:**

```python
import PyTokenCounter as tc

modelName = tc.GetModelForEncodingName(encodingName="cl100k_base")
print(modelName)
```

---

#### `GetEncodingForModel(modelName: str) -> tiktoken.Encoding`

Retrieves the encoding associated with a given model name, ensuring accurate tokenization for the selected **LLM**.

**Parameters:**

- `modelName` (`str`): The name of the model.

**Returns:**

- `tiktoken.Encoding`: The encoding corresponding to the given model name.

**Raises:**

- `ValueError`: If the model name is not valid.

**Example:**

```python
import PyTokenCounter as tc

encoding = tc.GetEncodingForModel(modelName="gpt-4o")
print(encoding)
```

---

#### `GetEncodingNameForModel(modelName: str) -> str`

Retrieves the encoding name associated with a given model name, ensuring accurate tokenization for the selected **LLM**.

**Parameters:**

- `modelName` (`str`): The name of the model.

**Returns:**

- `str`: The encoding name corresponding to the given model name.

**Raises:**

- `ValueError`: If the model name is not valid.

**Example:**

```python
import PyTokenCounter as tc

encodingName = tc.GetEncodingNameForModel(modelName="gpt-4o")
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

encoding = tc.GetEncoding(model="gpt-4o")
print(encoding)
encoding = tc.GetEncoding(encodingName="p50k_base")
print(encoding)
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

tokens = tc.TokenizeStr(string="Hail to the Victors!", model="gpt-4o")
print(tokens)

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = tc.TokenizeStr(string="2024 National Champions", encoding=encoding)
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

numTokens = tc.GetNumTokenStr(string="Hail to the Victors!", model="gpt-4o")
print(numTokens)

numTokens = tc.GetNumTokenStr(string="Corum 4 Heisman", encoding=tiktoken.get_encoding("cl100k_base"))
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
- `UnsupportedEncodingError`: If the file encoding is not supported.
- `FileNotFoundError`: If the file does not exist

**Example:**

```python
from pathlib import Path
import PyTokenCounter as tc

filePath = Path("TestFile1.txt")
tokens = tc.TokenizeFile(filePath=filePath, model="gpt-4o")
print(tokens)

import tiktoken
encoding = tiktoken.get_encoding("p50k_base")
filePath = Path("TestFile2.txt")
tokens = tc.TokenizeFile(filePath=filePath, encoding=encoding)
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

- `TypeError`: If the types of `filePath`, `model`, `encodingName`, or `encoding` are incorrect.
- `ValueError`: If the provided `model` or `encodingName` is invalid, or if there is a mismatch between the model and encoding name, or between the provided encoding and the derived encoding.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined
- `FileNotFoundError`: If the file does not exist

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

filePath = Path("TestFile1.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o")
print(numTokens)

filePath = Path("TestFile2.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o")
print(numTokens)
```

---

#### `TokenizeFiles(inputPath: Path | str | list[Path | str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True, quiet: bool = False, exitOnListError: bool = True) -> list[int] | dict[str, list[int] | dict]`

Tokenizes multiple files or all files within a directory into lists of token IDs.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): If `inputPath` is a directory, whether to tokenize files in subdirectories recursively. Defaults to `True`.
- `quiet` (`bool`, optional): If `True`, suppress progress updates. Default is False.
- `exitOnListError` (`bool`, optional): If `True`, stop processing the list upon encountering an error. If False, skip files that cause errors. Default is True.

**Returns:**

- `list[int] | dict[str, list[int] | dict]`:
   - If `inputPath` is a file, returns a list of token IDs for that file.
   - If `inputPath` is a list of files, returns a dictionary where each key is the file name and the value is the list of token IDs for that file.
   - If `inputPath` is a directory:
    - If `recursive` is True, returns a nested dictionary where each key is a file or subdirectory name with corresponding token lists or sub-dictionaries.
    - If `recursive` is False, returns a dictionary with file names as keys and their token lists as values.

**Raises:**

- `TypeError`: If the types of `inputPath`, `model`, `encodingName`, `encoding`, or `recursive` are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory.
- `UnsupportedEncodingError`: If any of the files to be tokenized have an unsupported encoding.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
from PyTokenCounter import TokenizeFiles
from pathlib import Path

inputFiles = [
    Path("TestFile1.txt"),
    Path("TestFile2.txt"),
]
tokens = tc.TokenizeFiles(inputPath=inputFiles, model="gpt-4o")
print(tokens)

import tiktoken
encoding = tiktoken.get_encoding('p50k_base')
dirPath = Path("TestDir")
tokens = tc.TokenizeFiles(inputPath=dirPath, encoding=encoding, recursive=False)
print(tokens)

tokens = tc.TokenizeFiles(inputPath=dirPath, model="gpt-4o", recursive=True)
print(tokens)
```

---

#### `GetNumTokenFiles(inputPath: Path | str | list[Path | str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens across multiple files or in all files within a directory.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): If `inputPath` is a directory, whether to count tokens in files in subdirectories recursively. Defaults to `True`.

**Returns:**

- `int`: The total number of tokens in the specified files or directory.

**Raises:**

- `TypeError`: If the types of `inputPath`, `model`, `encodingName`, `encoding`, or `recursive` are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory, or if the provided model or encoding is invalid.
- `UnsupportedEncodingError`: If any of the files to be tokenized have an unsupported encoding.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

inputFiles = [
    Path("TestFile1.txt"),
    Path("TestFile2.txt"),
]
numTokens = tc.GetNumTokenFiles(inputPath=inputFiles, model='gpt-4o')
print(numTokens)


import tiktoken
encoding = tiktoken.get_encoding('p50k_base')
dirPath = Path("TestDir")
numTokens = tc.GetNumTokenFiles(inputPath=dirPath, encoding=encoding, recursive=False)
print(numTokens)
numTokens = tc.GetNumTokenFiles(inputPath=dirPath, model='gpt-4o', recursive=True)
print(numTokens)
```

---

#### `TokenizeDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> dict[str, list[int] | dict]`

Tokenizes all files within a directory into lists of token IDs.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): Whether to tokenize files in subdirectories recursively. Defaults to `True`.

**Returns:**

- `dict[str, list[int] | dict]`: A nested dictionary where each key is a file or subdirectory name:
    - If the key is a file, its value is a list of token IDs.
    - If the key is a subdirectory, its value is another dictionary following the same structure.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined.
- `FileNotFoundError`: If the directory does not exist.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

dirPath = Path("TestDir")
tokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=True)
print(tokenizedDir)

tokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(tokenizedDir)
```

---

#### `GetNumTokenDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens in all files within a directory.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding.
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): Whether to count tokens in subdirectories recursively. Defaults to `True`.

**Returns:**

- `int`: The total number of tokens in the directory.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined.
- `FileNotFoundError`: If the directory does not exist.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path

dirPath = Path("TestDir")
numTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=True)
print(numTokensDir)
numTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(numTokensDir)
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