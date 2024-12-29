# PyTokenCounter

PyTokenCounter is a Python library designed to simplify text tokenization and token counting. It supports various encoding schemes, with a focus on those used by **Large Language Models (LLMs)**, particularly those developed by OpenAI. Leveraging the `tiktoken` library for efficient processing, PyTokenCounter facilitates seamless integration with LLM workflows. This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

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

import PyTokenCounter as ptc
import tiktoken

# Count tokens in a string for an LLM model
numTokens = ptc.GetNumTokenStr(
    string="This is a test string.", model="gpt-3.5-turbo"
)
print(f"Number of tokens: {numTokens}")

# Count tokens in a file intended for LLM processing
numTokensFile = ptc.GetNumTokenFile(
    filePath=Path("./test_file.txt"), model="gpt-4"
)
print(f"Number of tokens in file: {numTokensFile}")

# Count tokens in a directory of documents for batch processing with an LLM
numTokensDir = ptc.GetNumTokenDir(
    dirPath=Path("./test_dir"), model="gpt-4o", recursive=True
)
print(f"Number of tokens in directory: {numTokensDir}")

# Get the encoding for a specific LLM model
encoding = ptc.GetEncoding(model="gpt-3.5-turbo")

# Tokenize a string using a specific encoding for LLM input
tokens = ptc.TokenizeStr(string="This is another test.", encoding=encoding)
print(f"Token IDs: {tokens}")
```

### CLI

PyTokenCounter can also be used as a command-line tool, making it convenient to integrate into scripts and workflows that involve **LLMs**:

```bash
# Example usage for tokenizing a string for an LLM
ptc tokenize-str "This is a test string." --model gpt-3.5-turbo

# Example usage for tokenizing a file for an LLM
ptc tokenize-file test_file.txt --model gpt-4

# Example usage for tokenizing a directory of files for an LLM
ptc tokenize-dir test_dir --model gpt-4o --no-recursive

# Example usage for counting tokens in a string for an LLM
ptc count-str "This is a test string." --model gpt-3.5-turbo

# Example usage for counting tokens in a file for an LLM
ptc count-file test_file.txt --model gpt-4

# Example usage for counting tokens in a directory for an LLM
ptc count-dir test_dir --model gpt-4o --no-recursive
```

**CLI Usage Details:**

The `ptc` CLI provides several subcommands for tokenizing and counting tokens in strings, files, and directories, tailored for use with **LLMs**.

**Subcommands:**

-   `tokenize-str`: Tokenizes a provided string.
    -   `ptc tokenize-str "Your string here" --model gpt-3.5-turbo`
-   `tokenize-file`: Tokenizes the contents of a file.
    -   `ptc tokenize-file path/to/your/file.txt --model gpt-4`
-   `tokenize-dir`: Tokenizes all files in a directory.
    -   `ptc tokenize-dir path/to/your/directory --model gpt-4o --no-recursive`
-   `count-str`: Counts the number of tokens in a provided string.
    -   `ptc count-str "Your string here" --model gpt-3.5-turbo`
-   `count-file`: Counts the number of tokens in a file.
    -   `ptc count-file path/to/your/file.txt --model gpt-4`
-   `count-dir`: Counts the total number of tokens in all files within a directory.
    -   `ptc count-dir path/to/your/directory --model gpt-4o --no-recursive`

**Options:**

-   `-m`, `--model`: Specifies the model to use for encoding, aligning with **LLM** specifications.
-   `-e`, `--encoding`: Specifies the encoding to use directly.
-   `-nr`, `--no-recursive`: When used with `tokenize-dir` or `count-dir`, it prevents the tool from processing subdirectories recursively.

**Note:** For detailed help on each subcommand, use `ptc <subcommand> -h`.

## API

Here's a detailed look at the PyTokenCounter API, designed to integrate seamlessly with **LLM** workflows:

### `GetModelMappings() -> dict`

Retrieves the mappings between models and their corresponding encodings, essential for selecting the correct tokenization strategy for different **LLMs**.

**Returns:**

-   `dict`: A dictionary where keys are model names and values are their corresponding encodings.

**Example:**

```python
import PyTokenCounter as ptc

modelMappings = ptc.GetModelMappings()
print(modelMappings)
```

---

### `GetValidModels() -> list[str]`

Returns a list of valid model names supported by PyTokenCounter, primarily focusing on **LLMs**.

**Returns:**

-   `list[str]`: A list of valid model names.

**Example:**

```python
import PyTokenCounter as ptc

validModels = ptc.GetValidModels()
print(validModels)
```

---

### `GetValidEncodings() -> list[str]`

Returns a list of valid encoding names, ensuring compatibility with various **LLMs**.

**Returns:**

-   `list[str]`: A list of valid encoding names.

**Example:**

```python
import PyTokenCounter as ptc

validEncodings = ptc.GetValidEncodings()
print(validEncodings)
```

---

### `GetModelForEncoding(encodingName: str) -> str`

Determines the model name associated with a given encoding, facilitating the selection of appropriate **LLMs**.

**Parameters:**

-   `encodingName` (`str`): The name of the encoding.

**Returns:**

-   `str`: The model name corresponding to the given encoding.

**Raises:**

-   `ValueError`: If the encoding name is not valid.

**Example:**

```python
import PyTokenCounter as ptc

modelName = ptc.GetModelForEncoding(encodingName="cl100k_base")
print(modelName)
```

---

### `GetEncodingForModel(modelName: str) -> str`

Retrieves the encoding associated with a given model name, ensuring accurate tokenization for the selected **LLM**.

**Parameters:**

-   `modelName` (`str`): The name of the model.

**Returns:**

-   `str`: The encoding corresponding to the given model name.

**Raises:**

-   `ValueError`: If the model name is not valid.

**Example:**

```python
import PyTokenCounter as ptc

encodingName = ptc.GetEncodingForModel(modelName="gpt-3.5-turbo")
print(encodingName)
```

---

### `GetEncoding(model: str | None = None, encodingName: str | None = None) -> tiktoken.Encoding`

Obtains the `tiktoken` encoding based on the specified model or encoding name, tailored for **LLM** usage.

**Parameters:**

-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.

**Returns:**

-   `tiktoken.Encoding`: The `tiktoken` encoding object.

**Raises:**

-   `ValueError`: If neither model nor encoding is provided, or if the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as ptc
import tiktoken

encoding = ptc.GetEncoding(model="gpt-4")
print(type(encoding))

encoding = ptc.GetEncoding(encodingName="cl100k_base")
print(type(encoding))
```

---

### `TokenizeStr(string: str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> list[int]`

Tokenizes a string into a list of token IDs, preparing text for input into an **LLM**.

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
import PyTokenCounter as ptc

tokens = ptc.TokenizeStr(string="This is a test string.", model="gpt-3.5-turbo")
print(tokens)

encoding = ptc.GetEncoding(encodingName="cl100k_base")
tokens = ptc.TokenizeStr(string="This is another test.", encoding=encoding)
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
import PyTokenCounter as ptc

numTokens = ptc.GetNumTokenStr(string="This is a test string.", model="gpt-4")
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
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'test_file.txt' exists with some content
filePath = Path("./test_file.txt")
tokens = ptc.TokenizeFile(filePath=filePath, model="gpt-3.5-turbo")
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
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'test_file.txt' exists
filePath = Path("./test_file.txt")
numTokens = ptc.GetNumTokenFile(filePath=filePath, model="gpt-4")
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
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'file1.txt' and 'file2.txt' exist
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
tokenizedFiles = ptc.TokenizeFiles(filePaths=filePaths, model="gpt-3.5-turbo")
print(tokenizedFiles)
```

---

### `GetNumTokenFiles(filePaths: list[Path] | list[str], model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> int`

Counts the number of tokens across multiple files.

**Parameters:**

-   `filePaths` (`list[Path] | list[str]`): A list of paths to the files.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.

**Returns:**

-   `int`: The total number of tokens across all files.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'file1.txt' and 'file2.txt' exist
filePaths = [Path("./file1.txt"), Path("./file2.txt")]
numTokens = ptc.GetNumTokenFiles(filePaths=filePaths, model="gpt-3.5-turbo")
print(numTokens)
```

---

### `TokenizeDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> list[int | list] | list[int]`

Tokenizes all files within a directory into lists of token IDs.

**Parameters:**

-   `dirPath` (`Path | str`): The path to the directory.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.
-   `recursive` (`bool`, optional): Whether to tokenize subdirectories recursively. Defaults to `True`.

**Returns:**

-   `list[int | list] | list[int]`: A nested list of token IDs for each file (recursive) or a list of token IDs (non-recursive).

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'test_dir' exists and contains some files
dirPath = Path("./test_dir")
tokenizedDir = ptc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(tokenizedDir)
```

---

### `GetNumTokenDir(dirPath: Path | str, model: str | None = None, encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True) -> int`

Counts the number of tokens in all files within a directory.

**Parameters:**

-   `dirPath` (`Path | str`): The path to the directory.
-   `model` (`str`, optional): The name of the model.
-   `encodingName` (`str`, optional): The name of the encoding.
-   `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object.
-   `recursive` (`bool`, optional): Whether to count tokens in subdirectories recursively. Defaults to `True`.

**Returns:**

-   `int`: The total number of tokens in the directory.

**Raises:**

-   `TypeError`: If the types of input parameters are incorrect.
-   `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as ptc
from pathlib import Path

# Assuming 'test_dir' exists and contains some files
dirPath = Path("./test_dir")
numTokens = ptc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(numTokens)
```

## Maintainers

- [Kaden Gruizenga](https://github.com/kgruiz)

## Acknowledgements

- This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Contributing

Contributions are welcome! Feel free to [open an issue](https://github.com/kgruiz/PyTokenCounter/issues/new) or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.