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
  - [Token Mapping](#token-mapping)
- [Ignored Files](#ignored-files)
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
from collections import OrderedDict

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

# Map tokens to their decoded strings
mappedTokens = tc.MapTokens(tokens=tokens, encoding=encoding)
print(f"Mapped tokens: {mappedTokens}")

# Count tokens in a string using the default model
numTokens = tc.GetNumTokenStr(string="This is a test string.")
print(f"Number of tokens: {numTokens}")

# Count tokens in a file using the default model
filePath = Path("./TestFile.txt")
numTokensFile = tc.GetNumTokenFile(filePath=filePath)
print(f"Number of tokens in file: {numTokensFile}")

# Tokenize a string using the default model
tokens = tc.TokenizeStr(string="This is another test.")
print(f"Token IDs: {tokens}")

# Tokenize a string and map tokens to strings using the default model
mappedTokensResult = tc.TokenizeStr(string="This is another test.", mapTokens=True)
print(f"Mapped tokens result: {mappedTokensResult}")

# Map tokens to their decoded strings using the default model
mappedTokens = tc.MapTokens(tokens=tokens)
print(f"Mapped tokens: {mappedTokens}")

# Tokenize a directory and get mapped tokens with counts
dirPath = Path("./TestDir")
mappedDirTokens = tc.TokenizeDir(dirPath=dirPath, recursive=True, mapTokens=True)
print(f"Mapped directory tokens: {mappedDirTokens}")

# Count tokens in a directory and get mapped counts
mappedDirCounts = tc.GetNumTokenDir(dirPath=dirPath, recursive=True, mapTokens=True)
print(f"Mapped directory counts: {mappedDirCounts}")
```

### CLI

PyTokenCounter can also be used as a command-line tool. You can use either the `tokencount` entry point or its alias `tc`. Below are examples for both:

```bash
# Tokenize a string for an LLM
tokencount tokenize-str "Hello, world!" --model gpt-4o
tc tokenize-str "Hello, world!" --model gpt-4o

# Tokenize a string using the default model
tokencount tokenize-str "Hello, world!"
tc tokenize-str "Hello, world!"

# Tokenize a file for an LLM
tokencount tokenize-file TestFile.txt --model gpt-4o
tc tokenize-file TestFile.txt --model gpt-4o

# Tokenize a file using the default model
tokencount tokenize-file TestFile.txt
tc tokenize-file TestFile.txt

# Tokenize multiple files for an LLM
tokencount tokenize-files TestFile1.txt TestFile2.txt --model gpt-4o
tc tokenize-files TestFile1.txt TestFile2.txt --model gpt-4o

# Tokenize multiple files using the default model
tokencount tokenize-files TestFile1.txt TestFile2.txt
tc tokenize-files TestFile1.txt TestFile2.txt

# Tokenize a directory of files for an LLM (non-recursive)
tokencount tokenize-files MyDirectory --model gpt-4o --no-recursive
tc tokenize-files MyDirectory --model gpt-4o --no-recursive

# Tokenize a directory of files using the default model (non-recursive)
tokencount tokenize-files MyDirectory --no-recursive
tc tokenize-files MyDirectory --no-recursive

# Tokenize a directory (alternative command) for an LLM (non-recursive)
tokencount tokenize-dir MyDirectory --model gpt-4o --no-recursive
tc tokenize-dir MyDirectory --model gpt-4o --no-recursive

# Tokenize a directory (alternative command) using the default model (non-recursive)
tokencount tokenize-dir MyDirectory --no-recursive
tc tokenize-dir MyDirectory --no-recursive

# Count tokens in a string for an LLM
tokencount count-str "This is a test string." --model gpt-4o
tc count-str "This is a test string." --model gpt-4o

# Count tokens in a string using the default model
tokencount count-str "This is a test string."
tc count-str "This is a test string."

# Count tokens in a file for an LLM
tokencount count-file TestFile.txt --model gpt-4o
tc count-file TestFile.txt --model gpt-4o

# Count tokens in a file using the default model
tokencount count-file TestFile.txt
tc count-file TestFile.txt

# Count tokens in multiple files for an LLM
tokencount count-files TestFile1.txt TestFile2.txt --model gpt-4o
tc count-files TestFile1.txt TestFile2.txt --model gpt-4o

# Count tokens in multiple files using the default model
tokencount count-files TestFile1.txt TestFile2.txt
tc count-files TestFile1.txt TestFile2.txt

# Count tokens in a directory for an LLM (non-recursive)
tokencount count-files TestDir --model gpt-4o --no-recursive
tc count-files TestDir --model gpt-4o --no-recursive

# Count tokens in a directory using the default model (non-recursive)
tokencount count-files TestDir --no-recursive
tc count-files TestDir --no-recursive

# Count tokens in a directory (alternative command) for an LLM (non-recursive)
tokencount count-dir TestDir --model gpt-4o --no-recursive
tc count-dir TestDir --model gpt-4o --no-recursive

# Count tokens in a directory (alternative command) using the default model (non-recursive)
tokencount count-dir TestDir --no-recursive
tc count-dir TestDir --no-recursive

# Get the model associated with an encoding
tokencount get-model cl100k_base
tc get-model cl100k_base

# Get the encoding associated with a model
tokencount get-encoding gpt-4o
tc get-encoding gpt-4o

# Map tokens to strings for an LLM
tokencount map-tokens 123,456,789 --model gpt-4o
tc map-tokens 123,456,789 --model gpt-4o

# Map tokens to strings using the default model
tokencount map-tokens 123,456,789
tc map-tokens 123,456,789

# Tokenize a string and output mapped tokens
tokencount tokenize-str "Hello, mapped world!" --model gpt-4o -M

# Tokenize a file and output mapped tokens
tokencount tokenize-file TestFile.txt --model gpt-4o -M

# Tokenize a directory and output mapped tokens
tokencount tokenize-dir MyDirectory --model gpt-4o -M

# Count tokens in a directory and output mapped counts
tokencount count-dir MyDirectory --model gpt-4o -M

# Include binary files
tokencount tokenize-files MyDirectory --model gpt-4o -b
tc tokenize-files MyDirectory --model gpt-4o -b

# Include hidden files and directories
tokencount tokenize-files MyDirectory --model gpt-4o -H
tc tokenize-files MyDirectory --model gpt-4o -H

# Combine both options: include binary files and include hidden files
tokencount tokenize-files MyDirectory --model gpt-4o -b -H
tc tokenize-files MyDirectory --model gpt-4o -b -H
```

**CLI Usage Details:**

The `tokencount` (or `tc`) CLI provides several subcommands for tokenizing and counting tokens in strings, files, and directories, tailored for use with **LLMs**.

**Subcommands:**

- `tokenize-str`: Tokenizes a provided string.
  - `tokencount tokenize-str "Your string here" --model gpt-4o`
  - `tc tokenize-str "Your string here" --model gpt-4o`
- `tokenize-file`: Tokenizes the contents of a file.
  - `tokencount tokenize-file Path/To/Your/File.txt --model gpt-4o`
  - `tc tokenize-file Path/To/Your/File.txt --model gpt-4o`
- `tokenize-files`: Tokenizes the contents of multiple specified files or all files within a directory.
    - `tokencount tokenize-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
    - `tc tokenize-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
    - `tokencount tokenize-files Path/To/Your/Directory --model gpt-4o --no-recursive`
    - `tc tokenize-files Path/To/Your/Directory --model gpt-4o --no-recursive`
- `tokenize-dir`: Tokenizes all files within a specified directory into lists of token IDs.
    - `tokencount tokenize-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
    - `tc tokenize-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
- `count-str`: Counts the number of tokens in a provided string.
  - `tokencount count-str "Your string here" --model gpt-4o`
  - `tc count-str "Your string here" --model gpt-4o`
- `count-file`: Counts the number of tokens in a file.
  - `tokencount count-file Path/To/Your/File.txt --model gpt-4o`
  - `tc count-file Path/To/Your/File.txt --model gpt-4o`
- `count-files`: Counts the number of tokens in multiple specified files or all files within a directory.
  - `tokencount count-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
  - `tc count-files Path/To/Your/File1.txt Path/To/Your/File2.txt --model gpt-4o`
  - `tokencount count-files Path/To/Your/Directory --model gpt-4o --no-recursive`
  - `tc count-files Path/To/Your/Directory --model gpt-4o --no-recursive`
- `count-dir`: Counts the total number of tokens across all files in a specified directory.
  - `tokencount count-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
  - `tc count-dir Path/To/Your/Directory --model gpt-4o --no-recursive`
- `get-model`: Retrieves the model name from the provided encoding.
  - `tokencount get-model cl100k_base`
  - `tc get-model cl100k_base`
- `get-encoding`: Retrieves the encoding name from the provided model.
  - `tokencount get-encoding gpt-4o`
  - `tc get-encoding gpt-4o`
- `map-tokens`: Maps a list of token integers to their decoded strings.
    - `tokencount map-tokens 123,456,789 --model gpt-4o`
    - `tc map-tokens 123,456,789 --model gpt-4o`

**Options:**

- `-m`, `--model`: Specifies the model to use for encoding. **Default: `gpt-4o`**
- `-e`, `--encoding`: Specifies the encoding to use directly.
- `-nr`, `--no-recursive`: When used with `tokenize-files`, `tokenize-dir`, `count-files`, or `count-dir` for a directory, it prevents the tool from processing subdirectories recursively.
- `-q`, `--quiet`: When used with any of the above commands, it prevents the tool from showing progress bars and minimizes output.
- `-M`, `--mapTokens`: When specified, the output will be in a mapped (nested) format. For tokenize commands, this outputs a nested `OrderedDict` mapping decoded strings to their token IDs. For count commands, this outputs a nested `OrderedDict` with token counts, including keys such as `"numTokens"` and `"tokens"`.
- `-o`, `--output`: When used with any of the commands, specifies an output JSON file to save the results to.
- `-b`, `--include-binary`: Include binary files in processing. (Default: binary files are excluded.)
- `-H`, `--include-hidden`: Include hidden files and directories. (Default: hidden files and directories are skipped.)

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

Obtains the `tiktoken` encoding based on the specified model or encoding name, tailored for **LLM** usage. If neither `model` nor `encodingName` is provided, it defaults to the encoding associated with the `"gpt-4o"` model.

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
encoding = tc.GetEncoding()
print(encoding)
```

---

### String Tokenization and Counting

#### `TokenizeStr(string: str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, quiet: bool = False, mapTokens: bool = False) -> list[int] | OrderedDict[str, int]`

Tokenizes a string into a list of token IDs or a mapping of decoded strings to tokens, preparing text for input into an **LLM**.

**Parameters:**

- `string` (`str`): The string to tokenize.
- `model` (`str`, optional): The name of the model. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken` encoding object.
- `quiet` (`bool`, optional): If `True`, suppresses progress updates.
- `mapTokens` (`bool`, optional): If `True`, outputs an `OrderedDict` mapping decoded strings to their token IDs. **Default: `False`**

**Returns:**

- `list[int]`: A list of token IDs if `mapTokens` is `False`.
- `OrderedDict[str, int]`: An `OrderedDict` mapping decoded strings to token IDs if `mapTokens` is `True`.

**Raises:**

- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from collections import OrderedDict

tokens = tc.TokenizeStr(string="Hail to the Victors!", model="gpt-4o")
print(tokens)

tokens = tc.TokenizeStr(string="Hail to the Victors!")
print(tokens)


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
mappedTokens = tc.TokenizeStr(string="2024 National Champions", encoding=encoding, mapTokens=True)
print(mappedTokens)

mappedTokens = tc.TokenizeStr(string="2024 National Champions", mapTokens=True)
print(mappedTokens)
```

---

#### `GetNumTokenStr(string: str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, quiet: bool = False, mapTokens: bool = False) -> int | OrderedDict[str, int]`

Counts the number of tokens in a string.

**Parameters:**

- `string` (`str`): The string to count tokens in.
- `model` (`str`, optional): The name of the model. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding.
- `encoding` (`tiktoken.Encoding`, optional): A `tiktoken.Encoding` object to use for tokenization.
- `quiet` (`bool`, optional): If `True`, suppresses progress updates.
- `mapTokens` (`bool`, optional): If `True`, outputs an `OrderedDict` mapping decoded strings to token counts (which are always 1 for strings). Primarily for consistency with other functions.  **Default: `False`**

**Returns:**

- `int`: The number of tokens in the string if `mapTokens` is `False`.
- `OrderedDict[str, int]`: An `OrderedDict` mapping decoded strings to token counts if `mapTokens` is `True`.

**Raises:**

- `ValueError`: If the provided model or encoding is invalid.

**Example:**

```python
import PyTokenCounter as tc
from collections import OrderedDict
import tiktoken

numTokens = tc.GetNumTokenStr(string="Hail to the Victors!", model="gpt-4o")
print(numTokens)

numTokens = tc.GetNumTokenStr(string="Hail to the Victors!")
print(numTokens)

numTokens = tc.GetNumTokenStr(string="Corum 4 Heisman", encoding=tiktoken.get_encoding("cl100k_base"))
print(numTokens)

numTokens = tc.GetNumTokenStr(string="Corum 4 Heisman")
print(numTokens)

mappedCounts = tc.GetNumTokenStr(string="Mapped count example", mapTokens=True)
print(mappedCounts)
```

---

### File and Directory Tokenization and Counting

#### `TokenizeFile(filePath: Path | str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, quiet: bool = False, mapTokens: bool = False) -> list[int] | OrderedDict[str, OrderedDict[str, int | list[int]]]`

Tokenizes the contents of a file into a list of token IDs or a nested `OrderedDict` structure.

**Parameters:**

- `filePath` (`Path | str`): The path to the file to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `quiet` (`bool`, optional): If `True`, suppresses progress updates.
- `mapTokens` (`bool`, optional): If `True`, outputs an `OrderedDict` where the key is the filename and the value is another `OrderedDict` with keys `"tokens"` (the list of token IDs) and `"numTokens"` (the total token count). If `False`, returns just the list of token IDs. **Default: `False`**

**Returns:**

- `list[int]`: A list of token IDs representing the tokenized file contents if `mapTokens` is `False`.
- `OrderedDict[str, OrderedDict[str, int | list[int]]]`: An `OrderedDict` as described above if `mapTokens` is `True`.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided model or encoding is invalid.
- `UnsupportedEncodingError`: If the file encoding is not supported.
- `FileNotFoundError`: If the file does not exist.

**Example:**

```python
from pathlib import Path
import PyTokenCounter as tc
from collections import OrderedDict
import tiktoken

filePath = Path("TestFile1.txt")
tokens = tc.TokenizeFile(filePath=filePath, model="gpt-4o")
print(tokens)

filePath = Path("TestFile1.txt")
mappedTokensFile = tc.TokenizeFile(filePath=filePath, model="gpt-4o", mapTokens=True)
print(mappedTokensFile)

filePath = Path("TestFile1.txt")
tokens = tc.TokenizeFile(filePath=filePath)
print(tokens)

import tiktoken
encoding = tiktoken.get_encoding("p50k_base")
filePath = Path("TestFile2.txt")
mappedTokensFileEncoding = tc.TokenizeFile(filePath=filePath, encoding=encoding, mapTokens=True)
print(mappedTokensFileEncoding)

filePath = Path("TestFile2.txt")
mappedTokensFileDefault = tc.TokenizeFile(filePath=filePath, mapTokens=True)
print(mappedTokensFileDefault)
```

---

#### `GetNumTokenFile(filePath: Path | str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, quiet: bool = False, mapTokens: bool = False) -> int | OrderedDict[str, int]`

Counts the number of tokens in a file based on the specified model or encoding.

**Parameters:**

- `filePath` (`Path | str`): The path to the file to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `quiet` (`bool`, optional): If `True`, suppresses progress updates.
- `mapTokens` (`bool`, optional): If `True`, outputs an `OrderedDict` where the key is the filename and the value is the token count. If `False`, returns just the token count as an integer. **Default: `False`**

**Returns:**

- `int`: The number of tokens in the file if `mapTokens` is `False`.
- `OrderedDict[str, int]`: An `OrderedDict` mapping the filename to its token count if `mapTokens` is `True`.

**Raises:**

- `TypeError`: If the types of `filePath`, `model`, `encodingName`, or `encoding` are incorrect.
- `ValueError`: If the provided `model` or `encodingName` is invalid, or if there is a mismatch between the model and encoding name, or between the provided encoding and the derived encoding.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined.
- `FileNotFoundError`: If the file does not exist.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path
from collections import OrderedDict

filePath = Path("TestFile1.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o")
print(numTokens)

filePath = Path("TestFile1.txt")
mappedNumTokensFile = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o", mapTokens=True)
print(mappedNumTokensFile)

filePath = Path("TestFile1.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath)
print(numTokens)

filePath = Path("TestFile2.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath, model="gpt-4o")
print(numTokens)

filePath = Path("TestFile2.txt")
numTokens = tc.GetNumTokenFile(filePath=filePath)
print(numTokens)

filePath = Path("TestFile2.txt")
mappedNumTokensFileDefault = tc.GetNumTokenFile(filePath=filePath, mapTokens=True)
print(mappedNumTokensFileDefault)
```

---

#### `TokenizeFiles(inputPath: Path | str | list[Path | str], model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True, quiet: bool = False, exitOnListError: bool = True, mapTokens: bool = False, excludeBinary: bool = True, includeHidden: bool = False) -> list[int] | OrderedDict[str, list[int] | OrderedDict]`

Tokenizes multiple files or all files within a directory into lists of token IDs or a nested `OrderedDict` structure.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): If `inputPath` is a directory, whether to tokenize files in subdirectories recursively. **Default: `True`**
- `quiet` (`bool`, optional): If `True`, suppresses progress updates. **Default: `False`**
- `exitOnListError` (`bool`, optional): If `True`, stop processing the list upon encountering an error. If `False`, skip files that cause errors. **Default: `True`**
- `mapTokens` (`bool`, optional): If `True`, outputs a nested `OrderedDict` structure. For files, the value is an `OrderedDict` with keys `"tokens"` (the list of token IDs) and `"numTokens"` (the total token count). For directories, the output is wrapped with `"tokens"` and `"numTokens"` keys. If `False`, returns a list of token IDs for a single file, or a dictionary mapping filenames to token lists for multiple files. **Default: `False`**
- `excludeBinary` (`bool`, optional): Excludes any binary files by skipping over them. **Default: `True`**
- `includeHidden` (`bool`, optional): Skips over hidden files and directories, including subdirectories and files of a hidden directory. **Default: `False`**

**Returns:**

- `list[int] | OrderedDict[str, list[int] | OrderedDict]`:
   - If `inputPath` is a file:
     - If `mapTokens` is `False`, returns a list of token IDs for that file.
     - If `mapTokens` is `True`, returns an `OrderedDict` with the structure described in the `mapTokens` parameter description.
   - If `inputPath` is a list of files, returns a dictionary where each key is the file name and the value depends on `mapTokens`:
     - If `mapTokens` is `False`, the value is the list of token IDs for that file.
     - If `mapTokens` is `True`, the value is an `OrderedDict` with the structure described in the `mapTokens` parameter description, wrapped under the key `"tokens"`, and the total token count under the key `"numTokens"` at the top level.
   - If `inputPath` is a directory:
     - If `recursive` is `True`, returns a nested `OrderedDict` where each key is a file or subdirectory name with corresponding token lists or sub-dictionaries. If `mapTokens` is `True`, the directory output is wrapped with `"tokens"` and `"numTokens"` keys.
     - If `recursive` is `False`, returns a dictionary with file names as keys and their token lists as values. If `mapTokens` is `True`, the directory output is wrapped with `"tokens"` and `"numTokens"` keys.

**Raises:**

- `TypeError`: If the types of `inputPath`, `model`, `encodingName`, `encoding`, or `recursive` are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory.
- `UnsupportedEncodingError`: If any of the files to be tokenized have an unsupported encoding.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
from PyTokenCounter import TokenizeFiles
from pathlib import Path
from collections import OrderedDict
import tiktoken

inputFiles = [
    Path("TestFile1.txt"),
    Path("TestFile2.txt"),
]
tokens = tc.TokenizeFiles(inputPath=inputFiles, model="gpt-4o")
print(tokens)

mappedTokensFiles = tc.TokenizeFiles(inputPath=inputFiles, model="gpt-4o", mapTokens=True)
print(mappedTokensFiles)

tokens = tc.TokenizeFiles(inputPath=inputFiles)
print(tokens)

# Tokenizing multiple files using the default model
mappedTokensFilesDefault = tc.TokenizeFiles(inputPath=inputFiles, mapTokens=True)
print(mappedTokensFilesDefault)

import tiktoken
encoding = tiktoken.get_encoding('p50k_base')
dirPath = Path("TestDir")
mappedDirTokensNonRecursiveEncoding = tc.TokenizeFiles(inputPath=dirPath, encoding=encoding, recursive=False, mapTokens=True)
print(mappedDirTokensNonRecursiveEncoding)

mappedDirTokensRecursiveModel = tc.TokenizeFiles(inputPath=dirPath, model="gpt-4o", recursive=True, mapTokens=True)
print(mappedDirTokensRecursiveModel)

mappedDirTokensRecursiveDefault = tc.TokenizeFiles(inputPath=dirPath, recursive=True, mapTokens=True)
print(mappedDirTokensRecursiveDefault)

# Tokenizing a directory using the default model
mappedDirTokensDefaultModel = tc.TokenizeFiles(inputPath=dirPath, recursive=True, mapTokens=True)
print(mappedDirTokensDefaultModel)
```

---

#### `GetNumTokenFiles(inputPath: Path | str | list[Path | str], model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True, quiet: bool = False, exitOnListError: bool = True, excludeBinary: bool = True, includeHidden: bool = False, mapTokens: bool = False) -> int | OrderedDict[str, int | OrderedDict]`

Counts the number of tokens across multiple files or in all files within a directory, or returns a nested `OrderedDict` structure with counts.

**Parameters:**

- `inputPath` (`Path | str | list[Path | str]`): The path to a file or directory, or a list of file paths to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): If `inputPath` is a directory, whether to count tokens in files in subdirectories recursively. **Default: `True`**
- `quiet` (`bool`, optional): If `True`, suppresses progress updates. **Default: `False`**
- `exitOnListError` (`bool`, optional): If `True`, stop processing the list upon encountering an error. If `False`, skip files that cause errors. **Default: `True`**
- `excludeBinary` (`bool`, optional): Excludes any binary files by skipping over them. **Default: `True`**
- `includeHidden` (`bool`, optional): Skips over hidden files and directories, including subdirectories and files of a hidden directory. **Default: `False`**
- `mapTokens` (`bool`, optional): If `True`, outputs a nested `OrderedDict` structure. For files, the value is the token count. For directories, the output is wrapped with `"tokens"` and `"numTokens"` keys, where `"tokens"` contains the nested counts. If `False`, returns the total token count as an integer. **Default: `False`**

**Returns:**

- `int`: The total number of tokens in the specified files or directory if `mapTokens` is `False`.
- `OrderedDict[str, int | OrderedDict]`: An `OrderedDict` mirroring the directory structure with token counts if `mapTokens` is `True`.

**Raises:**

- `TypeError`: If the types of `inputPath`, `model`, `encodingName`, `encoding`, or `recursive` are incorrect.
- `ValueError`: If any of the provided file paths in a list are not files, or if a provided directory path is not a directory, or if the provided model or encoding is invalid.
- `UnsupportedEncodingError`: If any of the files to be tokenized have an unsupported encoding.
- `RuntimeError`: If the provided `inputPath` is neither a file, a directory, nor a list.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path
from collections import OrderedDict
import tiktoken

inputFiles = [
    Path("TestFile1.txt"),
    Path("TestFile2.txt"),
]
numTokens = tc.GetNumTokenFiles(inputPath=inputFiles, model='gpt-4o')
print(numTokens)

mappedNumTokensFiles = tc.GetNumTokenFiles(inputPath=inputFiles, model='gpt-4o', mapTokens=True)
print(mappedNumTokensFiles)


numTokens = tc.GetNumTokenFiles(inputPath=inputFiles)
print(numTokens)

# Counting tokens in multiple files using the default model
mappedNumTokensFilesDefault = tc.GetNumTokenFiles(inputPath=inputFiles, mapTokens=True)
print(mappedNumTokensFilesDefault)


import tiktoken
encoding = tiktoken.get_encoding('p50k_base')
dirPath = Path("TestDir")
mappedNumTokensDirNonRecursiveEncoding = tc.GetNumTokenFiles(inputPath=dirPath, encoding=encoding, recursive=False, mapTokens=True)
print(mappedNumTokensDirNonRecursiveEncoding)
numTokensDirRecursiveModel = tc.GetNumTokenFiles(inputPath=dirPath, model='gpt-4o', recursive=True)
print(numTokensDirRecursiveModel)

mappedNumTokensDirRecursiveModel = tc.GetNumTokenFiles(inputPath=dirPath, model='gpt-4o', recursive=True, mapTokens=True)
print(mappedNumTokensDirRecursiveModel)


# Counting tokens in a directory using the default model
mappedNumTokensDirRecursiveDefault = tc.GetNumTokenFiles(inputPath=dirPath, recursive=True, mapTokens=True)
print(mappedNumTokensDirRecursiveDefault)
```

---

#### `TokenizeDir(dirPath: Path | str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True, quiet: bool = False, mapTokens: bool = False, excludeBinary: bool = True, includeHidden: bool = False) -> OrderedDict[str, list[int] | OrderedDict]`

Tokenizes all files within a directory into a nested `OrderedDict` structure or lists of token IDs.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory to tokenize.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): Whether to tokenize files in subdirectories recursively. **Default: `True`**
- `quiet` (`bool`, optional): If `True`, suppresses progress updates. **Default: `False`**
- `mapTokens` (`bool`, optional): If `True`, outputs a nested `OrderedDict` where each key is a file or subdirectory name. For files, the value is an `OrderedDict` with keys `"tokens"` (the list of token IDs) and `"numTokens"` (the total token count). For directories, the output is recursively structured. If `False`, returns a nested `OrderedDict` of token lists without the `"numTokens"` wrapper. **Default: `False`**
- `excludeBinary` (`bool`, optional): Excludes any binary files by skipping over them. **Default: `True`**
- `includeHidden` (`bool`, optional): Skips over hidden files and directories, including subdirectories and files of a hidden directory. **Default: `False`**

**Returns:**

- `OrderedDict[str, list[int] | OrderedDict]`: A nested `OrderedDict` where each key is a file or subdirectory name. If `mapTokens` is `True`, directory entries include `"numTokens"` and `"tokens"` keys.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined.
- `FileNotFoundError`: If the directory does not exist.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path
from collections import OrderedDict

dirPath = Path("TestDir")
tokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=True)
print(tokenizedDir)

mappedTokenizedDir = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=True, mapTokens=True)
print(mappedTokenizedDir)

tokenizedDir = tc.TokenizeDir(dirPath=dirPath, recursive=True)
print(tokenizedDir)

# Tokenizing a directory using the default model
mappedTokenizedDirDefault = tc.TokenizeDir(dirPath=dirPath, recursive=True, mapTokens=True)
print(mappedTokenizedDirDefault)

tokenizedDirNonRecursiveModel = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(tokenizedDirNonRecursiveModel)

mappedTokenizedDirNonRecursiveModel = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=False, mapTokens=True)
print(mappedTokenizedDirNonRecursiveModel)

tokenizedDirNonRecursiveDefault = tc.TokenizeDir(dirPath=dirPath, recursive=False)
print(tokenizedDirNonRecursiveDefault)


mappedTokenizedDirRecursiveModel = tc.TokenizeDir(dirPath=dirPath, model="gpt-4o", recursive=True, mapTokens=True)
print(mappedTokenizedDirRecursiveModel)

# Tokenizing a directory using the default model with token mapping
mappedTokenizedDirRecursiveDefaultModel = tc.TokenizeDir(dirPath=dirPath, recursive=True, mapTokens=True)
print(mappedTokenizedDirRecursiveDefaultModel)
```

---

#### `GetNumTokenDir(dirPath: Path | str, model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None, recursive: bool = True, quiet: bool = False, excludeBinary: bool = True, includeHidden: bool = False, mapTokens: bool = False) -> int | OrderedDict[str, int | OrderedDict]`

Counts the number of tokens in all files within a directory, or returns a nested `OrderedDict` structure with counts.

**Parameters:**

- `dirPath` (`Path | str`): The path to the directory to count tokens for.
- `model` (`str`, optional): The name of the model to use for encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): An existing `tiktoken.Encoding` object to use for tokenization.
- `recursive` (`bool`, optional): Whether to count tokens in subdirectories recursively. **Default: `True`**
- `quiet` (`bool`, optional): If `True`, suppresses progress updates. **Default: `False`**
- `excludeBinary` (`bool`, optional): Excludes any binary files by skipping over them. **Default: `True`**
- `includeHidden` (`bool`, optional): Skips over hidden files and directories, including subdirectories and files of a hidden directory. **Default: `False`**
- `mapTokens` (`bool`, optional): If `True`, outputs a nested `OrderedDict` structure mirroring the directory structure. For files, the value is the token count. For directories, the output is wrapped with `"tokens"` and `"numTokens"` keys, where `"tokens"` contains the nested counts. If `False`, returns the total token count as an integer. **Default: `False`**

**Returns:**

- `int`: The total number of tokens in the directory if `mapTokens` is `False`.
- `OrderedDict[str, int | OrderedDict]`: An `OrderedDict` mirroring the directory structure with token counts if `mapTokens` is `True`.

**Raises:**

- `TypeError`: If the types of input parameters are incorrect.
- `ValueError`: If the provided path is not a directory or if the model or encoding is invalid.
- `UnsupportedEncodingError`: If the file's encoding cannot be determined.
- `FileNotFoundError`: If the directory does not exist.

**Example:**

```python
import PyTokenCounter as tc
from pathlib import Path
from collections import OrderedDict

dirPath = Path("TestDir")
numTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=True)
print(numTokensDir)

mappedNumTokensDir = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=True, mapTokens=True)
print(mappedNumTokensDir)

# Counting tokens in a directory using the default model
mappedNumTokensDirDefault = tc.GetNumTokenDir(dirPath=dirPath, recursive=True, mapTokens=True)
print(mappedNumTokensDirDefault)

numTokensDirNonRecursiveModel = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=False)
print(numTokensDirNonRecursiveModel)

mappedNumTokensDirNonRecursiveModel = tc.GetNumTokenDir(dirPath=dirPath, model="gpt-4o", recursive=False, mapTokens=True)
print(mappedNumTokensDirNonRecursiveModel)

numTokensDirNonRecursiveDefault = tc.GetNumTokenDir(dirPath=dirPath, recursive=False)
print(numTokensDirNonRecursiveDefault)
```

---

### Token Mapping

#### `MapTokens(tokens: list[int] | OrderedDict[str, list[int] | OrderedDict], model: str | None = "gpt-4o", encodingName: str | None = None, encoding: tiktoken.Encoding | None = None) -> OrderedDict[str, str] | OrderedDict[str, OrderedDict[str, str] | OrderedDict]`

Maps tokens to their corresponding decoded strings based on a specified encoding.

**Parameters:**

- `tokens` (`list[int] | OrderedDict[str, list[int] | OrderedDict]`): The tokens to be mapped. This can either be:
    - A list of integer tokens to decode.
    - An `OrderedDict` with string keys and values that are either:
        - A list of integer tokens.
        - Another nested `OrderedDict` with the same structure.
- `model` (`str`, optional): The model name to use for determining the encoding. **Default: `"gpt-4o"`**
- `encodingName` (`str`, optional): The name of the encoding to use.
- `encoding` (`tiktoken.Encoding`, optional): The encoding object to use.

**Returns:**

- `OrderedDict[str, str] | OrderedDict[str, OrderedDict[str, str] | OrderedDict]`: A mapping of decoded strings to their corresponding integer tokens. If `tokens` is a nested structure, the result will maintain the same nested structure with decoded mappings.

**Raises:**

- `TypeError`: If `tokens` is not a list of integers or an `OrderedDict` of strings mapped to tokens.
- `ValueError`: If an invalid model or encoding name is provided, or if the encoding does not match the model or encoding name.
- `KeyError`: If a token is not in the given encoding's vocabulary.
- `RuntimeError`: If an unexpected error occurs while validating the encoding.

**Example:**

```python
import PyTokenCounter as tc
import tiktoken
from collections import OrderedDict

encoding = tiktoken.get_encoding("cl100k_base")
tokens = [123,456,789]
mapped = tc.MapTokens(tokens=tokens, encoding=encoding)
print(mapped)

mapped = tc.MapTokens(tokens=tokens, encoding=encoding)
print(mapped)

# Mapping tokens using the default model
mapped = tc.MapTokens(tokens=tokens)
print(mapped)
```

---

## Ignored Files

When the functions are set to exclude binary files (default behavior), the following file extensions are ignored:

| Category                        | Extensions                                                                                                                                                      |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Image formats**               | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.pbm`, `.webp`, `.avif`, `.tiff`, `.tif`, `.ico`, `.svgz`                                                                    |
| **Video formats**               | `.mp4`, `.mkv`, `.mov`, `.avi`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.mpeg`, `.mpg`, `.3gp`, `.3g2`                                                               |
| **Audio formats**               | `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.wma`, `.aiff`, `.ape`, `.opus`                                                                               |
| **Compressed archives**         | `.zip`, `.rar`, `.7z`, `.tar`, `.gz`, `.bz2`, `.xz`, `.lz`, `.zst`, `.cab`, `.deb`, `.rpm`, `.pkg`                                                               |
| **Disk images**                 | `.iso`, `.dmg`, `.img`, `.vhd`, `.vmdk`                                                                                                                           |
| **Executables & Libraries**     | `.exe`, `.msi`, `.bat`, `.dll`, `.so`, `.bin`, `.o`, `.a`, `.dylib`                                                                                             |
| **Fonts**                       | `.ttf`, `.otf`, `.woff`, `.woff2`, `.eot`                                                                                                                         |
| **Documents**                   | `.pdf`, `.ps`, `.eps`                                                                                                                                             |
| **Design & Graphics**           | `.psd`, `.ai`, `.indd`, `.sketch`                                                                                                                                  |
| **3D & CAD files**              | `.blend`, `.stl`, `.step`, `.iges`, `.fbx`, `.glb`, `.gltf`, `.3ds`, `.obj`, `.cad`                                                                             |
| **Virtual Machines & Firmware** | `.qcow2`, `.vdi`, `.vhdx`, `.rom`, `.bin`, `.img`                                                                                                               |
| **Miscellaneous binaries**      | `.dat`, `.pak`, `.sav`, `.nes`, `.gba`, `.nds`, `.iso`, `.jar`, `.class`, `.wasm`                                                                               |

Along with ignoring the extensions in the exclude list to quickly bypass known files that cannot be read, the code also catches decoding errors and skips files when `excludeBinary` is `True`. This approach ensures all unreadable files are handled efficiently, combining fast extension-based checks with robust decoding error handling.

---

## Maintainers

- [Kaden Gruizenga](https://github.com/kgruiz)

## Acknowledgements

- This project is based on the `tiktoken` library created by [OpenAI](https://github.com/openai/tiktoken).

## Contributing

Contributions are welcome! Feel free to [open an issue](https://github.com/kgruiz/PyTokenCounter/issues/new) or submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.