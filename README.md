<div align="center">
  
![AutoTikTokenizer Logo](./assets/AutoTikTokenizer%20Logo.png)

# AutoTikTokenizer

[![PyPI version](https://img.shields.io/pypi/v/autotiktokenizer.svg)](https://pypi.org/project/autotiktokenizer/)
[![Downloads](https://static.pepy.tech/badge/autotiktokenizer)](https://pepy.tech/project/autotiktokenizer)
![Package size](https://img.shields.io/badge/size-9.7MB-blue)
[![License](https://img.shields.io/github/license/bhavnicksm/autotiktokenizer)](https://github.com/bhavnicksm/autotiktokenizer/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/bhavnicksm/autotiktokenizer#readme)
[![Last Commit](https://img.shields.io/github/last-commit/bhavnicksm/autotiktokenizer)](https://github.com/bhavnicksm/autotiktokenizer/commits/main)
[![GitHub Stars](https://img.shields.io/github/stars/bhavnicksm/autotiktokenizer?style=social)](https://github.com/bhavnicksm/autotiktokenizer/stargazers)

🚀 Accelerate your HuggingFace tokenizers by converting them to TikToken format with AutoTikTokenizer - get TikToken's speed while keeping HuggingFace's flexibility.

[Features](#key-features) •
[Installation](#installation) •
[Examples](#examples) •
[Supported Models](#supported-models) •
[Benchmarks](#benchmarks) •
[Sharp Bits](#sharp-bits) •
[Citation](#citation)

</div>

# Key Features

- 🚀 **High Performance** - Built on TikToken's efficient tokenization engine
- 🔄 **HuggingFace Compatible** - Seamless integration with the HuggingFace ecosystem
- 📦 **Lightweight** - Minimal dependencies, just TikToken and Huggingface-hub
- 🎯 **Easy to Use** - Simple, intuitive API that works out of the box
- 💻 **Well Tested** - Comprehensive test suite across supported models

# Installation

Install `autotiktokenizer` from PyPI via the following command:

```bash
pip install autotiktokenizer
```

You can also install it from _source_, by the following command:

```bash
pip install git+https://github.com/bhavnicksm/autotiktokenizer
```

# Examples

This section provides a basic usage example of the project. Follow these simple steps to get started quickly.

```python
# step 1: Import the library
from autotiktokenizer import AutoTikTokenizer

# step 2: Load the tokenizer
tokenizer = AutoTikTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# step 3: Enjoy the Inferenece speed 🏎️
text = "Wow! I never thought I'd be able to use Llama on TikToken"
encodings = tokenizer.encode(text)

# (Optional) step 4: Decode the outputs
text = tokenizer.decode(encodings)
```

# Supported Models

AutoTikTokenizer should ideally support ALL models on HF Hub but because of the vast diversity of models out there, we _cannot_ test out every single model. These are the models we have already validated for, and know that AutoTikTokenizer works well for them. If you have a model you wish to see here, raise an issue and we would validate and add it to the list. Thanks :)

- [x] GPT2
- [x] GPT-J Family
- [x] SmolLM Family: Smollm2-135M, Smollm2-350M, Smollm2-1.5B etc.
- [x] LLaMa 3 Family: LLama-3.2-1B-Instruct, LLama-3.2-3B-Instruct, LLama-3.1-8B-Instruct etc.
- [x] Deepseek Family: Deepseek-v2.5 etc 
- [x] Gemma2 Family: Gemma2-2b-It, Gemma2-9b-it etc
- [x] Mistral Family: Mistral-7B-Instruct-v0.3 etc
- [x] Aya Family: Aya-23B, Aya Expanse etc
- [x] BERT Family: BERT, RoBERTa, MiniLM, TinyBERT, DeBERTa etc.

**NOTE:** Some models use the _unigram_ tokenizers, which are not supported with TikToken and hence, 🧰 AutoTikTokenizer cannot convert the tokenizers for such models. Some models that use _unigram_ tokenizers include T5, ALBERT, Marian and XLNet. 

# Benchmarks

Benchmarking results for tokenizing **1 billion tokens** from fineweb-edu dataset using **Llama 3.2 tokenizer** on CPU (Google colab)

| Configuration | Processing Type | AutoTikTokenizer | HuggingFace | Speed Ratio | 
|--------------|-----------------|------------------|--------------|-------------|
| Single Thread | Sequential | **14:58** (898s) | 40:43 (2443s) | 2.72x faster |
| Batch x1 | Batched | 15:58 (958s) | **10:30** (630s) | 0.66x slower |
| Batch x4 | Batched | **8:00** (480s) | 10:30 (630s) | 1.31x faster |
| Batch x8 | Batched | **6:32** (392s) | 10:30 (630s) | 1.62x faster |
| 4 Processes | Parallel | **2:34** (154s) | 8:59 (539s) | 3.50x faster |

The above table shows that AutoTikTokenizer's tokenizer (TikToken) is actually way faster than HuggingFace's Tokenizer by 1.6-3.5 times under fair comparison! While, it's not making the most optimal use of TikToken (yet), its still way faster than the stock solutions you might be getting otherwise.

# Sharp Bits

A known issue of the repository is that it does not do any pre-processing or post-processing, which means that if a certain tokenizer (like `minilm`) expect all lower-case letters only, then you would need to convert it to lower case manually. Similarly, any spaces added in the process are not removed during decoding, so they need to handle them on your own. 

There might be more sharp bits to the repository which are unknown at the moment, please raise an issue if you encounter any!

# Acknowledgement

Special thanks to HuggingFace and OpenAI for making their respective open-source libraries that make this work possible. I hope that they would continue to support the developer ecosystem for LLMs in the future!

**If you found this repository useful, give it a ⭐️! Thank You :)**

# Citation

If you use `autotiktokenizer` in your research, please cite it as follows:

```bibtex
@misc{autotiktokenizer,
    author = {Bhavnick Minhas},
    title = {AutoTikTokenizer},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/bhavnicksm/autotiktokenizer}},
}
```
