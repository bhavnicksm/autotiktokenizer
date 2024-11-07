<div align="center">
  
![AutoTikTokenizer Logo](./assets/AutoTikTokenizer%20Logo.png)

# AutoTikTokenizer

[![PyPI version](https://img.shields.io/pypi/v/autotiktokenizer.svg)](https://pypi.org/project/autotiktokenizer/)
[![Downloads](https://static.pepy.tech/badge/autotiktokenizer)](https://pepy.tech/project/autotiktokenizer)
[![License](https://img.shields.io/github/license/bhavnicksm/autotiktokenizer)](https://github.com/bhavnicksm/autotiktokenizer/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/bhavnicksm/autotiktokenizer#readme)
[![Last Commit](https://img.shields.io/github/last-commit/bhavnicksm/autotiktokenizer)](https://github.com/bhavnicksm/autotiktokenizer/commits/main)
[![GitHub Stars](https://img.shields.io/github/stars/bhavnicksm/autotiktokenizer?style=social)](https://github.com/bhavnicksm/autotiktokenizer/stargazers)

A great way to leverage the speed and lightweight of OpenAI's TikToken with the universal support of HuggingFace's Tokenizers. Now, you can run ANY tokenizer at 3-6x the speed out of the box!

[Installation](#installation) •
[Examples](#examples) •
[Supported Models](#supported-models) •
[Citation](#citation) 

</div>

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

AutoTikTokenizer current supports the following models (and their variants) out of the box, with support for other models to be tested and added soon!

- [x] GPT2
- [x] GPT-J Family
- [x] SmolLM Family: Smollm2-135M, Smollm2-350M, Smollm2-1.5B etc.
- [x] LLaMa 3 Family: LLama-3.2-1B-Instruct, LLama-3.2-3B-Instruct, LLama-3.1-8B-Instruct etc.
- [x] Deepseek Family: Deepseek-v2.5 etc 
- [x] Gemma2 Family: Gemma2-2b-It, Gemma2-9b-it etc
- [x] Mistral Family: Mistral-7B-Instruct-v0.3 etc
- [ ] BERT Family: BERT, RoBERTa, MiniLM, TinyBERT, DeBERTa etc.

**NOTE:** Some models use the _unigram_ tokenizers, which are not supported with TikToken and hence, 🧰 AutoTikTokenizer cannot convert the tokenizers for such models. Some models that use _unigram_ tokenizers include T5, ALBERT, Marian and XLNet. 

# Acknowledgement

Special thanks to HuggingFace and OpenAI for making their respective open-source libraries that make this work possible. I hope that they would continue to support the developer ecosystem for LLMs in the future! 

**If you found this repository useful, give it a ⭐️! Thank You :)**

# Citation

If you use `autotiktokenizer` in your research, please cite it as follows:

```
@misc{autotiktokenizer,
    author = {Bhavnick Minhas},
    title = {AutoTikTokenizer},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/bhavnicksm/autotiktokenizer}},
}
```
