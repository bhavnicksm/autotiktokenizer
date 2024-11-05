![AutoTikTokenizer Logo](./assets/AutoTikTokenizer%20Logo.png)
# AutoTikTokenizer

A great way to leverage the speed and lightweight of OpenAI's TikToken with the universal support of HuggingFace's Tokenizers. Now, you can run ANY tokenizer at 3-6x the speed out of the box!

# Quick Install and Use

Install `autotiktokenizer` from PyPI via the following command:

```bash
pip install autotiktokenizer
```

And just run it in a couple of easy steps,

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
- [x] LLaMa 3 Family: LLama-3.2-1B-Instruct, LLama-3.2-3B-Instruct, LLama-3.1-8B-Instruct etc.
- [x] SmolLM Family: Smollm2-135M, Smollm2-350M, Smollm2-1.5B etc.
- [x] GPT-J Family
- [x] Gemma2 Family: Gemma2-2b-It, Gemma2-9b-it etc
- [x] Deepseek Family: Deepseek-v2.5 etc 
- [ ] Mistral Family: Mistral-7B-Instruct-v0.3    


# Acknoledgement

Special thanks to HuggingFace and OpenAI for making their respective open-source libraries that make this work possible. I hope that they would continue to support the developer ecosystem for LLMs in the future! 

**If you found this repository useful, I would appriciate if you could star this repository and boost it on Socials so a greater audience could benefit from this. Thank you so much! :)**

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
