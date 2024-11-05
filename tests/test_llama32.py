import pytest
from tokenizers import Tokenizer
from autotiktokenizer import AutoTikTokenizer

@pytest.fixture
def tiktokenizer():
    encoder = AutoTikTokenizer.from_pretrained('unsloth/Llama-3.2-3B-Instruct')
    return encoder

@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained('unsloth/Llama-3.2-3B-Instruct')

def test_simple_sentence(tiktokenizer, tokenizer):
    sentence = "Hey, I am Bhavnick Singh Minhas and I am building a tool to use TikToken tokenizers."
    ttk_enc = tiktokenizer.encode(sentence)
    hf_enc = tokenizer.encode(sentence, add_special_tokens=False).ids

    assert ttk_enc == hf_enc, f"{ttk_enc} != {hf_enc}"
    assert tokenizer.decode(hf_enc) == sentence, f"{tokenizer.decode(hf_enc)} != {sentence}"
    assert tiktokenizer.decode(ttk_enc) == sentence, f"{tiktokenizer.decode(ttk_enc)} != {sentence}"
    assert tokenizer.decode(hf_enc) == tiktokenizer.decode(ttk_enc), f"{tokenizer.decode(hf_enc)} != {tiktokenizer.decode(ttk_enc)}"
    assert tiktokenizer.decode(hf_enc) == tokenizer.decode(ttk_enc), f"{tiktokenizer.decode(hf_enc)} != {tokenizer.decode(ttk_enc)}"