"""
Currently, this test is for the Mistral-7B-Instruct-v0.3 model. And, 
this test currently does not pass, because of thread errors coming through.
Later versions of this repository will support this as well. 
"""
import pytest
from tokenizers import Tokenizer
from autotiktokenizer import AutoTikTokenizer

@pytest.fixture
def tiktokenizer():
    encoder = AutoTikTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
    return encoder

@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')

def test_simple_sentence(tiktokenizer, tokenizer):
    sentence = "Hey, I am Bhavnick Singh Minhas and I am building a tool to use TikToken tokenizers."
    ttk_enc = tiktokenizer.encode(sentence)
    hf_enc = tokenizer.encode(sentence).ids

    assert ttk_enc == hf_enc, f"{ttk_enc} != {hf_enc}"
    assert tokenizer.decode(hf_enc) == sentence, f"{tokenizer.decode(hf_enc)} != {sentence}"
    assert tiktokenizer.decode(ttk_enc) == sentence, f"{tiktokenizer.decode(ttk_enc)} != {sentence}"
    assert tokenizer.decode(hf_enc) == tiktokenizer.decode(ttk_enc), f"{tokenizer.decode(hf_enc)} != {tiktokenizer.decode(ttk_enc)}"
    assert tiktokenizer.decode(hf_enc) == tokenizer.decode(ttk_enc), f"{tiktokenizer.decode(hf_enc)} != {tokenizer.decode(ttk_enc)}"