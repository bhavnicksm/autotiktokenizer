from tokenizers import Tokenizer
import json
import tiktoken


class _AutoTikTokenizer:
    """
    _AutoTikTokenizer is a class designed to interface with HuggingFace tokenizers to provide a TikToken tokenizer
    that can be used for the tokenization process. It mimics the functionality of AutoTokenizer in HuggingFace
    but is tailored for TikToken.
    Attributes:
        tokenizer (Tokenizer): The HuggingFace tokenizer instance.
        name (str): The name of the tokenizer.
        vocab (dict): The vocabulary of the tokenizer.
        tokenizer_config (dict): The configuration of the tokenizer.
        mergeable_ranks (dict): The mergeable ranks of tokens in binary format.
        special_tokens (dict): The special tokens used by the tokenizer.
        pattern (str): The regex pattern used for tokenization.
    Methods:
        __init__():
            Initializes the _AutoTikTokenizer with default values.
        get_mergable_ranks():
            Converts the vocabulary to binary mergeable ranks and returns it.
        get_special_tokens():
            Retrieves and returns the special tokens used by the tokenizer.
        get_pattern_str():
            Returns the regex pattern used for tokenization.
        get_tiktoken_encoding():
            Constructs and returns a TikToken encoding using the tokenizer's attributes.
        from_pretrained(tokenizer_name_or_path: str):
            Loads a pretrained tokenizer from the specified path or name and returns the TikToken encoding.
        __call__():
            Returns the TikToken encoding.
    """
    def __init__(self) -> None:
        self.bytes_encoder = self._bytes_to_unicode()
        self.bytes_decoder = {v:k for k,v in self.bytes_encoder.items()}
    
    def _bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _normalize_token_bytes(self, token):
        """Convert bytes to unicode."""
        result = bytearray([self.bytes_decoder[b] for b in token])
        result = bytes(result)
        return result

    def get_mergable_ranks(self, vocab, special_tokens):
        """Convert vocab to binary mergeable_ranks."""
        self.mergeable_ranks = {}
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for rank, (token, _) in enumerate(sorted_vocab, start=0):
            if token not in special_tokens:
                key = self._normalize_token_bytes(token)
            else:
                key = token.encode()
            self.mergeable_ranks[key] = rank
        return self.mergeable_ranks

    def get_special_tokens(self):
        self.special_tokens = {}
        sp = self.tokenizer.get_added_tokens_decoder()
        for idx, token in sp.items():
            self.special_tokens[token.content] = idx
        return self.special_tokens

    def get_pattern_str(self):
        self.pattern = r'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s'
        return self.pattern

    def get_tiktoken_encoding(self, vocab):
        special_tokens = self.get_special_tokens()
        mergeable_ranks = self.get_mergable_ranks(vocab, special_tokens)
        pattern = self.get_pattern_str()

        encoding = tiktoken.Encoding(
            self.name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        return encoding

    def from_pretrained(self, tokenizer_name_or_path: str):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)
        vocab = self.tokenizer.get_vocab()

        self.tokenizer_config = dict(json.loads(self.tokenizer.to_str()))
        self.name = self.tokenizer_name_or_path.split('/')[-1]
        return self.get_tiktoken_encoding(vocab)

    def __call__(self, tokenizer_name_or_path: str):
      return self.from_pretrained(tokenizer_name_or_path)
    
    def __repr__(self) -> str:
        return "AutoTikTokenizer"

AutoTikTokenizer = _AutoTikTokenizer()