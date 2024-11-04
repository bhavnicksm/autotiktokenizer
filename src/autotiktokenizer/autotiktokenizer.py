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
    def __init__(self):
        self.tokenizer = None
        self.name = None
        self.vocab = None
        self.tokenizer_config = None
        self.mergeable_ranks = None
        self.special_tokens = None
        self.pattern = None

    def get_mergable_ranks(self):
        # Convert vocab to binary mergeable_ranks
        self.mergeable_ranks = {}
        
        # Sort vocab by token id to ensure correct ordering
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        
        # Create binary format ranks starting from 1
        for rank, (token, _) in enumerate(sorted_vocab, start=0):
            # Handle GPT-2 style tokens
            if token.startswith('Ġ'):
                token = ' ' + token[1:]
            self.mergeable_ranks[token.encode('utf-8')] = rank
        
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

    def get_tiktoken_encoding(self):
        special_tokens = self.get_special_tokens()
        mergeable_ranks = self.get_mergable_ranks()
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
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer_config = dict(json.loads(self.tokenizer.to_str()))
        self.name = self.tokenizer_name_or_path.split('/')[-1]
        return self.get_tiktoken_encoding()
        
    def __call__(self):
        return self.get_tiktoken_encoding()


AutoTikTokenizer = _AutoTikTokenizer()