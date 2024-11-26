from typing import Dict
import os 
import json

import tiktoken
from huggingface_hub import snapshot_download

class AutoTikTokenizer:
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
        # Initialize BYTES encoder and decoder for unicode conversion
        self.bytes_encoder = self._bytes_to_unicode()
        self.bytes_decoder = {v:k for k,v in self.bytes_encoder.items()}
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.

        Returns:
            dict: A dictionary of utf-8 bytes and corresponding unicode strings.
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

    def _download_from_hf_hub(self,
                              repo_name: str) -> str:
        """
        Downloads the necessary files from the HuggingFace Hub for the tokenizer.

        Args:
            repo_name (str): The name of the repository to download the files from.
        
        Returns:
            path (str): The path to the downloaded files.
        """ 

        # Download all the necessary files from HF Hub
        files_needed = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
        path = snapshot_download(repo_id=repo_name, allow_patterns=files_needed)
        files_downloaded = os.listdir(path)

        # Assertions to make sure the necessary files are there
        assert 'config.json' in files_downloaded, \
                "config.json not found in downloaded files"
        assert 'tokenizer.json' in files_downloaded or 'vocab.json' in files_downloaded,\
                "tokenizer.json not found in downloaded files"
        assert 'tokenizer_config.json' in files_downloaded, \
                "tokenizer_config.json not found in downloaded files"

        return path
    
    def _normalize_token_bytes(self, token: str) -> bytes:
        """Convert bytes to unicode.
        
        Args:
            token (str): The token to convert.
        
        Returns:
            result (bytes): The converted token.
        """
        try:
          result = bytearray([self.bytes_decoder[b] for b in token])
        except Exception:
          result = token.encode()
        result = bytes(result)
        return result


    def _get_mergeable_ranks(self,
                             vocab: Dict[str, int],
                             special_tokens: Dict[str, int],
                             tokenizer_type: str) -> Dict[str, int]:
        """Convert vocab to binary mergeable_ranks.
        
        Args:
            vocab (dict): The vocabulary of the tokenizer.
            special_tokens (dict): The special tokens used by the tokenizer.
        
        Returns:
            mergeable_ranks (dict): The mergeable ranks of tokens in binary format.
        """
        mergeable_ranks = {}
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for rank, (token, _) in enumerate(sorted_vocab, start=0):
            # Converting wordpiece to equivalent std BPE form
            if tokenizer_type == 'wordpiece':
                if token.startswith('##'):
                    token = token[2:]
                else:
                    token = 'Ġ' + token
            elif tokenizer_type == 'bpe':
              # For uniformity, will convert any sentencepiece like beginnings
              # into standard HF Ġ format
              if token.startswith('▁'):
                  token = token.replace('▁', 'Ġ')

            if token not in special_tokens:
                key = self._normalize_token_bytes(token)
            else:
                key = token.encode()
            mergeable_ranks[key] = rank
        return mergeable_ranks

    def _get_vocab(self, tokenizer: Dict, path: str) -> Dict[str, int]:
        """
        Returns the vocabulary of the tokenizer.

        Args:
            tokenizer (dict): The tokenizer dictionary.
            path (str): The path to the tokenizer files.
        
        Returns:
            vocab (dict): The vocabulary of the tokenizer.
        """
        try:
          vocab = tokenizer['model']['vocab']
        except KeyError:
          raise Warning("No vocab found inside tokenizer.json" + \
                        "Trying to load vocab from vocab.json")
          try:
            vocab = self._read_json(os.path.join(path,'vocab.json'))
          except Exception:
            raise Exception("Vocab not found in tokenizer.json or vocab.json")

        return vocab

    def _get_special_tokens(self, tokenizer: Dict) -> Dict[str, int]:
        """
        Returns the special tokens used by the tokenizer.

        Args:
            tokenizer (dict): The tokenizer dictionary.
            
        Returns:
            special_tokens (dict): The special tokens used by the tokenizer.
        """

        try:
          special_tokens = {at['content'] : at['id'] for at in tokenizer['added_tokens']}
        except KeyError:
          raise Warning("No special tokens found inside tokenizer.json")
          special_tokens = {}
        return special_tokens

    def _get_pattern_str(self, tokenizer_type: str) -> str:
        """
        Returns the regex pattern used for tokenization.
        Args:
            tokenizer_config (dict): The configuration of the tokenizer.
        Returns:
            str: The regex pattern used for tokenization.
        """
        if tokenizer_type == 'wordpiece':
            pattern = r'[!-/:-@\[-`{-~]|[^\s!-/:-@\[-`{-~]+'
        else:
            pattern = r'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s'
        return pattern
    
    def _read_json(self, path: str) -> Dict:
        """
        Reads a JSON file and returns the contents.
        
        Args:
            path (str): The path to the JSON file.
            
        Returns:
            contents (dict): The contents of the JSON file."""
        with open(path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def _get_tiktoken_encoding(self,
                               name: str,
                               pattern: str,
                               mergeable_ranks: Dict[str, int],
                               special_tokens: Dict[str, int]) -> tiktoken.Encoding:
        """
        Constructs and returns a TikToken encoding using the tokenizer's attributes.
        
        Args:
            name (str): The name of the tokenizer.
            pattern (str): The regex pattern used for tokenization.
            mergeable_ranks (dict): The mergeable ranks of tokens in binary format.
            special_tokens (dict): The special tokens used by the tokenizer.
        
        Returns:
            encoding (Encoding): The TikToken encoding.
        """

        encoding = tiktoken.Encoding(
            name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return encoding
    
    def _detect_tokenizer_type(self, tokenizer: Dict,  tokenizer_config: Dict) -> str:
        """
        Detects whether the tokenizer is WordPiece or BPE based.

        Args:
            tokenizer (Dict): The tokenizer dictionary.
            tokenizer_config (Dict): The tokenizer configuration dictionary.

        Returns:
            str: The tokenizer type ('wordpiece' or 'bpe').
        """
        if tokenizer_config.get('tokenizer_class', '').lower() in ['berttokenizer', 'wordpiece']:
            return 'wordpiece'
        if tokenizer.get('model', '').get('type', '').lower() == 'wordpiece':
            return 'wordpiece'
        return 'bpe'

    @classmethod
    def from_pretrained(cls, tokenizer_name_or_path: str) -> tiktoken.Encoding:
        """
        Loads a pretrained tokenizer from the specified path or name and returns the TikToken encoding.

        Args:
            tokenizer_name_or_path (str): The name or path of the pretrained tokenizer.

        Returns:
            encoding (Encoding): The TikToken encoding.
        """
        #init instance
        instance = cls()

        # Load from a local directory
        path = tokenizer_name_or_path
        if os.path.isfile(os.path.join(tokenizer_name_or_path, 'tokenizer.json')) :
            tokenizer = instance._read_json(os.path.join(tokenizer_name_or_path, 'tokenizer.json'))
        else:
            try : 
                path = instance._download_from_hf_hub(tokenizer_name_or_path)
                tokenizer = instance._read_json(os.path.join(path, 'tokenizer.json'))
            except Exception as e:
                print("Tokenizer could not be loaded from a local directory nor from the hub")
                raise e

        # Load the vocab from the tokenizer
        vocab = instance._get_vocab(tokenizer, path)

        # Load the added_tokens from the tokenizer.json
        special_tokens = instance._get_special_tokens(tokenizer)

        # Load the tokenizer config
        tokenizer_config = instance._read_json(os.path.join(path,'tokenizer_config.json'))

        # Detect the tokenizer type
        tokenizer_type = instance._detect_tokenizer_type(tokenizer, tokenizer_config)

        #Load the name
        name = tokenizer_name_or_path.split('/')[-1]

        # Get the mergeable ranks from the vocab and special_tokens
        mergeable_ranks = instance._get_mergeable_ranks(vocab, special_tokens, tokenizer_type)

        # Get the pattern string
        pattern = instance._get_pattern_str(tokenizer_type)

        return instance._get_tiktoken_encoding(name, pattern, mergeable_ranks, special_tokens)

    def __call__(self, tokenizer_name_or_path: str) -> tiktoken.Encoding:
      return self.from_pretrained(tokenizer_name_or_path)
    
    def __repr__(self) -> str:
        return "AutoTikTokenizer"
