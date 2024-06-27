# import re

# from os import PathLike
from typing import Dict, List
# from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.text.base_tokenizer import BaseTokenizer


class RouterTokenizer(BaseTokenizer):

    def __init__(self) -> None:
        pass

    def text2tokens(self, line: str) -> List[str]:
        return [int(line)]

    def tokens2text(self, tokens: List[str]) -> str:
        return str(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = []

        for ch in tokens:
            ids.append(int(ch))
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        content = [self.char_dict[w] for w in ids]
        return content

    def vocab_size(self) -> int:
        return len(self.char_dict)

    @property
    def symbol_table(self) -> Dict[str, int]:
        return self._symbol_table
