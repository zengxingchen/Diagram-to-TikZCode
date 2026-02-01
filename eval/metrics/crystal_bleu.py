# eval/crystalbleu.py
# pip install crystalbleu sacremoses pygments

import os
import pickle
from collections import Counter
from hashlib import md5
from itertools import chain, tee
from typing import List
from tqdm import tqdm

# Local compatibility shim limited to the crystalbleu module.
import fractions as _fractions
import crystalbleu as _cr  # import as a module first

# Older crystalbleu versions do `from fractions import Fraction` and call
# Fraction(_normalize=False); Python's stdlib Fraction now rejects that
# keyword. Patch _cr.Fraction (NOT global fractions.Fraction) to ignore
# the extra kwargs.
try:
    _ = _cr.Fraction(1, 1, _normalize=False)
except TypeError:
    def _fraction_compat(numerator=0, denominator=None, **kwargs):
        # Drop the _normalize / other unsupported kwargs.
        if denominator is None:
            return _fractions.Fraction(numerator)
        return _fractions.Fraction(numerator, denominator)
    _cr.Fraction = _fraction_compat

# Continue using helpers from the crystalbleu module below.
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer


def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


class CrystalBLEU:
    def __init__(self, corpus: List[str], k: int = 500, n: int = 4, use_cache: bool = True, cache_path: str = "./crystalbleu_cache"):
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.corpus = corpus
        self.k = k
        self.n = n
        self.use_cache = use_cache
        self.list_of_references = []
        self.hypotheses = []
        self._cache_path = cache_path
        os.makedirs(self._cache_path, exist_ok=True)

    def _tokenize(self, text):
        tokens = []
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and tokentype is not Comment:
                if tokentype in (Text, Name.Attribute, Name.Builtin):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())
        return tokens

    def _get_trivial_ngrams(self):
        dhash = md5()
        dhash.update(str(sorted(self.corpus)).encode())
        cache_file = os.path.join(self._cache_path, f"{dhash.hexdigest()}.pkl")

        if os.path.isfile(cache_file) and self.use_cache:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            all_ngrams = []
            for o in range(1, self.n + 1):
                for tex in self.corpus:
                    all_ngrams.extend(ngrams(self._tokenize(tex), o))
            freq = Counter(all_ngrams)
            most_common = dict(freq.most_common(self.k))
            if self.use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(most_common, f)
            return most_common

    def update(self, list_of_references: List[List[str]], hypotheses: List[str]):
        assert len(list_of_references) == len(hypotheses)
        for refs, hyp in tqdm(zip(list_of_references, hypotheses), total=len(hypotheses), desc="Updating CrystalBLEU"):
            self.list_of_references.append([self._tokenize(r) for r in refs])
            self.hypotheses.append(self._tokenize(hyp))

    def compute(self):
        ignoring = self._get_trivial_ngrams()
        print("computing corpus_bleu.....")
        # Use crystalbleu.corpus_bleu directly; its Fraction has been patched above.
        return _cr.corpus_bleu(
            list_of_references=self.list_of_references,
            hypotheses=self.hypotheses,
            ignoring=ignoring
        )