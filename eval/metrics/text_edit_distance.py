# eval/eed.py

from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Text
from tqdm import tqdm
from torchmetrics.text import ExtendedEditDistance
from torchmetrics.functional.text.eed import (
    _compute_sentence_statistics,
    _preprocess_en,
    _preprocess_ja,
)
import multiprocessing as mp

def _compute_single_eed(args):
    hyp, refs, alpha, rho, deletion, insertion = args
    return _compute_sentence_statistics(hyp, refs, alpha, rho, deletion, insertion)


class TexEditDistance(ExtendedEditDistance):
    """Adapt torchmetrics ExtendedEditDistance for TeX"""

    def __init__(self, *args, language="en", **kwargs):
        super().__init__(*args, **kwargs)
        self.lexer = TexLexer()
        self.language = language

    def __str__(self):
        return self.__class__.__name__

    def _preprocess_sentences(self, preds, target, language):
        def tokenize(text):
            tokens = []
            for tokentype, value in self.lexer.get_tokens(text):
                if value.strip():
                    if tokentype is Text:
                        preprocess_fn = _preprocess_en if language == "en" else _preprocess_ja
                        tokens.extend(preprocess_fn(value).split())
                    elif tokentype is not Comment:
                        tokens.extend(value.split())
            return " " + " ".join(tokens) + " "

        preds = [tokenize(p) for p in preds]
        target = [[tokenize(t)] for t in target]  # list of list
        return preds, target

    def update(self, preds, target):
        preds, target = self._preprocess_sentences(preds, target, self.language)

        if self.sentence_eed is None:
            self.sentence_eed = []

        if 0 in (len(preds), len(target[0])):
            return self.sentence_eed

        # Prepare input for multiprocessing
        args_list = [
            (hyp, refs,
             self.alpha, self.rho,
             self.deletion, self.insertion)
            for hyp, refs in zip(preds, target)
        ]

        with mp.Pool(processes=32) as pool:
            # Wrap pool.imap with tqdm for a progress bar.
            results = list(tqdm(pool.imap(_compute_single_eed, args_list), 
                                total=len(args_list), 
                                desc="Computing EED"))

        self.sentence_eed.extend(results)
        return self.sentence_eed

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs).item()