import numpy as np
from collections import defaultdict, Counter

class Distinct:
    def __init__(self, n=4):
        self.n_grams = [1, 2, 3, 4]

    def method(self):
        return "Distinct"

    def dist_func(self, generate_sentence, ngram):
        ngram_dict = defaultdict(int)
        tokens = generate_sentence[:ngram]
        ngram_dict[" ".join(tokens)] += 1
        for i in range(1, len(generate_sentence) - ngram + 1):
            tokens = tokens[1:]
            tokens.append(generate_sentence[i + ngram - 1])
            ngram_dict[" ".join(tokens)] += 1
        return ngram_dict

    def compute_score(self, gts, res):
        generate_corpus = []
        for key, value in res.items():
            generate_corpus.append(value[0].split())
        scores = []
        for n_gram in self.n_grams:
            ngrams_all = Counter()
            intra_ngram = []
            for generate_sentence in generate_corpus:
                result = self.dist_func(generate_sentence=generate_sentence, ngram=n_gram)
                intra_ngram.append(len(result) / sum(result.values()))
                ngrams_all.update(result)
            scores.append(len(ngrams_all) / sum(ngrams_all.values()) * 100)
        return scores