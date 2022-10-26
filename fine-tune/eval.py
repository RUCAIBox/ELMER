import pickle
import os
import collections
import sys
import numpy as np

sys.path.append('./pyeval')
from pyeval.bleu.bleu import Bleu
from pyeval.rouge.rouge import Rouge
from pyeval.meteor.meteor import Meteor
from pyeval.distinct.distinct import Distinct


class Evaluate(object):
    def __init__(self):
        self.metrics = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
            # (Distinct(4), ["Distinct_1", "Distinct_2", "Distinct_3", "Distinct_4"])
        ]

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.metrics:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, candidate_list, reference_list):
        # make dictionary
        hypo = {}
        ref = {}
        for i in range(len(candidate_list)):
            hypo[i] = [candidate_list[i]]
            ref[i] = [reference_list[i]]

        # compute scores
        final_scores = self.score(ref, hypo)

        return final_scores

