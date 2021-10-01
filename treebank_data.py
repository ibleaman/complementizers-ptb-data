import nltk
from nltk.corpus import ptb
import re

import pickle

import algorithms


detokenizer = nltk.treebank.TreebankWordDetokenizer()

successes = []
false_positives = []
false_negatives = []


def sent_without_symbols(words):
    sent_lst = [w for w in words if w != '0' and '*' not in w]
    return detokenizer.detokenize(sent_lst)


def contains_overt_comp(parsed):
    return bool(re.search(r'\(SBAR\s+\(IN that\)', str(parsed)))


def contains_null_comp(parsed):
    return bool(re.search(r'\(SBAR\s+\(\-NONE\- 0\)(?!\s*\(S\s+\(\-NONE\-\s*\*T\*\-\d\)\))',
                          str(parsed)))


def run_tests(testing_data):
    count = 0
    for sent in testing_data:
        count += 1
        if count > 2000:
            return
        outcome = algorithms.contains_variable(sent['sent_string'], 0.075, 0.3)
        print(outcome)
        if outcome[0] == sent['contains_overt_comp'] or outcome[0] == sent['contains_null_comp']:
            successes.append(outcome)
        elif outcome[0]:
            false_positives.append(outcome)
        else:
            false_negatives.append(outcome)


def generate_treebank_data():
    zipped = zip(nltk.corpus.ptb.parsed_sents(), nltk.corpus.ptb.sents())
    ptb_sents = [{'parsed_sent': parsed_sent,
                  'sent_list': sent_list,
                  'sent_string': sent_without_symbols(sent_list),
                  'contains_overt_comp': contains_overt_comp(parsed_sent),
                  'contains_null_comp': contains_null_comp(parsed_sent)}
                 for parsed_sent, sent_list in zipped]
    with open('ptb_data.p', 'wb') as f:
        pickle.dump(ptb_sents, f)


with open('ptb_data.p', 'rb') as f:
    run_tests(pickle.load(f))
