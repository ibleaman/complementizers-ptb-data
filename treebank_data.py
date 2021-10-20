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
    for subtree in parsed.subtrees():
        if subtree.label() == 'SBAR' and len(subtree) > 1:
            for daughter in subtree:
                if daughter.label() == 'IN' and daughter[0] == 'that':
                    return True
    return False

def contains_null_comp(parsed):
    for subtree in parsed.subtrees():
        if subtree.label() == 'SBAR' and len(subtree) > 1:
            for daughter_pair in nltk.bigrams(subtree):
                if daughter_pair[0].label() == '-NONE-' and daughter_pair[0][0] == '0' and \
                not (daughter_pair[1].label() == 'S' and '*T*' in daughter_pair[1].leaves()[0]):
                    return True
    return False


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
