import nltk
from nltk.corpus import ptb

import re
import unicodedata
import pickle
import random
from pathlib import Path

detokenizer = nltk.treebank.TreebankWordDetokenizer()

successes = []
false_positives = []
false_negatives = []

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

# sentence as a single string, without weird symbols, punctuation errors
def sent_without_symbols(words):
    sent_lst = [w for w in words if w != '0' and '*' not in w]
    sent = detokenizer.detokenize(sent_lst)
    sent = re.sub(r'(?<=\S)``', ' "', sent) # fix formatting of quotes
    sent = re.sub(r' ?-- ?', '--', sent) # fix formatting of dashes
    sent = re.sub(r' ([\.\,\;\:])', r'\1', sent) # fix formatting of punctuation
    sent = re.sub(r'^\\" ', '"', sent) # fix formatting of quotation marks
    sent = re.sub(r" ''", '"', sent)
    sent = re.sub(r'--" ', '--"', sent)
    sent = re.sub(r',"(?=\w)', '," ', sent)
    sent = re.sub(r'("[\w\s]+")(?=\w)', '\1 ', sent)
    sent = re.sub(r'(\w+")(?=\w)', '\1 ', sent)
    sent = re.sub(r'-LRB-', '(', sent) # fix parentheses
    sent = re.sub(r'-RRB-', ')', sent)
    sent = re.sub(r'\( ', '(', sent)
    sent = re.sub(r' \)', ')', sent)
    sent = re.sub(r'  +', ' ', sent) # fix double spaces
    sent = remove_control_characters(sent) # remove \x01 etc.
    return sent


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

def save_data_individually(dirname, pos_criterion, num_train_pos, num_train_neg,
                           num_test_pos, num_test_neg, seed=1):
    postotal = num_train_pos + num_test_pos
    negtotal = num_train_neg + num_test_neg
    sections = ['/train', '/test']
    labels = ['/pos', '/neg']
    for s in sections:
        for l in labels:
            Path(dirname + s + l).mkdir(parents=True, exist_ok=True)

    with open('ptb_data.p', 'rb') as f:
        orig_data = pickle.load(f)
    random.Random(seed).shuffle(orig_data) # adding a seed number for reproducibility
    poscount = 0
    negcount = 0
    for i, t in enumerate(orig_data):
        if poscount > postotal and negcount > negtotal:
            return
        if pos_criterion(t):
            label = 'pos'
            poscount += 1
            if poscount <= num_train_pos:
                section = 'train'
            elif poscount <= postotal:
                section = 'test'
            else:
                continue
        else:
            label = 'neg'
            negcount += 1
            if negcount <= num_train_neg:
                section = 'train'
            elif negcount <= negtotal:
                section = 'test'
            else:
                continue
        with open(f'{dirname}/{section}/{label}/sentence_{i:07}.txt', 'w') as f:
            f.write(t['sent_string'] + '\n')




generate_treebank_data()
for train_sample in [50, 100, 200, 500]:
    test = 1000
    for i in range(1, 11):
        save_data_individually('comp_overt_' + str(train_sample) + '_' + str(test) + '_seed_' + "{:02d}".format(i), lambda t: t['contains_overt_comp'], train_sample, train_sample, test, test, i)
        save_data_individually('comp_null_' + str(train_sample) + '_' + str(test) + '_seed_' + "{:02d}".format(i), lambda t: t['contains_null_comp'], train_sample, train_sample, test, test, i)