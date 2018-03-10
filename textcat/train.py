'''
Train a Rocchio-tfidf text categorizer.
'''

import os
import math
from collections import defaultdict
import pickle


class TokenStatistics():
    ''' Helper class that stores the tf, idf and number of documents
    for a token. '''
    def __init__(self):
        self.tf_dict = {}             # Key, value = category, tf
        self.num_docs_with_token = 0  # Number of documents containing token
        self.idf = 0                  # idf


class InvertedIndex():
    ''' Class that implements inverted index for Rocchio-tfidf. '''

    def __init__(self):
        # key, value = token, TokenStatistics
        self.inverted_index = defaultdict(lambda: TokenStatistics())
        # key, value = category, number of docs in category
        self.category_count = defaultdict(lambda: 0)
        # total number of docs in training corpus
        self.num_documents = 0

    def compute_tfidfs(self, train_labels_filename):
        ''' Computes tf values for training corpus. '''
        train_dir_absolute_path = \
            os.path.dirname(os.path.abspath(train_labels_filename))

        with open(train_labels_filename, 'r') as f:
            for line in f:
                article_relative_path, category = line.split()

                token_list = tokenize(os.path.join(train_dir_absolute_path,
                                                   article_relative_path))

                for token in token_list:
                    self.inverted_index[token].tf_dict[category] += 1

                for token in set(token_list):
                    self.inverted_index[token].doc_count += 1

                self.num_documents += 1

                self.category_count[category] += 1

        for token in self.inverted_index.keys():
            self.inverted_index[token].idf = \
                math.log(self.num_documents /
                         self.inverted_index[token].num_docs_with_token)

    def normalize_tfidfs(self):
        ''' Normalizes tf scores for each document in each category. '''
        normalization_constants = defaultdict(lambda: 0)

        # Compute sum squared of tfidfs, per category
        for token, token_stats in self.inverted_index.items():
            for category, tf in token_stats.tf_dict.items():
                weight = tf * token_stats.idf
                normalization_constants[category] += weight**2

        # Compute square root of sum of squares of tfidfs
        for category, sum_sqr_tfidfs in normalization_constants.items():
            normalization_constants[category] = math.sqrt(sum_sqr_tfidfs)

        # Normalize
        for token, token_stats in self.inverted_index.items():
            for category in token_stats.tf_dict.keys():
                self.inverted_index[token].tf_dict[category] /= \
                    normalization_constants[category]

    def save(self, filename):
        ''' Serializes InvertedIndex instance as a pickle object. '''
        with open(filename, 'w') as f:
            pickle.dump(self, f)


def tokenize(file_path):
    ''' Takes article path, and returns list of tokens. '''
    return ['foobar']


if __name__ == '__main__':
    train_labels_filename = input('Train labels file:')
    model_filename = input('Train labels file:')

    print('Training text categorizer...')

    inverted_index = InvertedIndex()

    inverted_index.compute_tfidfs(train_labels_filename)
    print('Computed tf-idfs.')

    inverted_index.normalize_tfidfs()
    print('Normalized tf-idfs.')

    inverted_index.save(model_filename)
