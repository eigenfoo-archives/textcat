'''
Train a Rocchio-tfidf text categorizer.
'''

import os
import sys
import math
from collections import defaultdict
import dill as pickle
import nltk
from nltk.corpus import wordnet
from nltk.tag.util import tuple2str


class TokenStatistics:
    '''
    Helper class that stores the tf, idf and number of documents for a token.
    Something akin to a C++ struct would have been preferable to a full-fledged
    class, but no suitable alternative was found (e.g. collections.namedtuple
    is immutable).

    Attributes
    ----------
    tf_dict : collections.defaultdict
        Dictionary (a.k.a. hash map) where keys are category strings and values
        are term frequencies. Default value is 0.

    num_docs_with_token : int
        Number of documents with this particular token

    idf : float
        Inverse document frequency

    Notes
    -----
    Because Rocchio-tfidf is implemented, only the term frequencies for each
    token _for each category_ need to be stored, not the term frequencies for
    each token _for each document_.

    '''
    def __init__(self):
        self.tf_dict = defaultdict(lambda: 0)
        self.num_docs_with_token = 0
        self.idf = 0


class InvertedIndex:
    '''
    Class that implements an "inverted index" for Rocchio-tfidf (see Notes
    below).

    Attributes
    ----------
    inverted_index : collections.defaultdict
        Dictionary (a.k.a. hash map) where keys are token strings, and values
        are TokenStatistics.

    category_count : collections.defaultdict
        Dictionary (a.k.a. hash map) where keys are category strings, and
        values are number of documents in this category.

    num_documents : int
        Total number of documents in the training corpus.

    Notes
    -----
    As above, because Rocchio-tfidf is implemented, only the term frequencies
    for each token _for each category_ need to be stored, not the term
    frequencies for each token _for each document_.
    '''
    def __init__(self):
        self.inverted_index = defaultdict(lambda: TokenStatistics())
        self.category_count = defaultdict(lambda: 0)
        self.num_documents = 0

    def compute_tfidfs(self, train_labels_filename):
        '''
        Computes tf and idf values for training corpus.
        
        Parameters
        ----------
        train_labels_filename : string
            Filename of list of labelled training documents.
        '''
        train_dir_absolute_path = \
            os.path.dirname(os.path.abspath(train_labels_filename))

        with open(train_labels_filename, 'r') as f:
            for line in f:
                # Tokenize article
                article_relative_path, category = line.split()
                token_list = tokenize(os.path.join(train_dir_absolute_path,
                                                   article_relative_path))

                # Increment tf values for each token
                for token in token_list:
                    self.inverted_index[token].tf_dict[category] += 1

                # Increment number of documents with token
                for token in set(token_list):
                    self.inverted_index[token].num_docs_with_token += 1

                # Increment number of documents
                # and number of documents in categoy
                self.num_documents += 1
                self.category_count[category] += 1

        # Compute idfs
        for token in self.inverted_index.keys():
            self.inverted_index[token].idf = \
                math.log(self.num_documents
                         / self.inverted_index[token].num_docs_with_token)

    def normalize_tfidfs(self):
        ''' Normalizes tf scores for each document in each category.  '''
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
        '''
        Serializes InvertedIndex instance as a pickle object.

        Parameters
        ----------
        filename : string
            Filename to save InvertedIndex object to.
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def tokenize(file_path):
    '''
    Helper function to preprocess and tokenize articles.

    1) Lowercase
    2) Tokenize using Punkt tokenizer
    3) Part-of-speech tag using Averaged Perceptron tagger
    4) Lemmatize using WordNet lemmatizer
    5) Filter out stopwords

    Parameters
    ----------
    file_path : string
        Path to article to be tokenized.
    '''
    tokens = []
    stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    with open(file_path, 'r') as f:
        data = f.read().lower()

    for sent in nltk.sent_tokenize(data):
        for word in nltk.word_tokenize(sent):
            tokens += [word]

    tagged_tokens = nltk.pos_tag(tokens)

    if len(tagged_tokens) <= 100:
        # Assume that this document is from corpus 2
        # Strip proper nouns and cardinal numbers
        tokens = [tuple2str((lemmatizer.lemmatize(token, penn_to_wordnet(tag)),
                             tag))
                  for token, tag in tagged_tokens
                  if (token not in stopwords
                      and tag not in ['NNP', 'NNPS', 'CD'])]
    else:
        tokens = [tuple2str((lemmatizer.lemmatize(token, penn_to_wordnet(tag)),
                             tag))
                  for token, tag in tagged_tokens if token not in stopwords]

    return tokens


def penn_to_wordnet(tag):
    ''' Helper function to convert Penn Treebank tagset to WordNet tagset. '''
    if tag in ['JJ', 'JJR', 'JJS']:
        return wordnet.ADJ
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wordnet.NOUN
    elif tag in ['RB', 'RBR', 'RBS']:
        return wordnet.ADV
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wordnet.VERB
    return wordnet.NOUN


def main():
    ''' Driver function for training program. '''
    try:
        train_labels_filename = sys.argv[1]
        model_filename = sys.argv[2]
    except IndexError:
        msg = ("Usage: python train.py TRAIN_LABELS_FILENAME MODEL_FILENAME"
               "\n\tTRAIN_LABELS_FILENAME: name of file with list of labelled "
               "training documents\n\tMODEL_FILENAME: name of file where "
               "model should should be saved")
        print(msg)
        return

    print('Training text categorizer...')

    inverted_index = InvertedIndex()

    inverted_index.compute_tfidfs(train_labels_filename)
    print('Computed tf-idfs.')

    inverted_index.normalize_tfidfs()
    print('Normalized tf-idfs.')

    inverted_index.save(model_filename)
    print('Saved model checkpoint.')

    print('Success.')


if __name__ == '__main__':
    main()
