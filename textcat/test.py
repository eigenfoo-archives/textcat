'''
Test the Rocchio-tfidf text categorizer.
'''

import os
from collections import defaultdict
import dill as pickle
from train import TokenStatistics, InvertedIndex, tokenize


class RocchioCategorizer:
    ''' Rocchio-tfidf text categorizer. '''

    def __init__(self, inverted_index):
        self.ii = inverted_index

    def categorize(self, test_list_path, doc_path, outfile):
        '''
        Helper function to categorize one document and write the
        results to the outfile.
        '''

        # Generate list of tokens for the given document
        token_list = tokenize(os.path.join(test_list_path, doc_path))

        # Compute similarity metric for each of the categories
        similarities = {}
        for category in self.ii.category_count.keys():
            similarities[category] = self.similarity(token_list, category)

        # Pick the category with highest similarity and write results to
        # output file
        label = max(similarities, key=similarities.get)
        print(doc_path + ' ' + label, file=outfile)

    def similarity(self, token_list, category):
        '''
        Helper function to compute cosine similarity.

        Parameters
        ----------
        token_list : array-like
            List of tokens in document

        category : string
            Category to compute similarity to (i.e. compute similarity to this
            category's centroid)
        '''

        doc_tfs = defaultdict(lambda: 0)
        similarity = 0

        # Compute tfs of document
        for token in token_list:
            doc_tfs[token] += 1

        # Compute similarity metric
        for token in doc_tfs:
            if (token in self.ii.inverted_index) \
                    and (category in self.ii.inverted_index[token].tf_dict):
                category_tf = self.ii.inverted_index[token].tf_dict[category]
                doc_tf = doc_tfs[token]
                idf = self.ii.inverted_index[token].idf
                # FIXME Are we already normalized...?
                similarity += category_tf * doc_tf * (idf ** 2)

        return similarity


if __name__ == '__main__':
    test_list_filename = input('Test list file:\t\t')
    model_filename = input('Model checkpoint file:\t')
    output_filename = input('Output file:\t\t')
    print('Categorizing...')

    with open(model_filename, 'rb') as f:
        ii = pickle.load(f)

    textcat = RocchioCategorizer(ii)

    # Loop through documents
    test_list_path = os.path.dirname(os.path.abspath(test_list_filename))

    with open(test_list_filename, 'r') as test_file_list:
        with open(output_filename, 'w') as outfile:
            for doc in test_file_list:
                textcat.categorize(test_list_path, doc.strip(), outfile)
