'''
Test the Rocchio-tfidf text categorizer.
'''

import os
import dill as pickle
from train import TokenStatistics, InvertedIndex


class RocchioCategorizer:
    ''' Rocchio-tfidf text categorizer. '''

    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def categorize(self, dir_path, doc, outfile):
        ''' Categorize text given a path to a text file. '''
        pass

    def similarity(self, token_list, category):
        ''' Helper function to compute cosine similarity. '''
        pass


if __name__ == '__main__':
    input_filename = input('Please specify the file containing the list of test documents: ')
    output_filename = input('Please specify the name for the output file containing the labeled test documents: ')
    print('Categorizing...')

    textcat = RocchioCategorizer()

    # Loop through documents
    dir_path = os.path.dirname(os.path.abspath(input_filename))
    with open(input_filename, 'r') as test_file_list:
            with open(output_filename, 'wb') as outfile:
                    for doc in test_file_list:
                            textcat._categorize(dir_path, doc.strip(), outfile)
