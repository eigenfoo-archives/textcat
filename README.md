# TextCat

A simple Rocchio-tfidf text categorizer.

## Requirements

* Python 3 interpreter (Not Python 2!)
* `pip` package manager (recommended)

## Installation

To get `textcat` up and running, the following code snippet should suffice on a
UNIX terminal (note that depending on your Python distribution, you may need to
use `pip3` instead of `pip`, and `python3` instead of `python`):

```
$ git clone https://github.com/eigenfoo/textcat.git
$ cd textcat
$ pip install -r requirements.txt
$ python nltk_download.py
```

This clones `textcat` from my [my GitHub
repository](https://github.com/eigenfoo/textcat), installs all required Python
packages using `pip`, and downloads all required `nltk` packages.

Note that, for the Python packages installed by `pip`, I have specified the
package versions I had on the machine that I developed this program on. It is
likely that the program will still work with more recent versions, but I have
not tested this.

## Usage

To train:

```
python train.py TRAIN_LABEL_FILENAME MODEL_FILENAME
```

where the arguments are, in order:
1. the filename of the list of labelled training documents, and 
2. the filename where you wish the classifier to be saved.

To test:

```
python test.py TEST_LABEL_FILENAME MODEL_FILENAME OUTPUT_FILENAME
```

where the arguments are, in order:
1. the filename of the list of documents to be categorized, 
2. the filename of the saved classifier,
3. the filename where you wish the results to be written.

## Text Preprocessing

Since Rocchio-tfidf is a simple centroid-based categorization technique, it has
no parameters to tune, and no smoothing is required. As such, the manner in
which the text is preprocessing is of primary importance.

The linguistic preprocessing in the final categorizer is as follows:

1. All text is lowercased.
2. The text is tokenized with the Punkt tokenizer.
3. The tokens are part-of-speech tagged with the Averaged Perceptron tagger.
4. The tokens are lemmatized with the WordNet lemmatizer.
5. Any stopwords (using the `nltk` builtin stopword list) are then stripped.

Lowercasing was done more as an act of habit than as a well thought-out
decision.

The tokenizer used is the Punkt tokenizer. Several tokenizers were considered
(e.g. the Stanford Tokenizer and Penn Treebank Tokenizer). The Stanford
Tokenizer is an improvement on the Penn Treebank Tokenizer, so the latter was
not tested. The Stanford Tokenizer appeared to give similar performance to the
Punkt tokenizer, and had a significantly more complicated interface (the Punkt
tokenizer is the default in `nltk`), so the Punkt tokenizer was used.

The part-of-speech tagger used is the Averaged Perceptron Tagger. Tagging
provided small boosts in performance, so it was adopted. The Averaged Perceptron
Tagger is [a state-of-the-art
tagger](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python), and is
also the default part-of-speech tagger in `nltk`.

Lemmatization was chosen instead of stemming, as several unrelated words may be
stemmed to the same token: lemmatization avoids this problem, at the cost of
a greater computation load (which for the present application is not a problem).
Having decided on lemmatization, there is only one lemmatizer in `nltk`: the
WordNet lemmatizer.

Stop words were filtered using the built-in `nltk` stopword list, again, more as
an action of habit than as a well thought-out decision. In any case, stopwords
usually have low tf-idf values, and would not contribute much to the cosine
similarity metric in any case.
