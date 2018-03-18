# TextCat

A simple text categorizer.

## Requirements

* A Python 3.x interpreter (Python 2.x not supported!)
* `pip` package manager (recommended)

## Installation

To get `textcat` up and running, the following code snippet should suffice on a
UNIX terminal:

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
python train.py
```

You will be prompted for:
1. the filename of the list of labelled training documents, and 
2. the filename where you wish the classifier to be saved.

To test:

```
python test.py
```

You will be prompted for:
1. the filename of the list of documents to be categorized, 
2. the filename of the saved classifier,
3. the filename where you wish the results to be written.
