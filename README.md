# Practical Group Exercise

In this exercise an HMM model + Viterbi is implemented. The model is trained with a conllu file and tested with another one. **You can find the analysis of the results in the analysis.ipynb file.** In case you want to use the model separately, you can use the our_HMM.py file (command line usage is explained below).

> Authors: *Andrea Jaunarena*, *Markel Ferro*, *Juliana Planas* and *Lukas Arana*.

### Implementing an HMM Pos Tagger 

### Instalation

```bash
python -m venv .venv
source .venv/bin/activate # Linux ¿and MacOS?
source .\.venv\Scripts\activate.bat # Windows

pip install -r requirements.txt
```

### Usage

You must have a file in conllu format to fit the model. That will be the first parameter. Apart from that you have various options.

#### Use a test file

In this mode you test your model in another conllu file. It is passed as `-t`.

```bash
python our_HMM.py <YOUR_TRAIN_FILE>.conllu -t <YOUR_TEST_FILE>.conllu
```

Output:

```python
{'precision_micro': 0.8136248775204992, 'precision_macro': 0.7852392072793883, 'recall_micro': 0.8136248775204992, 'recall_macro': 0.612734531152367, 'f1_micro': 0.8136248775204992, 'f1_macro': 0.6513179995385716, 'accuracy': 0.8136248775204992}
```

#### Test a single sentence

This mode is useful for analyzing how the model performed in a single sentence. Provide the sentence with `-s` (please, separate the symbols like dots with a space).

```bash
python our_HMM.py <YOUR_FILE>.conllu -s "the quick brown fox jump over the lazy dog ."
```

Output:

```
          the_0   quick_1   brown_2      fox_3     jump_4     over_5      the_6     lazy_7      dog_8        ._9
SYM        -inf      -inf      -inf       -inf       -inf       -inf       -inf       -inf       -inf       -inf
NUM        -inf      -inf      -inf -21.279377 -25.656326       -inf       -inf -37.533384       -inf       -inf
NOUN       -inf      -inf      -inf -13.719044 -21.486401       -inf       -inf -30.930994 -34.528426       -inf
X          -inf      -inf      -inf       -inf -28.656326       -inf       -inf -40.118346       -inf       -inf
INTJ       -inf      -inf      -inf       -inf -29.656326       -inf       -inf       -inf       -inf       -inf
ADJ        -inf -4.772738 -8.846183 -17.089552 -25.132764       -inf       -inf -32.229603       -inf       -inf
PRON       -inf      -inf      -inf -19.578937 -22.633958       -inf       -inf -37.417907       -inf       -inf
AUX        -inf      -inf      -inf -22.279377 -21.792140       -inf       -inf       -inf       -inf       -inf
CCONJ      -inf      -inf      -inf -17.755815 -21.804577       -inf       -inf       -inf       -inf       -inf
ADV        -inf      -inf      -inf -19.279377 -22.928405 -25.670105       -inf -37.030884       -inf       -inf
DET   -2.841302      -inf      -inf -19.694414 -25.264008       -inf -26.128242 -37.658915       -inf       -inf
PROPN      -inf      -inf      -inf -19.191914 -24.230061       -inf       -inf -34.708955       -inf       -inf
VERB       -inf      -inf      -inf -19.819945 -21.563569       -inf       -inf -37.533384       -inf       -inf
_          -inf      -inf      -inf -21.694414 -25.568863       -inf       -inf -37.118346       -inf       -inf
ADP        -inf      -inf      -inf -16.921825 -19.954153 -24.868556       -inf       -inf       -inf       -inf
SCONJ      -inf      -inf      -inf -19.279377 -23.983900       -inf       -inf       -inf       -inf       -inf
PART       -inf      -inf      -inf -18.372486 -23.230061       -inf       -inf       -inf       -inf       -inf
PUNCT      -inf      -inf      -inf -16.060208 -19.534792       -inf       -inf       -inf       -inf -36.174249
[('the', 'DET'), ('quick', 'ADJ'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jump', 'PUNCT'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'NOUN'), ('dog', 'NOUN'), ('.', 'PUNCT')]
```

#### Export the matrices

With the `-e` option you may export the emmision and transition matrices generated from the train file to a .csv

```bash
python our_HMM.py <YOUR_FILE>.conllu -e
```