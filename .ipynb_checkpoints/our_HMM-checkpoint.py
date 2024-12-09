from collections import Counter
from os import PathLike
from typing import Literal
import numpy as np
from get_list_upos import parse_conllu, Lemma, Upos
import pandas as pd
import argparse
import evaluate

UNKNOWN_KEYWORD: Lemma = "UNK"
START_TAG: Upos = "START"
END_TAG: Upos = "END"


class our_HMM:

    # Vocabulary: words (it is formed by the given words)
    # Set of tags: tags (it is formed by given PoS tags)
    # Result matrix = matrix with probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Emission matrix = matrix with emission probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Transition matrix = matrix with transition probabilities with len(Q)+1 x len(Q)+1 size (tags in rows and columns) (+1 in both because start and stop states have to be taken into account)

    def __init__(self, file_path: PathLike = None, unk_threshold: int = 1, word_model: Literal["form", "lemma"] = "form"):
        
        self.word_model: Literal["form", "lemma"] = word_model

        self.words: list[Lemma]
        self.tags: list[Upos]
        
        # See the 41th slide for the format of the Dataframes
        self.emission: pd.DataFrame
        self.transition: pd.DataFrame # Index: to, Columns: from

        if file_path is not None:
            self.fit(file_path, unk_threshold)

    def viterbi_algorithm(self, sentence: list[Lemma]) -> tuple[list[tuple[Lemma, Upos]], pd.DataFrame]:

        """print(f'Vocabulary: {self.words}')
        print(f'Tags: {self.tags}')
        print(f'Emission features: {self.emission}')
        print(f'Transition features: {self.transition}')
        print(f'Result matrix: {self.result}')"""

        # A single sentence may use multiple times the same word, so we have to assign word_pos
        sentence_enumarated = [word + "_" + str(i) for i, word in enumerate(sentence)]

        result = pd.DataFrame(0.0, index=self.tags, columns=sentence_enumarated)
        previous_max_prob = 0
        previous_max_prob_upos: Upos = START_TAG

        # For each word the tag with the highest probability will be stored
        final_result = []

        # for i in words
        for i, i_word in enumerate(sentence):
            # print(f'Word: {self.words[i]}')

            using_word = i_word if i_word in self.words else UNKNOWN_KEYWORD # Word taking into account the unknowns
            results_word = sentence_enumarated[i] # Column index for repeated words
            # if using_word == UNKNOWN_KEYWORD:
            #     print(f"UNKNOWN: {i_word}")

            # for j in tags
            for j_upos in self.tags:
                # print(f'Tag: {self.tags[j]}')

                # probability of tag j and word i =
                #   best probability of the previous word for all tags +
                #   emission probability of j tag and i word +
                #   transition probability of best previous j tag and actual j tag
                result.loc[j_upos, results_word] = (previous_max_prob +
                                                    self.emission.loc[j_upos, using_word] +
                                                    self.transition.loc[j_upos, previous_max_prob_upos]
                )

            # Calculate the higher probability tag for the previous word
            previous_max_prob = result.loc[:, results_word].max()
            previous_max_prob_upos = result.loc[:, results_word].idxmax()
            # print(f'Previous max prob: {self.previos_max_prob}')
            # print(f'Previous max prob tag: {self.tags[self.previos_max_prob_index]}')
            final_result.append((i_word, previous_max_prob_upos))

        return (final_result, result)

    def fit(self, file_path: PathLike, unk_threshold: int = 0):
        parsed_file = parse_conllu(file_path, mode=self.word_model)

        # Get the lists of words and tags
        self.tags = list(set([value[1] for sentence in parsed_file for value in sentence]))

        # Remove the words that are not very frequent
        c = Counter([value[0] for sentence in parsed_file for value in sentence])
        c -= Counter({k: v for k, v in c.items() if v < unk_threshold})
        # Add UNK to the vocabulary
        c.update([UNKNOWN_KEYWORD])
        
        self.words = list(c.keys())

        # The probability matrices
        self.emission = pd.DataFrame(0, index=self.tags, columns=self.words, dtype=float)
        self.transition = pd.DataFrame(0, index=self.tags + [END_TAG], columns=self.tags + [START_TAG], dtype=float)

        # Count the situations
        for sentence in parsed_file:
            previousUpos: Upos = START_TAG
            for word, upos in sentence:
                if word in self.words:
                    self.emission.loc[upos, word] += 1
                else:
                    self.emission.loc[upos, UNKNOWN_KEYWORD] += 1

                self.transition.loc[upos, previousUpos] += 1
                previousUpos = upos
            self.transition.loc[END_TAG, previousUpos] += 1

        # A fail safe
        if self.emission.loc[:, UNKNOWN_KEYWORD].sum() == 0:
            self.emission.loc[:, UNKNOWN_KEYWORD] = 1

        assert self.transition.loc[END_TAG, START_TAG] == 0, "There should be no transition from START to END"
        assert (self.emission.sum(axis=0) > 0).all(), "Each column in emission should sum > 0"

        # Normalize the values (if not a more common class will inflate the numbers, it also helps to get set up for log2)
        self.emission = self.emission.div(self.emission.sum(axis=0), axis=1)
        self.transition = self.transition.div(self.transition.sum(axis=0), axis=1)

        # Apply log transformation
        self.emission = self.emission.map(lambda x: np.log2(x) if x > 0 else -np.inf)
        self.transition = self.transition.map(lambda x: np.log2(x) if x > 0 else -np.inf)

    def test(self, file_path: PathLike) -> dict:
        
        def compute_metrics(gold: list[Upos], pred: list[Upos]) -> dict:
            metric1 = evaluate.load("precision")
            metric2 = evaluate.load("recall")
            metric3 = evaluate.load("f1")
            metric4 = evaluate.load("accuracy")

            precision_micro = metric1.compute(predictions=pred, references=gold, average="micro")["precision"]
            precision_macro = metric1.compute(predictions=pred, references=gold, average="macro")["precision"]
            recall_micro = metric2.compute(predictions=pred, references=gold, average="micro")["recall"]
            recall_macro = metric2.compute(predictions=pred, references=gold, average="macro")["recall"]
            f1_micro = metric3.compute(predictions=pred, references=gold, average="micro")["f1"]
            f1_macro = metric3.compute(predictions=pred, references=gold, average="macro")["f1"]
            accuracy = metric4.compute(predictions=pred, references=gold)["accuracy"]

            return {"precision_micro": precision_micro, "precision_macro": precision_macro, "recall_micro": recall_micro, "recall_macro": recall_macro, "f1_micro": f1_micro, "f1_macro": f1_macro, "accuracy": accuracy}

        parsed_file = parse_conllu(file_path, mode=self.word_model)
        
        # We just put all the tags in a continuos list, reducing in that way one dimension
        gold: list[Upos] = []
        pred: list[Upos] = []

        for sentence in parsed_file:
            gold.extend([word_pair[1] for word_pair in sentence])

            lemma_sentence = [word_pair[0] for word_pair in sentence]

            predictions = self.viterbi_algorithm(lemma_sentence)[0]
            predictions = [word_pair[1] for word_pair in predictions]
            pred.extend(predictions)
        
        # Change the gold and pred values for the index values in self.tags
        gold = [self.tags.index(x) for x in gold]
        pred = [self.tags.index(x) for x in pred]

        results = compute_metrics(pred=pred, gold=gold)

        print(results)
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("train_file_path", type=str, help="Path to the file for fitting the HMM")
    parser.add_argument("--unk_threshold", "-u", default=1, type=int, help="Threshold for unknown words, minimum times that must appear to register. Defaults to 1.")
    parser.add_argument("--test_file_path", "-t", type=str, help="Path to the file for testing on the fitted model.")
    parser.add_argument("--word_model", "-w", type=str, default="form", help="Which format to use when parsing words. 'form' and 'lemma' are possible.")
    parser.add_argument("--sentence", "-s", type=str, help="A single sentence to test the model (splitted by spaces)")
    parser.add_argument("--export_model", "-e", help="Export the trasition and emission matrices to a file.", action="store_true")
    args = parser.parse_args()

    hmm = our_HMM(args.train_file_path, unk_threshold=args.unk_threshold, word_model=args.word_model)

    if args.test_file_path is not None:
        hmm.test(args.test_file_path)
    if args.sentence is not None:
        sentencePOS, result = hmm.viterbi_algorithm(args.sentence.split(" "))
        print(result)
        print(sentencePOS)
    if args.export_model is True:
        hmm.transition.to_csv("transition.csv")
        hmm.emission.to_csv("emission.csv")
    if args.sentence is None and args.test_file_path is None:
        example_sentences = [
            "the current Windows NT user must be an administrator for the computer .",
            "the can fish",
            "the aged bottle fly fast",
            "the quick brown fox jump over the lazy dog ."
        ]

        for i in example_sentences:
            print(i)
            print(hmm.viterbi_algorithm(i.split(" "))[0])