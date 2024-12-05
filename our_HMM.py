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

    # Vocabulary V (it is formed by the given words)
    # Set of tags Q (it is formed by given PoS tags)
    # Result matrix = matrix with probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Emission matrix = matrix with emission probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Transition matrix = matrix with transition probabilities with len(Q)+1 x len(Q)+1 size (tags in rows and columns) (+1 in both because start and stop states have to be taken into account)

    def __init__(self, file_path: PathLike = None, unk_threshold: int = 0, word_model: Literal["form", "lemma"] = "form"):
        
        self.word_model: Literal["form", "lemma"] = word_model

        self.words: list[Lemma]
        self.tags: list[Upos]
        
        # See the 41th slide for the format of the Dataframes
        self.emission: pd.DataFrame
        self.transition: pd.DataFrame # Index: to, Columns: from

        if file_path is not None:
            self.fit(file_path, unk_threshold)

        """
        self.tags = Q # tags in the given tag set
        self.words = V # words in the given sentence

        # Result probabilities
        self.result = np.zeros((len(self.tags), len(self.words)))

        # Emission probabilities
        self.emission = np.random.rand(len(self.tags), len(self.words))

        # Transition probabilities
        self.transition = np.random.rand(len(self.tags)+1, len(self.tags)+1) #

        # Previos word maximum probability for all tags
        self.previos_max_prob = 0
        # Previos word maximum probability's index for all tags
        self.previos_max_prob_index = 0
        """

    def viterbi_algorithm(self, sentence: list[Lemma]) -> tuple[list[tuple[Lemma, Upos]], pd.DataFrame]:

        """print(f'Vocabulary: {self.words}')
        print(f'Tags: {self.tags}')
        print(f'Emission features: {self.emission}')
        print(f'Transition features: {self.transition}')
        print(f'Result matrix: {self.result}')"""

        # A single sentence may use multiple times the same word, so we have to assign word_pos
        sentence_enumarated = [word + "_" + str(i) for i, word in enumerate(sentence)]

        result = pd.DataFrame(0, index=self.tags, columns=sentence_enumarated)
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

        c = Counter([value[0] for sentence in parsed_file for value in sentence])
        if unk_threshold > 0:
            c -= Counter({k: v for k, v in c.items() if v <= unk_threshold})
            # Add UNK to the vocabulary
            c.update([UNKNOWN_KEYWORD]) # TODO: If the train is done with no UNK and the test has unknowns it will raise an error
        self.words = list(c.keys())

        # Emission probabilities
        self.emission = pd.DataFrame(0, index=self.tags, columns=self.words)
        self.transition = pd.DataFrame(0, index=self.tags + [END_TAG], columns=self.tags + [START_TAG])

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

        assert self.transition.loc[END_TAG, START_TAG] == 0, "There should be no transition from START to END"
        assert (self.emission.sum(axis=0) > 0).all(), "Each column in emission should sum > 0"

        # TODO: Apply log transformation to columns (and then use the minimum in virbeti)

    def test(self, file_path: PathLike) -> dict:
        parsed_file = parse_conllu(file_path, mode=self.word_model)
        
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

        clf_metrics = evaluate.combine(["accuracy"])
        results = clf_metrics.compute(predictions=pred, references=gold)

        print(results)
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("train_file_path", type=str, help="Path to the file for fitting the HMM")
    parser.add_argument("--unk_threshold", "-u", default=0, type=int, help="Threshold for unknown words")
    parser.add_argument("--test_file_path", "-t", type=str, help="Path to the file for testing on the fitted model.")
    parser.add_argument("--word_model", "-w", type=str, default="form", help="Which format to use when parsing words. 'form' and 'lemma' are possible.")
    args = parser.parse_args()

    hmm = our_HMM(args.train_file_path, unk_threshold=args.unk_threshold, word_model=args.word_model)

    if args.test_file_path is not None:
        hmm.test(args.test_file_path)
    else:
        example_sentences = [
            "the current Windows NT user must be an administrator for the computer .",
            "the can fish",
            "the aged bottle fly fast",
            "the quick brown fox jump over the lazy dog ."
        ]

        for i in example_sentences:
            print(i)
            print(hmm.viterbi_algorithm(i.split(" "))[0])

    if args.test_file_path is not None:
        hmm.test(args.test_file_path)
