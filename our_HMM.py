from collections import Counter
from os import PathLike
import numpy as np 
from get_list_upos import parse_conllu, Lemma, Upos
import pandas as pd

UNKNOWN_KEYWORD: Lemma = "UNK"
START_TAG: Upos = "START"
END_TAG: Upos = "END"


class our_HMM:
    
    # Vocabulary V (it is formed by the given words)
    # Set of tags Q (it is formed by given PoS tags)
    # Result matrix = matrix with probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Emission matrix = matrix with emission probabilities with len(Q) x len(V) size (tags in rows and words in columns)
    # Transition matrix = matrix with transition probabilities with len(Q)+1 x len(Q)+1 size (tags in rows and columns) (+1 in both because start and stop states have to be taken into account)

    def __init__(self, file_path: PathLike, unk_threshold: int = 0):
        
        self.words: list[Lemma]
        self.tags: list[Upos]
        self.emission: pd.DataFrame
        self.transition: pd.DataFrame

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

    def viterbi_algorithm(self):

        """print(f'Vocabulary: {self.words}')
        print(f'Tags: {self.tags}')
        print(f'Emission features: {self.emission}')
        print(f'Transition features: {self.transition}')
        print(f'Result matrix: {self.result}')"""

        # For each word the tag with the highest probability will be stored
        final_result = []
        
        # for i in words 
        for i in range(len(self.words)):
            # print(f'Word: {self.words[i]}')

            # for j in tags 
            for j in range(len(self.tags)):
                # print(f'Tag: {self.tags[j]}')

                # probability of tag j and word i = 
                #   best probability of the previous word for all tags + 
                #   emission probability of j tag and i word +
                #   transition probability of best previous j tag and actual j tag
                if i != 0:
                    self.result[j,i] = self.previos_max_prob + self.emission[j,i] + self.transition[self.previos_max_prob_index+1,j]

                else: 
                    self.result[j,i] = self.emission[j,i]*self.transition[0,j]
            
            # Calculate the higher probability tag for the previous word 
            self.previos_max_prob = np.max(self.result[:, i-1])
            self.previos_max_prob_index = np.argmax(self.result[:, i-1])
            # print(f'Previous max prob: {self.previos_max_prob}')
            # print(f'Previous max prob tag: {self.tags[self.previos_max_prob_index]}')
            final_result.append([i,self.previos_max_prob_index])
        
        # print(self.result)
        return final_result
    
    def fit(self, file_path: PathLike, unk_threshold: int = 0):
        parsed_file = parse_conllu(file_path)

        # Get the lists of words and tags
        self.tags = list(set([value[1] for sentence in parsed_file for value in sentence])) + [START_TAG, END_TAG]
        
        c = Counter([value[0] for sentence in parsed_file for value in sentence])
        if unk_threshold > 0:
            c -= Counter({k: v for k, v in c.items() if v <= unk_threshold})
            # Add UNK to the vocabulary
            c.update([UNKNOWN_KEYWORD])
        self.words = list(c.keys())
        
        # Emission probabilities
        self.emission = pd.DataFrame(0, index=self.tags, columns=self.words)
        self.transition = pd.DataFrame(0, index=self.tags, columns=self.tags)
        self.transition.drop(index=END_TAG, columns=START_TAG, inplace=True)

        for sentence in parsed_file:
            previousUpos: Upos = START_TAG
            for word, upos in sentence:
                if word in self.words:
                    self.emission.loc[upos, word] += 1
                else:
                    self.emission.loc[upos, UNKNOWN_KEYWORD] += 1
                self.transition.loc[previousUpos, upos] += 1
                previousUpos = upos
            self.transition.loc[previousUpos, END_TAG] += 1

        assert self.transition.loc[START_TAG, END_TAG] == 0, "There should be no transition from START to END"
        assert (self.emission.sum(axis=0) > 0).all(), "Each column in emission should sum > 0"

        # Apply log2 to all columns
        self.emission = np.log2(self.emission)
        print(self.transition)

    
if __name__ == '__main__':

    # Class first example 
    Q = ["N", "V"] # Tag set 
    V = ["the", "can", "fish"] # Sentence

    hmm = our_HMM(Q, V)
    final_result = hmm.viterbi_algorithm()
    print("Class first example")
    for cords in final_result:
        x = cords[0] # word
        y = cords[1] # max prob tag
        print(f'Word: {V[x]}, tag: {Q[y]}')

    # Class second example 
    Q = ["ADJ", "ADV", "DET", "NOUN", "VERB"] # Tag set 
    V = ["the", "aged", "bottle", "flies", "fast"] # Sentence

    hmm = our_HMM(Q, V)
    final_result = hmm.viterbi_algorithm()
    print("Class second example")
    for cords in final_result:
        x = cords[0] # word
        y = cords[1] # max prob tag
        print(f'Word: {V[x]}, tag: {Q[y]}')