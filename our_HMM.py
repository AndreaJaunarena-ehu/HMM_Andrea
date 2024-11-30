import numpy as np 
class our_HMM:
    # Vocabulary V (it is formed by the given words)
    # A set of N states Q = q1, q2, ..., qN
    # A sequence of T observations O = o1, o2, ..., oN 
    # A transition probability matrix A = a11, ..., aij, aNN
    # A sequence of observation likelihoods B = bi(ot)
    # A initial probability distribution over states π = π1, π2, ..., πN

    def __init__(self, Q, V):
        
        self.tags = Q # tags
        self.words = V # words in the given sentence  
        
        # result probabilities
        self.result = np.random.zeros((len(self.tags), len(self.words)))
        
        # emission probabilities 
        self.emission = np.random.rand((len(self.tags), len(self.words)))
        
        # transition probabilities 
        self.transition = np.random.rand((len(self.tags)+1, len(self.tags)+1))

    def viterbi_algorithm(self):

        print(f'Vocabulary: {self.words}')
        print(f'Tags: {self.tags}')
        print(f'Emission features: {self.emission}')
        print(f'Transition features: {self.transition}')
        print(f'Result matrix: {self.result}')

        final_result = []
        # for i in tags 
        for i in range(len(self.tags)):
            # for j in words 
            for j in range(len(self.words)):
                # probability of tag i and word j = 
                #   best probability of the previous word for all tags
                #   emission probability of i tag and j word
                #   transition probability of best previos i tag and actual i tag
                if j != 0:
                    previos_max_prob = np.max(self.result[:, j-1])
                    previos_max_prob_index = np.argmax(self.result[:, j-1])
                    self.result[i,j] = previos_max_prob + self.emission[i,j] + self.transition[previos_max_prob_index+1,i]

                else: 
                    self.result[i,j] = self.emission[i,j]*self.transition[0,i]
        
        # Add final probability 
        previos_max_prob = np.max(self.result[:, len(self.words)-1])
        previos_max_prob_index = np.argmax(self.result[:, len(self.words)-1])
        stop_result = self.transition[previos_max_prob_index+1,len(self.tags)-1] + previos_max_prob
        for w in range(len(self.words)):
            max = np.argmax(self.result[:,i])
            final_result.append(self.tags[max])
        
        return final_result
    
def main(): 

    Q = ['N', 'V']
    V = ["they", "can", "fish"]

    hmm = our_HMM(Q, V)
    final_result = hmm.viterbi_algorithm()
    print(f'Given the following vocabulary: {V}')
    print(f'That is the result: {final_result}')