# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

import argparse  
import os
import numpy as np
import string
import re
import pandas as pd

# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size; 255 is a safe upper bound
#
# Note: You may want to add fields for expectations.
class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, vocab_size):
        self.num_states = num_states
        self.vocab_size = vocab_size
        # A
        self.transitions = np.ones((num_states, num_states))/num_states
        # B
        self.emissions = np.zeros((num_states, vocab_size))
        # pi
        self.pi = np.zeros(num_states)
        for i in range(num_states):
            self.pi[i] = np.random.random()
            for j in range(num_states):
                k = np.random.random()
                self.transitions[i, j] = k
            self.transitions[i, :] /= np.sum(self.transitions[i, :], axis = 0)
            self.pi /= self.pi.sum()
        for i in range(num_states):
            for j in range(vocab_size):
                k = np.random.random()
                self.emissions[i, j] = k
            self.emissions[i, :] /= np.sum(self.emissions[i, :], axis = 0)
        print("transitions init:\n", self.transitions)
        print("pi:\n", self.pi)
        print("emissions init:\n", self.emissions)

    def get_alpha(self, observation):
        target = np.where(observation[0] == 1)[0][0]
        alpha = np.zeros((self.num_states, len(observation)))
        ct = [0]*len(observation)
        for i in range(self.num_states):
            alpha[i][0] += self.pi[i] * self.emissions[i][target]
        ct[0] = np.sum(alpha[:,0])
        ct[0] = 1/ct[0]
        print(ct[0])
        alpha[:, 0] = np.multiply(alpha[:, 0], ct[0])
        for T in range(1, len(observation)):
            print(T)
            for i in range(self.num_states):
                for j in range(self.num_states):
                    alpha[i][T] += alpha[i][T-1] * self.transitions[j][i]
                alpha[i][T] *= self.emissions[i][np.where(observation[T] == 1)[0][0]]
            ct[T] = np.sum(alpha[:, T])
            ct[T] = 1/ct[T]
            alpha[:,T] = np.multiply(alpha[:, T], ct[T])
        return alpha, ct
        
    def get_beta(self, observation, ct):
        beta = np.zeros((self.num_states, len(observation)))
        T = len(observation)
        for i in range(self.num_states):
            beta[i][T-1] = 1
        for t in range(T-2, 0,-1):
            for i in range(self.num_states):
                beta[i][t] = 0
                for j in range(self.num_states):
                    beta[i][t] += self.transitions[j][i] * self.emissions[j][np.where(observation[t+1] == 1)[0][0]] * beta[j][t+1]
                beta[:, t] = np.divide(beta[:, t], ct[t+1])
        return beta

    def get_gammas(self, observation, alpha, beta):
        digammaTwo = np.zeros((self.num_states, len(observation)))
        digammaThree = np.zeros((self.num_states, self.num_states, len(observation)))

        for t in range(len(observation) - 1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    digammaThree[i][j][t] = alpha[i][t] * self.transitions[i][j] * self.emissions[j][np.where(observation[t+1]==1)[0][0]] * beta[j][t+1]
                    digammaTwo[i][t] += digammaThree[i][j][t]
        for i in range(self.num_states):
            digammaTwo[i][len(observation)-1] = alpha[i][len(observation)-1]
        return digammaTwo, digammaThree

    def em_step(self, observation, digammaTwo, digammaThree, beta):
        for i in range(self.num_states):
            self.pi[i] = digammaTwo[i][0]
        # re-estimate A
        for i in range(self.num_states):
            denom = 0
            for t in range(len(observation)-1):
                denom += digammaTwo[i][t]
            for j in range(self.num_states):
                numer = 0
                for t in range(len(observation)-1):
                    numer += digammaThree[i][j][t]
                self.transitions[i][j] = numer/denom

        # re-estimate B
        for i in range(self.num_states):
            denom = 0
            for t in range(len(observation)):
                denom += digammaTwo[i][t]
            for j in range(self.vocab_size):
                numer = 0
                for t in range(len(observation)):
                    if(self.emissions[j][np.where(observation[t] == 1)[0][0]]):
                        numer += digammaTwo[i][t]
                beta[j][i] = numer/denom
        return 'a'
    
    # return the avg loglikelihood for a complete dataset (train OR test) (list of arrays)
    def LL(self, dataset):
        pass

    # return the LL for a single sequence (numpy array)
    def LL_helper(self, sample):
        pass

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        pass

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass


    # Save the complete model to a file (most likely using np.save and pickles)
    def save_model(self, filename):
        pass

# Load a complete model from a file and return an HMM object (most likely using np.load and pickles)
def load_hmm(filename):
    pass



# Load all the files in a subdirectory and return a giant list.
def load_subdir(path, maxCount):
    data = []
    count = 0
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
            # for i in data[count]:
            #     if i not in countsOfCharacters.keys():
            #         countsOfCharacters[i] = 1
            #     else:
            #         countsOfCharacters[i] += 1
            fh.close()
            count+=1
            print(count)
            if(count > maxCount):
                return data
    return data


ascii_cols = [' ', '!', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def one_hot_encoding(sample):
    sample = re.sub(r'[^\x20-\x7a]', r'', sample)
    # print(sample)
    sample = re.sub(r'[\x22-\x60]', r'', sample)
    # print(sample)
    sample = ascii_cols + list(sample)
    # print(sample)
    one_hot = pd.get_dummies(sample, columns=ascii_cols)
    # print(one_hot)
    one_hot = one_hot.iloc[len(ascii_cols):-1]
    return one_hot.to_numpy()
    # print(one_hot)
        
# state = 2 (pos or neg)
# N = 2
# T = 12500
# M = 28
# 


def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()
    # 12500
    data = load_subdir('/Users/fires/OneDrive/Desktop/Machine Learning/Project_3/aclImdbNorm/aclImdbNorm/train/pos', 10)
    train_pos = []
    print(args.dev_path)
    for i in data:
        train_pos.append(one_hot_encoding(i))
    print(np.where(train_pos[0][2] == 1)[0][0])
    tester = HMM(5, 28)
    # print(train_pos)
    # for i in range(len(train_pos)):
    alpha, ct = tester.get_alpha(train_pos[0])
    # print(alpha)
    print(ct)
    beta = tester.get_beta(train_pos[0], ct)
    digammaTwo, digammaThree = tester.get_gammas(train_pos[0], alpha, beta)
    log_likelihood = np.sum(np.log(ct))*-1
    a = tester.em_step(train_pos[0], digammaTwo, digammaThree, beta)
    print(log_likelihood)
    # print(len(train_pos[0]))
        
    #12500
    # data1 = load_subdir('/Users/fires/OneDrive/Desktop/Machine Learning/Project_3/aclImdbNorm/aclImdbNorm/train/neg')
    # # #11324
    # data2 = load_subdir('/Users/fires/OneDrive/Desktop/Machine Learning/Project_3/aclImdbNorm/aclImdbNorm/test/neg')
    # #12500
    # data3 = load_subdir('/Users/fires/OneDrive/Desktop/Machine Learning/Project_3/aclImdbNorm/aclImdbNorm/test/pos')
    # values = np.array(data)

    # print(len(data1))
    # print(len(data2))
    # print(len(data3))
    # print(countsOfCharacters)
    # print(len(countsOfCharacters))
    # count = 0
    # for i in countsOfCharacters.values():
    #     count += i
    # for i in countsOfCharacters.keys():
    #     countsOfCharacters[i] /= count
    # print(count)
    # count = 0
    # for i in countsOfCharacters.values():
    #     count += i
    # print(count)
    # lst = list(countsOfCharacters.items())
    # array = np.array(lst)
    # # print(array.shape())
    # print(array)
    # for filename in os.listdir('/Users/fires/OneDrive/Desktop/Machine Learning/Project_3/aclImdbNorm/aclImdbNorm/test/neg'):
    #     print(filename)

    # print(args.train_path)

    # OVERALL PROJECT ALGORITHM:
    # 1. load training and testing data into memory
    #
    # 2. build vocabulary using training data ONLY
    #
    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #
    # 4. output initial loglikelihood on training data and on testing data
    #
    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message

if __name__ == '__main__':
    main()
