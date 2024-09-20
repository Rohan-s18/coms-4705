import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, "r") as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """

    ngrams = []

    if n == 1:
        ngrams.append(tuple(["START"]))
        for word in sequence:
            ngrams.append(tuple([word]))
        ngrams.append(tuple(["STOP"]))
        return ngrams

    sequence.append("STOP")
    seq = ["START" for _ in range(n - 1)]
    for word in sequence:
        seq.append(word)

    for i in range(0, len(seq) - (2 * n - 3), 1):
        sub = [word for word in seq[i : i + n]]
        sub = tuple(sub)
        ngrams.append(sub)

    return ngrams


class TrigramModel(object):
    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        # initializing count dictionaries
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:

            # generating ngrams for the current sentence
            unigrams = get_ngrams(sentence.copy(), 1)
            bigrams = get_ngrams(sentence.copy(), 2)
            trigrams = get_ngrams(sentence.copy(), 3)

            # updating counts
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
            
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        if self.total_unigrams == 0:
            return 0.0
        return self.unigramcounts[unigram] / self.total_unigrams

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability.
        """
        # Count of the first word (unigram)
        first_word = (bigram[0],)
        if self.unigramcounts[first_word] == 0:
            return 0.0
        return self.bigramcounts[bigram] / self.unigramcounts[first_word]

    def raw_trigram_probability(self, trigram):
        """
        Returns the raw (unsmoothed) trigram probability.
        """
        bigram_context = (trigram[0], trigram[1])
        
        # If the bigram context count is 0, return uniform distribution over vocabulary
        if self.bigramcounts[bigram_context] == 0:
            return 1 / len(self.lexicon)
        
        return self.trigramcounts[trigram] / self.bigramcounts[bigram_context]


    def generate_sentence(self, t=20):
        """
        Generate a random sentence from the trigram model.
        Starts with ("START", "START") and generates the next word
        based on the trigram probabilities until the "STOP" token is generated,
        or the maximum sentence length `t` is reached.
        """
        sentence = []
        current_bigram = ("START", "START")

        for _ in range(t):
            
            candidates = []
            probabilities = []
            
            # populating candidates and their corresponding trigram probabilities
            for word in self.lexicon:
                trigram = (current_bigram[0], current_bigram[1], word)
                prob = self.raw_trigram_probability(trigram)
                
                if prob > 0:
                    candidates.append(word)
                    probabilities.append(prob)

            if not candidates:
                break

            # randomly selecting the next word based on trigram probabilities
            next_word = random.choices(candidates, probabilities)[0]

            # stop generation if we reach the "STOP" token
            if next_word == "STOP":
                break

            sentence.append(next_word)

            # updating the current bigram context
            current_bigram = (current_bigram[1], next_word)

        return sentence

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability using linear interpolation.
        The interpolation parameters lambda1, lambda2, and lambda3 are set to 1/3.
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        
        w1, w2, w3 = trigram
        
        # Raw probabilities
        trigram_prob = self.raw_trigram_probability((w1, w2, w3))
        bigram_prob = self.raw_bigram_probability((w2, w3))
        unigram_prob = self.raw_unigram_probability((w3,))
        
        # Linear interpolation of trigram, bigram, and unigram probabilities
        smoothed_prob = (
            lambda1 * trigram_prob +
            lambda2 * bigram_prob +
            lambda3 * unigram_prob
        )
        
        return smoothed_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        return float("inf")


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        # ..

    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        # ..

    return 0.0


if __name__ == "__main__":
    # model = TrigramModel(sys.argv[1])
    generator = corpus_reader(
        "/Users/rohansingh/github_repos/coms-4705/HW-1/hw1_data/brown_train.txt"
    )
    

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
