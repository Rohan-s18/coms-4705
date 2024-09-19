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

        unigrams = get_ngrams(corpus, 1)
        bigrams = get_ngrams(corpus, 2)
        trigrams = get_ngrams(corpus, 3)

        unique_unigrams = set(unigrams)
        unique_bigrams = set(bigrams)
        unique_trigrams = set(trigrams)

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        for unigram in unique_unigrams:
            ct = 0
            for word in unigrams:
                if word == unigram:
                    ct += 1
            self.unigramcounts[unigram] = ct

        for bigram in unique_bigrams:
            ct = 0
            for word in bigrams:
                if word == bigram:
                    ct += 1
            self.bigramcounts[bigram] = ct

        for trigram in unique_trigrams:
            ct = 0
            for word in trigrams:
                if word == trigram:
                    ct += 1
            self.trigramcounts[trigram] = ct

        ##Your code here

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return 0.0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return 0.0

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        return 0.0

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
    corp_sentances = [sentance for sentance in generator]

    sentance = max(corp_sentances)
    print(f"\nsentance: {sentance}\n")

    monograms = get_ngrams(sequence=sentance, n=1)
    bigrams = get_ngrams(sequence=sentance, n=2)
    trigrams = get_ngrams(sequence=sentance, n=3)
    quadgrams = get_ngrams(sequence=sentance, n=4)

    print(f"\nmonograms: {monograms}\n")
    print(f"\nbigrams: {bigrams}\n")
    print(f"\ntrigrams: {trigrams}\n")
    print(f"\nquadgrams: {quadgrams}\n")

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
