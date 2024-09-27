"""
author: rohan singh
code for COMS W4705 HW 1
"""

from collections import defaultdict
import math
import os
import os.path
import random
import sys

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
    Given a sequence, this function returns a list of n-grams, where each n-gram is a Python tuple.
    This works for arbitrary values of n >= 1.
    """
    # handling the unigram case
    if n == 1:
        return [("START",)] + [(word,) for word in sequence] + [("STOP",)]

    # adding START and STOP tokens as padding for n >= 2
    padded_sequence = ["START"] * (n - 1) + sequence + ["STOP"]

    # generating n-grams
    ngrams = []
    for i in range(len(padded_sequence) - n + 1):
        ngram = tuple(padded_sequence[i : i + n])
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):
    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # counting ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # total number of tokens (for unigram probability)
        self.total_unigrams = sum(self.unigramcounts.values())

    def count_ngrams(self, corpus):
        """
        Populate dictionaries of unigram, bigram, and trigram counts.
        """
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            unigrams = get_ngrams(sentence.copy(), 1)
            bigrams = get_ngrams(sentence.copy(), 2)
            trigrams = get_ngrams(sentence.copy(), 3)

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
        # counting of the first word (unigram)
        first_word = (bigram[0],)
        if self.unigramcounts[first_word] == 0:
            return 0.0
        return self.bigramcounts[bigram] / self.unigramcounts[first_word]

    def raw_trigram_probability(self, trigram):
        """
        Returns the raw (unsmoothed) trigram probability.
        """
        bigram_context = (trigram[0], trigram[1])

        # if the bigram context count is 0, returning the uniform distribution over vocabulary
        if self.bigramcounts[bigram_context] == 0:
            return 1 / len(self.lexicon)

        return self.trigramcounts[trigram] / self.bigramcounts[bigram_context]

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability using linear interpolation.
        The interpolation parameters lambda1, lambda2, and lambda3 are set to 1/3.
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        w1, w2, w3 = trigram

        # raw probabilities
        trigram_prob = self.raw_trigram_probability((w1, w2, w3))
        bigram_prob = self.raw_bigram_probability((w2, w3))
        unigram_prob = self.raw_unigram_probability((w3,))

        # linear interpolation
        smoothed_prob = (
            lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob
        )

        return smoothed_prob

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

            # randomly selecting words based on probability
            next_word = random.choices(candidates, probabilities)[0]

            if next_word == "STOP":
                break

            # appending the word to the sentence
            sentence.append(next_word)

            # updating the current bigram context
            current_bigram = (current_bigram[1], next_word)

        return sentence

    def sentence_logprob(self, sentence):
        """
        Computes the log probability of an entire sentence based on the trigram model.

        Args:
            sentence (list of str): The sentence represented as a list of tokens.

        Returns:
            float: The log probability of the sentence.
        """

        # adding "START" and "STOP" tokens to the sentence
        sentence = ["START", "START"] + sentence + ["STOP"]

        # getting the trigrams for the sentence
        trigrams = get_ngrams(sentence, 3)

        log_prob = 0.0

        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)

            if prob > 0:
                log_prob += math.log2(prob)
            else:
                # edge case
                log_prob += 0

        return log_prob

    def perplexity(self, corpus):
        """
        Compute the perplexity of the model on the given corpus.
        Corpus is a corpus iterator, as returned by the corpus_reader method.
        """
        total_log_prob = 0.0
        total_word_count = 0

        for sentence in corpus:
            # computing the log probability of the sentence
            sentence_log_prob = self.sentence_logprob(sentence)
            total_log_prob += sentence_log_prob

            total_word_count += len(sentence)

        # calculating average log probability
        average_log_prob = total_log_prob / total_word_count

        perplexity = math.pow(2, -average_log_prob)

        return perplexity


def essay_scoring_experiment(
    train_high_file, train_low_file, test_high_dir, test_low_dir
):
    """
    This method compares the perplexities of two trigram models (high skill and low skill)
    on test essays and returns the accuracy of the predictions.
    """

    # training two trigram models
    high_model = TrigramModel(train_high_file)
    low_model = TrigramModel(train_low_file)

    correct_predictions = 0
    total_predictions = 0

    # evaluating essays in the high skill test set
    for essay_file in os.listdir(test_high_dir):
        with open(os.path.join(test_high_dir, essay_file), "r") as f:
            essay = f.read().split()

        high_perplexity = high_model.perplexity([essay])
        low_perplexity = low_model.perplexity([essay])

        # if the high model perplexity is lower, predict "high"; otherwise predict "low"
        if high_perplexity < low_perplexity:
            correct_predictions += 1  # correctly predicted as high
        total_predictions += 1

    # evaluating essays in the low skill test set
    for essay_file in os.listdir(test_low_dir):
        with open(os.path.join(test_low_dir, essay_file), "r") as f:
            essay = f.read().split()

        high_perplexity = high_model.perplexity([essay])
        low_perplexity = low_model.perplexity([essay])

        # if the low model perplexity is lower, predict "low"; otherwise predict "high"
        if low_perplexity < high_perplexity:
            correct_predictions += 1  # correctly predicted as low
        total_predictions += 1

        # calculating accuracy as the number of correct predictions / total predictions
        accuracy = correct_predictions / total_predictions

        return accuracy


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # python trigram_model.py "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/brown_train.txt" "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/brown_train.txt"
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment(
    #    "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/ets_toefl_data/train_high.txt",
    #    "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/ets_toefl_data/train_low.txt",
    #    "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/ets_toefl_data/test_high",
    #    "/Users/rohansingh/Documents/fall 24/coms 4705/HW/hw1_data/ets_toefl_data/test_low",
    # )
    # print(acc)
